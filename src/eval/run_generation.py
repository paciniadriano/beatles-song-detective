"""Runs the whole agent on the generation set and writes each answer, with
its automatic checks, to a YAML file a person grades by hand
(CLAUDE.md, "Evaluation: two separate sets").

    python -m src.eval.run_generation --limit 5              # trial: prints, saves nothing
    python -m src.eval.run_generation                         # one run, settings from config.yaml
    python -m src.eval.run_generation --max-steps 2 --reuse-grades RUN_ID
    python -m src.eval.run_generation --closed-book           # the model alone, no retrieval
    python -m src.eval.generation_report RUN_ID               # once graded

Each run writes eval/generation/runs/<run_id>/answers.yaml. A follow-up
question first asks its `previous_question`, whose answer only builds the
history; the second answer is the one checked and graded.

--reuse-grades copies the grades of an earlier run onto every answer that is
word for word the same, so only the answers that changed are graded again.

This spends the developer's key (OPENAI_API_KEY in .env): about $0.03
per full run. The max_steps pre-check needs the grades, so generation_report
prints it."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import yaml

from src.config import Config
from src.eval.generation_checks import Checks, check_answer
from src.eval.metrics import seeded_sample
from src.eval.question_file import load_question_set
from src.index.collections import collection_name
from src.ingest.normalize import load_documents
from src.rag.agent import AgentEvent, Done, Turn, answer, load_prompt
from src.rag.llm import LLMClient, Msg, TokenUsage, api_key_from_env, complete, make_client
from src.rag.search import Searcher, open_searcher
from src.rag.tools import cited_numbers
from src.schemas import PATH_SEPARATOR, Chunk, EvalQuestion, Section

ANSWERS_FILE = "answers.yaml"

GRADING_INSTRUCTIONS = """\
# Answers of one run of the generation set. Grade only the entries that have
# `correct` and `faithful` (the answerable questions); the others pass or fail
# on `checks.declined` alone.
#   correct:  true if the answer says what `expected` says and contradicts
#             nothing in it.
#   faithful: true if every sentence is supported by the passage it cites,
#             whichever of the sources that is: read `cited_passages`, not
#             `evidence`. An uncited factual sentence, or a true fact the
#             cited passages do not state, is false. `checks.evidence_cited`
#             says whether the evidence passage was among those cited.
#   note:     why, when the case is not clear-cut.
# Then run: python -m src.eval.generation_report <run_id>
"""

CLOSED_BOOK_INSTRUCTIONS = """\
# Closed-book answers: the model alone, with no retrieval. For each entry that
# has `closed_book_correct`, set it to true if the answer has the key fact of
# `expected` and contradicts nothing; false if it is wrong, misses the key
# fact or says it does not know. `note` when it is not clear-cut.
# Then pass this run to: python -m src.eval.generation_report <run_id> --closed-book <this run_id>
"""


# --- asking the agent ------------------------------------------------------------

def last_done(events: Iterable[AgentEvent]) -> Done:
    """Runs the agent's events to the end and keeps the final one."""
    done = None
    for event in events:
        if isinstance(event, Done):
            done = event
    if done is None:
        raise RuntimeError("the agent ended without a Done event")
    return done


def ask(
    question: EvalQuestion, searcher: Searcher, client: LLMClient, config: Config, system_prompt: str
) -> tuple[Done, TokenUsage]:
    """The agent's answer to the question, after its previous question if it
    is a follow-up. The usage covers both turns: both were paid for."""
    history: list[Turn] = []
    usage = TokenUsage()
    if question.previous_question:
        first = last_done(answer(question.previous_question, [], searcher, client, config.llm, system_prompt))
        history = [Turn(question.previous_question, first.answer)]
        usage.add(first.usage)
    done = last_done(answer(question.question, history, searcher, client, config.llm, system_prompt))
    usage.add(done.usage)
    return done, usage


def ask_closed_book(question: EvalQuestion, client: LLMClient, config: Config, prompt: str) -> tuple[str, TokenUsage]:
    """The model's own answer, with no retrieval. A follow-up gets its first
    turn the same way, so the second question makes sense."""
    llm = config.llm
    usage = TokenUsage()
    messages = [Msg("system", prompt)]
    if question.previous_question:
        messages.append(Msg("user", question.previous_question))
        first, first_usage = complete(client, llm.model, messages, llm.max_output_tokens, llm.extra_params)
        messages.append(Msg("assistant", first))
        usage.add(first_usage)
    messages.append(Msg("user", question.question))
    text, second_usage = complete(client, llm.model, messages, llm.max_output_tokens, llm.extra_params)
    usage.add(second_usage)
    return text, usage


# --- what gets written -------------------------------------------------------------

def source_label(number: int, chunk: Chunk) -> str:
    """[3] Hey Jude > Recording: enough to find the passage while grading."""
    return f"[{number}] {chunk.title}{PATH_SEPARATOR}{chunk.section_path}"


def cited_passages(answer: str, chunks: list[Chunk]) -> dict[str, str]:
    """The text of every passage the answer cites, keyed by its label: what
    `faithful` is judged against. Only the cited ones, not all eight, so the
    file stays readable; a number with no passage is left out (and fails
    `citations_exist`)."""
    passages = {}
    for number in cited_numbers(answer):
        if 1 <= number <= len(chunks):
            passages[source_label(number, chunks[number - 1])] = chunks[number - 1].text
    return passages


def question_fields(question: EvalQuestion, doc_titles: dict[str, str]) -> dict[str, object]:
    """What the grader needs to know about the question itself."""
    fields: dict[str, object] = {"question_id": question.question_id, "kind": question.kind}
    if question.previous_question:
        fields["previous_question"] = question.previous_question
    fields["question"] = question.question
    fields["expected"] = question.answer
    if question.answerable:
        fields["article"] = doc_titles[question.doc_id]
        fields["evidence"] = question.evidence
    return fields


def answer_entry(
    question: EvalQuestion, doc_titles: dict[str, str], done: Done, checks: Checks, usage: TokenUsage, seconds: float
) -> dict[str, object]:
    entry = question_fields(question, doc_titles)
    entry["answer"] = done.answer
    entry["cited_passages"] = cited_passages(done.answer, done.sources.chunks)
    entry["queries"] = done.queries
    entry["sources"] = [source_label(number, chunk) for number, chunk in enumerate(done.sources.chunks, start=1)]
    entry["checks"] = checks.model_dump()
    entry["tokens"] = {"input": usage.input_tokens, "cached": usage.cached_input_tokens, "output": usage.output_tokens}
    entry["seconds"] = round(seconds, 1)
    if question.answerable:
        entry.update({"correct": None, "faithful": None, "note": ""})
    return entry


def closed_book_entry(question: EvalQuestion, doc_titles: dict[str, str], text: str) -> dict[str, object]:
    """Questions the sources cannot answer are not graded closed-book."""
    entry = question_fields(question, doc_titles)
    entry["answer"] = text
    if question.answerable:
        entry.update({"closed_book_correct": None, "note": ""})
    return entry


def reuse_grades(entries: list[dict], earlier: list[dict]) -> int:
    """Copies grades from `earlier` onto every entry whose answer is word for
    word the same. Returns how many were copied."""
    earlier_by_id = {entry["question_id"]: entry for entry in earlier}
    copied = 0
    for entry in entries:
        before = earlier_by_id.get(entry["question_id"])
        if "correct" not in entry or before is None or before.get("answer") != entry["answer"]:
            continue
        for key in ("correct", "faithful", "note"):
            entry[key] = before.get(key)
        copied += 1
    return copied


def read_answers(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_answers(path: Path, entries: list[dict], instructions: str) -> None:
    """Refuses to overwrite: the file may already hold grades."""
    if path.exists():
        raise FileExistsError(f"{path} already exists and may hold grades")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(instructions + "\n" + body, encoding="utf-8")


def make_run_id(now: datetime, strategy: str, config: Config, closed_book: bool) -> str:
    """2026-10-01_1030_section_t256_o0_k8_steps1, or ..._closed_book."""
    if closed_book:
        return f"{now:%Y-%m-%d_%H%M}_closed_book"
    return f"{now:%Y-%m-%d_%H%M}_{strategy}_k{config.retrieval.final_top_k}_steps{config.llm.max_steps}"


# --- the run -------------------------------------------------------------------------

def run_agent(
    questions: list[EvalQuestion], sections: dict[str, Section], doc_titles: dict[str, str],
    searcher: Searcher, client: LLMClient, config: Config,
) -> tuple[list[dict], TokenUsage]:
    system_prompt = load_prompt(config.llm.prompt_path)
    entries = []
    usage_total = TokenUsage()
    for number, question in enumerate(questions, start=1):
        start = time.perf_counter()
        done, usage = ask(question, searcher, client, config, system_prompt)
        seconds = time.perf_counter() - start
        section = sections.get(question.section_id) if question.section_id else None
        checks = check_answer(question, done.answer, done.sources.chunks, done.truncated, section)
        entries.append(answer_entry(question, doc_titles, done, checks, usage, seconds))
        usage_total.add(usage)
        print(f"  {number}/{len(questions)} {question.question_id} {seconds:.1f} s")
    return entries, usage_total


def run_closed_book(
    questions: list[EvalQuestion], doc_titles: dict[str, str], client: LLMClient, config: Config
) -> tuple[list[dict], TokenUsage]:
    prompt = load_prompt(config.eval.generation.closed_book_prompt_path)
    entries = []
    usage_total = TokenUsage()
    for number, question in enumerate(questions, start=1):
        text, usage = ask_closed_book(question, client, config, prompt)
        entries.append(closed_book_entry(question, doc_titles, text))
        usage_total.add(usage)
        print(f"  {number}/{len(questions)} {question.question_id}")
    return entries, usage_total


def print_trial(entries: list[dict]) -> None:
    for entry in entries:
        print(f"\n--- {entry['question_id']} ({entry['kind']})")
        if "previous_question" in entry:
            print(f"Previous: {entry['previous_question']}")
        print(f"Q: {entry['question']}")
        for query in entry.get("queries", []):
            print(f"Search: {query}")
        print(f"A: {entry['answer']}")
        if "checks" in entry:
            print(f"Checks: {entry['checks']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on the generation set and write the answers to grade.")
    parser.add_argument("--questions", type=Path, help="the question set; default: eval.generation.questions")
    parser.add_argument("--strategy", help="chunking strategy id; default: chunking.active")
    parser.add_argument("--final-top-k", type=int, help="chunks per search; default: retrieval.final_top_k")
    parser.add_argument("--max-steps", type=int, help="most searches per question; default: llm.max_steps")
    parser.add_argument("--closed-book", action="store_true", help="the model alone, with no retrieval")
    parser.add_argument("--reuse-grades", metavar="RUN_ID", help="copy the grades of identical answers from RUN_ID")
    parser.add_argument("--limit", type=int, help="trial on N random questions: prints the answers, saves nothing")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml")).with_overrides(final_top_k=args.final_top_k, max_steps=args.max_steps)
    strategy = args.strategy or config.chunking.active
    questions = load_question_set(args.questions or config.eval.generation.questions)
    if args.limit:
        questions = seeded_sample(questions, args.limit, config.eval.split_seed)
    documents = load_documents(config.paths.docs)
    sections = {section.section_id: section for doc in documents for section in doc.sections}
    doc_titles = {doc.doc_id: doc.title for doc in documents}
    client = make_client(config.llm.base_url, api_key_from_env(config.llm.api_key_env))

    run_id = make_run_id(datetime.now(), strategy, config, args.closed_book)
    print(f"Run {run_id} · {len(questions)} questions · {config.llm.model}\n")
    if args.closed_book:
        entries, usage = run_closed_book(questions, doc_titles, client, config)
        instructions = CLOSED_BOOK_INSTRUCTIONS
    else:
        searcher = open_searcher(config, collection_name(config, strategy), config.paths.qdrant_experiments)
        try:
            entries, usage = run_agent(questions, sections, doc_titles, searcher, client, config)
        finally:
            searcher.close()
        instructions = GRADING_INSTRUCTIONS

    prices = config.model_prices[config.llm.model]
    print(f"\nTokens: {usage.input_tokens:,} in ({usage.cached_input_tokens:,} cached), "
          f"{usage.output_tokens:,} out -> ${usage.cost_usd(prices.input, prices.output, prices.cached_input):.4f}")
    if args.limit:
        print_trial(entries)
        print("\nTrial run (--limit): nothing saved.")
        return

    if args.reuse_grades:
        earlier = read_answers(config.eval.generation.runs_output / args.reuse_grades / ANSWERS_FILE)
        print(f"Grades reused from {args.reuse_grades}: {reuse_grades(entries, earlier)} identical answers")
    path = config.eval.generation.runs_output / run_id / ANSWERS_FILE
    write_answers(path, entries, instructions)
    print(f"Answers to grade -> {path}")


if __name__ == "__main__":
    main()
