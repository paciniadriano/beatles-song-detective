"""Writes the retrieval pool: one evaluation question per eligible section of
the corpus.

Each question is anchored to a document, a section and a verbatim evidence
quote, never to chunk ids, so one frozen pool can score every chunking
strategy. The pool is checked automatically (validate_candidate) and a person
audits a random sample of it; nobody reviews every question. The small,
hand-reviewed generation set is a separate thing (CLAUDE.md, "Evaluation: two separate sets").
Runs offline on the developer's own key (CLAUDE.md, "Operating constraints")."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

from src.config import Config, EvalGenerateConfig, TokenPrices
from src.eval.metrics import seeded_sample
from src.eval.question_file import load_questions
from src.ingest.normalize import load_documents, mentions_band
from src.rag.llm import LLMClient, Msg, TokenUsage, api_key_from_env, complete_structured, make_client
from src.schemas import Document, EvalQuestion, Section

Candidate = tuple[Document, Section]


class GeneratedQuestion(BaseModel):
    """What the model must return for one section (Structured Outputs). The
    API enforces the keys; that `evidence` really is a quote from the section
    is checked afterwards by validate_candidate().

    The field order is deliberate: the model writes them in this order, so it
    picks the quote first and then asks and answers from it. With the quote
    last, the answer was written from the whole section and could say more
    than the quote supports."""
    evidence: str
    question: str
    answer: str

# The audience is curious fans, not engineers (CLAUDE.md, "What this is"). "What to
# ask" steers the one question per section towards substance and plain words;
# without it the model tends to pick whatever is easiest to quote (a date, a
# chart position) and to reuse Wikipedia's studio jargon.
PROMPT = """You write test questions for a search system that answers curious fans' \
questions about the Beatles' songs: the stories behind them, how they were written \
and recorded, who played what, and the unusual details of their sound. You are given \
one section of a Wikipedia article.

First choose one quote from the section, then write ONE question that the quote \
answers, and a short answer.

What to ask:
- Ask about the most interesting thing in the section: a surprising detail, how or \
why something was done, where an idea came from, or who played, sang or engineered \
something.
- Do not ask about release dates, chart positions, sales, catalogue numbers, \
compilations or reissues, or take and remix numbers.
- Do not ask about a song's or album's reputation, rankings, or influence on other \
artists.
- Do not ask a question whose answer is that nobody knows.
- Do not ask a yes/no question.
- Ask about the fact, not about who reported it: do not name the writer, biographer \
or book it comes from.
- Write for a fan who loves the music but does not know studio jargon. Use plain \
words. Name a studio technique only if a fan would know the name; otherwise describe \
what it sounds like ("how did they make the vocal sound so distant?").

Rules:
- The question will be asked with no article in front of the reader, so it must carry \
enough to be answered by this section alone and not by some other song or album. \
It may name the song, album or person, or describe it ("the Revolver song with the \
sped-up vocal").
- Do not copy wording from the text, and never mention "the article", "the section" \
or "the text".
- The quote must be copied EXACTLY, character for character, from the section, and \
be one sentence (or part of one) that alone supports the WHOLE answer. If the answer \
needs more than that one quote, ask a narrower question.
- The answer says only what the quote says: no detail from elsewhere in the section.
- Write the question and the answer as plain text, with no Markdown or italics.
- Never use song lyrics as the quote or the answer.

Article: {title} ({doc_type})
Section: {path}

{text}"""


def eligible_sections(doc: Document, settings: EvalGenerateConfig, out_of_scope: list[str]) -> list[Section]:
    """The sections that get a question. Two rules, both general:
    - the section has at least one topic in scope: "Recording and release"
      gets one, "Release and reception" does not;
    - in an article whose subject is not the Beatles (a person, a studio, a
      piece of equipment, a technique), the section names the band or one of
      them. Otherwise it is the history of an instrument or of a producer's
      other work, which no user of this app asks about."""
    needs_mention = doc.doc_type in settings.mention_required_for_doc_types
    return [
        section for section in doc.sections
        if set(section.topics) - set(out_of_scope)
        and (not needs_mention or mentions_band(section.text, settings.mention_terms))
    ]


def all_candidates(documents: list[Document], settings: EvalGenerateConfig, out_of_scope: list[str]) -> list[Candidate]:
    """Every eligible section of every document, in corpus order: one
    question each. Coverage is complete by construction, not by the luck of a
    draw. The imbalance this leaves (songs are ~60% of eligible sections) is
    intended, since songs are what users ask about, and it is handled when
    results are reported (one row per doc_type), not here."""
    candidates: list[Candidate] = []
    for doc in documents:
        for section in eligible_sections(doc, settings, out_of_scope):
            candidates.append((doc, section))
    return candidates


def build_prompt(doc: Document, section: Section) -> str:
    return PROMPT.format(
        title=doc.title,
        doc_type=doc.doc_type,
        path=section.path,
        text=section.text,
    )


# Characters the model tends to swap for a look-alike when it copies a quote:
# typographic quotes, en and em dashes (credit lists are full of "Name – bass"),
# line breaks between list entries, non-breaking spaces. Each replacement is ONE
# character for ONE character, so a position in the normalised text is the same
# position in the original, and the original can be sliced back out.
LOOKALIKES = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "\n": " ", " ": " ",
})


def find_evidence(evidence: str, section_text: str) -> str | None:
    """The section's own text for the quote the model copied, or None if the
    quote is not in the section. First an exact match; failing that, the same
    search with look-alike characters made equal on both sides, and then
    without a final full stop, which the model adds to credit lines that have
    none ("George Martin – piano."). Either way the section's ORIGINAL
    characters are returned, so the stored evidence is always verbatim and its
    offsets are exact."""
    if evidence in section_text:
        return evidence
    wanted = evidence.strip().translate(LOOKALIKES)
    haystack = section_text.translate(LOOKALIKES)
    for candidate in (wanted, wanted.rstrip(".").rstrip()):
        start = haystack.find(candidate) if candidate else -1
        if start != -1:
            return section_text[start:start + len(candidate)]
    return None


def validate_candidate(generated: GeneratedQuestion, section: Section) -> GeneratedQuestion:
    """Returns the candidate with its evidence replaced by the section's exact
    text, or raises ValueError. This is the check the schema cannot do, and it
    is build-time assertion 4: the quote must be inside THIS section.

    The question itself is not checked against the heading: headings are
    common words ("Recording", "Guitar riff"), so a natural question often
    contains them."""
    exact = find_evidence(generated.evidence, section.text)
    if exact is None:
        raise ValueError(f"evidence is not in the section: {generated.evidence[:80]!r}")
    return generated.model_copy(update={"evidence": exact})


def question_id_for(section: Section) -> str:
    """One question per section, so the section id makes the question id."""
    return f"{section.section_id}__q0"


def to_question(generated: GeneratedQuestion, doc: Document, section: Section) -> EvalQuestion:
    return EvalQuestion(
        question_id=question_id_for(section),
        question=generated.question.strip(),
        answer=generated.answer.strip(),
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        section_id=section.section_id,
        section_heading=section.title,
        evidence=generated.evidence,  # never stripped: it must match the section exactly
    )


def generate_one(
    client: LLMClient, doc: Document, section: Section, config: Config, usage_total: TokenUsage
) -> EvalQuestion:
    """Raises ValueError with the reason when the reply is unusable or fails
    validation, so the caller can count it as rejected and carry on.

    The call's tokens go into `usage_total` before validation: a question
    rejected afterwards was still paid for."""
    messages = [Msg("user", build_prompt(doc, section))]
    generated, usage = complete_structured(
        client, config.eval.generate.model, messages, GeneratedQuestion,
        config.eval.generate.max_output_tokens, config.llm.extra_params,
    )
    usage_total.add(usage)
    return to_question(validate_candidate(generated, section), doc, section)


def sections_still_to_do(candidates: list[Candidate], done: list[EvalQuestion]) -> list[Candidate]:
    """Candidates without a question in `done`, for --resume after an
    interruption. Sections rejected before are tried again: they have no
    question, and a new reply may pass."""
    done_ids = {question.question_id for question in done}
    return [(doc, section) for doc, section in candidates if question_id_for(section) not in done_ids]


def cost_report(usage: TokenUsage, sections_sent: int, pool_size: int, prices: TokenPrices) -> str:
    """What this run cost, and what the whole pool would cost at the same
    average per section. That projection is the point of a pilot run."""
    cost = usage.cost_usd(prices.input, prices.output, prices.cached_input)
    lines = [
        f"Tokens: {usage.input_tokens:,} in ({usage.cached_input_tokens:,} cached), "
        f"{usage.output_tokens:,} out -> ${cost:.4f}"
    ]
    if sections_sent > 0:
        per_section = cost / sections_sent
        lines.append(
            f"~${per_section:.5f} per section; "
            f"the full pool ({pool_size} sections) would cost ~${per_section * pool_size:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the retrieval pool of eval questions.")
    parser.add_argument(
        "--limit", type=int,
        help="try it on N randomly chosen sections; writes to pilot_output, not to the pool",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the prompts, call no API")
    parser.add_argument(
        "--resume", action="store_true",
        help="keep the questions already in the output file and do only the sections missing from it",
    )
    args = parser.parse_args()

    config = Config.load(Path("config.yaml"))
    settings = config.eval.generate
    documents = load_documents(config.paths.docs)
    candidates = all_candidates(documents, settings, config.topics.out_of_scope)
    pool_size = len(candidates)
    output = settings.output
    if args.limit:
        # Random, not the first N: those would all be the first songs in file order.
        candidates = seeded_sample(candidates, args.limit, config.eval.split_seed)
        output = settings.pilot_output
    print(f"{len(candidates)} sections selected")

    mode = "w"
    if args.resume and output.exists():
        candidates = sections_still_to_do(candidates, load_questions(output))
        mode = "a"
        print(f"resuming: {len(candidates)} sections still to do")

    if args.dry_run:
        for doc, section in candidates:
            print(f"\n=== {section.section_id} ===\n{build_prompt(doc, section)}")
        return

    client = make_client(config.llm.base_url, api_key_from_env(config.llm.api_key_env))
    output.parent.mkdir(parents=True, exist_ok=True)
    kept: list[EvalQuestion] = []
    usage_total = TokenUsage()
    # Written as it goes, so an interruption (network, Ctrl+C) keeps what was
    # already generated, and --resume carries on from there. Only a bad reply
    # is skipped; an API error still raises.
    with output.open(mode, encoding="utf-8") as handle:
        for doc, section in candidates:
            try:
                question = generate_one(client, doc, section, config, usage_total)
            except ValueError as error:
                print(f"rejected {section.section_id}: {error}")
                continue
            handle.write(question.model_dump_json() + "\n")
            handle.flush()
            kept.append(question)
    print(f"{len(kept)} of {len(candidates)} kept -> {output}")
    print(cost_report(usage_total, len(candidates), pool_size, config.model_prices[settings.model]))


if __name__ == "__main__":
    main()
