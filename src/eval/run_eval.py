"""Scores one retrieval configuration on the question pool: for every
question, where the evidence and its article appear in what search() returns,
then hit@k, MRR and doc_hit overall, per doc_type and per topic.

Each run saves its per-question results (what compare.py reads) and adds one
row to eval/experiments.csv, with the settings that define the run. Retrieval
is local and deterministic, so the same run always gives the same numbers:
there is nothing to average over repeated runs.

    python -m src.eval.run_eval --strategy section_t256_o0 --stage alpha --alpha 0.75

--strategy, --embedding, --alpha, --reranker, --final-top-k and
--max-per-section override config.yaml for this run only; the file keeps the
values the app uses."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import tiktoken
from pydantic import BaseModel

from src.config import NO_CAP, NO_RERANKER, Config
from src.eval.metrics import evidence_spans, first_doc_rank, first_hit_rank, reciprocal_rank, seeded_sample, within
from src.eval.question_file import load_questions
from src.eval.recoverable import needed, recoverable_ids
from src.index.collections import collection_name
from src.ingest.normalize import load_documents, load_tokenizer
from src.rag.search import Searcher, open_searcher
from src.schemas import Chunk, Document, EvalQuestion, Section


class QuestionResult(BaseModel):
    """One question's outcome: the rank of the first chunk that hits the
    evidence, and of the first chunk from the right article (None: not in the
    list). Every metric is computed from these two ranks, so two runs can be
    compared question by question without searching again."""
    question_id: str
    doc_type: str
    topics: list[str]
    hit_rank: int | None
    doc_rank: int | None
    context_tokens: int


# --- one question ----------------------------------------------------------------

def context_tokens(chunks: list[Chunk], tokenizer: tiktoken.Encoding) -> int:
    """What the LLM would read: each chunk's header lines and its text."""
    total = 0
    for chunk in chunks:
        header = "\n".join(f"{field}: {value}" for field, value in chunk.header.items())
        total += len(tokenizer.encode(f"{header}\n\n{chunk.text}"))
    return total


def score_question(
    question: EvalQuestion, section: Section, chunks: list[Chunk], final_top_k: int, tokenizer: tiktoken.Encoding
) -> QuestionResult:
    """`chunks` is what search() returned for the question, best first."""
    spans = evidence_spans(question.evidence, section.text, section.char_start)
    return QuestionResult(
        question_id=question.question_id,
        doc_type=question.doc_type,
        topics=section.topics,
        hit_rank=first_hit_rank(chunks, question.doc_id, spans),
        doc_rank=first_doc_rank(chunks, question.doc_id),
        context_tokens=context_tokens(chunks[:final_top_k], tokenizer),
    )


# --- summaries -------------------------------------------------------------------

def summarize(results: list[QuestionResult], k_values: list[int]) -> dict[str, float]:
    """The metrics of a set of questions, for every k: hit@k and MRR@k (what
    the course reports when search returns k chunks) and doc_hit@k; plus the
    average context size. The gap between doc_hit@k and hit@k is the share of
    questions where the right article came back but not the passage."""
    summary: dict[str, float] = {"n": len(results)}
    for metric, value in (
        ("hit", lambda r, k: within(r.hit_rank, k)),
        ("mrr", lambda r, k: reciprocal_rank(r.hit_rank, k)),
        ("doc_hit", lambda r, k: within(r.doc_rank, k)),
    ):
        for k in k_values:
            summary[f"{metric}@{k}"] = statistics.fmean(value(r, k) for r in results)
    summary["context_tokens"] = statistics.fmean(r.context_tokens for r in results)
    return summary


def group_by(results: list[QuestionResult], key: str) -> dict[str, list[QuestionResult]]:
    """Results per doc_type or per topic, largest group first. A question with
    two topics counts in both."""
    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in results:
        values = result.topics if key == "topic" else [result.doc_type]
        for value in values:
            groups[value].append(result)
    return dict(sorted(groups.items(), key=lambda item: -len(item[1])))


# --- output ------------------------------------------------------------------------

def _decimal(value: float, width: int) -> str:
    """Up to 4 decimals without trailing zeros (0.3, 0.9265, 1.0), right-aligned."""
    text = f"{value:.4f}".rstrip("0")
    return (text + "0" if text.endswith(".") else text).rjust(width)


def format_report(results: list[QuestionResult], k_values: list[int], final_top_k: int) -> str:
    """One row per k, as in the course: hit rate and MRR, plus doc_hit. The
    row equal to final_top_k decides. The tables per doc_type and per topic
    show that row only."""
    overall = summarize(results, k_values)
    lines = [
        f"All questions ({len(results):,})",
        f"{'   k':6s}{'hit rate':>10s}{'MRR':>9s}{'doc_hit':>10s}",
    ]
    for k in k_values:
        marker = "   ← decides (final_top_k)" if k == final_top_k else ""
        lines.append(f"{k:4d}  " + _decimal(overall[f"hit@{k}"], 10) + _decimal(overall[f"mrr@{k}"], 9)
                     + _decimal(overall[f"doc_hit@{k}"], 10) + marker)
    lines.append(f"Context tokens (top {final_top_k}): {overall['context_tokens']:,.0f}")
    for key in ("doc_type", "topic"):
        lines += ["", f"{'By ' + key + f' (k = {final_top_k})':24s}{'n':>6s}{'hit rate':>10s}{'MRR':>9s}{'doc_hit':>10s}"]
        for name, group in group_by(results, key).items():
            s = summarize(group, k_values)
            lines.append(f"  {name:22s}{len(group):6d}" + _decimal(s[f"hit@{final_top_k}"], 10)
                         + _decimal(s[f"mrr@{final_top_k}"], 9) + _decimal(s[f"doc_hit@{final_top_k}"], 10))
    return "\n".join(lines)


def reranker_label(config: Config) -> str:
    return config.reranking.active or NO_RERANKER


def cap_label(config: Config) -> str:
    return str(config.retrieval.max_per_section or NO_CAP)


def make_run_id(now: datetime, strategy: str, config: Config, split: str = "dev") -> str:
    """Date, then every setting that tells runs apart, so a folder name says
    what was run: 2026-09-30_1149_section_t256_o0_nomic_a0.5_none_k8_cap2.
    The one test run ends in _test, so it never looks like a dev run."""
    retrieval = config.retrieval
    run_id = (f"{now:%Y-%m-%d_%H%M}_{strategy}_{config.embeddings.active}_a{retrieval.alpha:g}"
              f"_{reranker_label(config)}_k{retrieval.final_top_k}_cap{cap_label(config)}")
    return run_id + "_test" if split == "test" else run_id


def cap_argument(value: str) -> int | str:
    """--max-per-section takes a number, or "none" for no cap."""
    return value if value == NO_CAP else int(value)


def experiment_row(
    run_id: str, stage: str, strategy: str, split: str, config: Config, summary: dict[str, float], search_ms: float
) -> dict[str, str | int | float]:
    """The settings that define the run, then its overall metrics and the
    average time search() took per question."""
    retrieval = config.retrieval
    return {
        "run_id": run_id, "date": run_id[:10], "stage": stage, "strategy": strategy,
        "embedding": config.embeddings.active, "alpha": retrieval.alpha, "reranker": reranker_label(config),
        "final_top_k": retrieval.final_top_k, "max_per_section": retrieval.max_per_section,
        "split": split, "n": int(summary["n"]),
        **{key: round(value, 4) for key, value in summary.items() if key != "n"},
        "search_ms": round(search_ms),
    }


def append_experiment(path: Path, row: dict) -> None:
    """Adds the row, writing the header if the file is new. Refuses a row whose
    columns differ from the file's (k_values changed): the old rows would no
    longer line up with the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
        if header != list(row):
            raise ValueError(f"{path} has columns {header}; this run has {list(row)}. Start a new file.")
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def save_results(path: Path, results: list[QuestionResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")


def load_results(path: Path) -> list[QuestionResult]:
    return [QuestionResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- the run -----------------------------------------------------------------------

def run(
    questions: list[EvalQuestion], sections: dict[str, Section], searcher: Searcher,
    k_values: list[int], final_top_k: int, tokenizer: tiktoken.Encoding,
) -> tuple[list[QuestionResult], float]:
    """The results, and the average milliseconds search() took per question
    (embedding the question, the hybrid query and the reranker, if any)."""
    results = []
    search_seconds = 0.0
    for number, question in enumerate(questions, start=1):
        start = time.perf_counter()
        chunks = searcher.search(question.question, top_k=max(k_values))
        search_seconds += time.perf_counter() - start
        results.append(score_question(question, sections[question.section_id], chunks, final_top_k, tokenizer))
        if number % 100 == 0:
            print(f"  {number}/{len(questions)} questions")
    return results, 1000 * search_seconds / len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score one retrieval configuration on the question pool.")
    parser.add_argument("--strategy", help="chunking strategy id; default: chunking.active")
    parser.add_argument("--embedding", help="embedding candidate; default: embeddings.active")
    parser.add_argument("--alpha", type=float, help="0 = sparse only, 1 = dense only; default: retrieval.alpha")
    parser.add_argument("--reranker", help=f"reranker candidate, or {NO_RERANKER}; default: reranking.active")
    parser.add_argument("--final-top-k", type=int, help="chunks the LLM receives, one of eval.k_values; default: retrieval.final_top_k")
    parser.add_argument("--max-per-section", type=cap_argument,
                        help=f"cap of chunks per section, or {NO_CAP}; default: retrieval.max_per_section")
    parser.add_argument("--stage", default="adhoc", help="the experiment stage this run belongs to (chunking, alpha, top_k...)")
    parser.add_argument("--split", choices=["dev", "test"], default="dev", help="test runs once, at the very end")
    trial = parser.add_mutually_exclusive_group()
    trial.add_argument("--limit", type=int, help="trial on N random questions; prints the tables, saves nothing")
    trial.add_argument("--recoverable-from", metavar="RUN_ID",
                       help="trial on the questions RUN_ID ranked below final_top_k but within the reranker's "
                            "top_n (recoverable.py); saves nothing")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml")).with_overrides(
        embedding=args.embedding, alpha=args.alpha, reranker=args.reranker,
        final_top_k=args.final_top_k, max_per_section=args.max_per_section,
    )
    strategy = args.strategy or config.chunking.active
    documents: list[Document] = load_documents(config.paths.docs)
    sections = {section.section_id: section for doc in documents for section in doc.sections}
    questions = [q for q in load_questions(config.eval.generate.output) if q.split == args.split and q.answerable]
    if not questions:
        raise ValueError(f"no {args.split} questions: run `python -m src.eval.split` first")
    if args.limit:
        questions = seeded_sample(questions, args.limit, config.eval.split_seed)
    rescue_target = None
    if args.recoverable_from:
        baseline = load_results(config.eval.runs_output / args.recoverable_from / "per_question.jsonl")
        ids = recoverable_ids(baseline, config.retrieval.final_top_k, config.reranking.top_n)
        rescue_target = needed(len(questions), config.eval.min_meaningful_gap_points)
        questions = [q for q in questions if q.question_id in ids]

    run_id = make_run_id(datetime.now(), strategy, config, args.split)
    final_top_k = config.retrieval.final_top_k
    print(f"Run {run_id} · stage {args.stage}")
    print(f"{strategy} · {config.embeddings.active} · alpha {config.retrieval.alpha} · reranker {reranker_label(config)} "
          f"· final_top_k {final_top_k} · max_per_section {config.retrieval.max_per_section} "
          f"· {args.split} · {len(questions)} questions")

    searcher = open_searcher(config, collection_name(config, strategy), config.paths.qdrant_experiments)
    if searcher.reranker is not None:
        print(f"Reranker {searcher.reranker.model_name}: reorders the top {config.reranking.top_n}, "
              f"cuts each pair at {searcher.reranker.max_tokens} tokens")
    print()
    try:
        results, search_ms = run(questions, sections, searcher, config.eval.k_values, final_top_k, load_tokenizer(config))
    finally:
        searcher.close()

    print("\n" + format_report(results, config.eval.k_values, final_top_k))
    print(f"\nsearch(): {search_ms:,.0f} ms per question on this machine")
    if rescue_target is not None:
        rescued = sum(within(r.hit_rank, final_top_k) for r in results)
        print(f"\nRescued {rescued} of {len(results)} recoverable questions; beating the tie rule's floor "
              f"needs at least {rescue_target}.\nTrial run (--recoverable-from): nothing saved.")
        return
    if args.limit:
        print("\nTrial run (--limit): nothing saved.")
        return
    results_path = config.eval.runs_output / run_id / "per_question.jsonl"
    save_results(results_path, results)
    summary = summarize(results, config.eval.k_values)
    append_experiment(config.eval.experiments_output, experiment_row(run_id, args.stage, strategy, args.split, config, summary, search_ms))
    print(f"\nSaved {results_path}\nAdded a row to {config.eval.experiments_output}")


if __name__ == "__main__":
    main()
