"""Draws the random sample of the retrieval pool that a person audits, and
reports the result once the verdicts are filled in.

Nobody reviews the pool question by question (CLAUDE.md, "Evaluation: two separate sets"). Instead, a
random sample is read by hand, and the share with
problems is what the README reports. The sample must be random: picked by
eye, it would say nothing about the rest of the pool.

Two steps:
    python -m src.eval.audit_sample            # writes the file to review
    python -m src.eval.audit_sample --report   # reads the verdicts back

The audit only measures. It does not edit or drop anything in the pool."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config import Config
from src.eval.metrics import seeded_sample, share_text
from src.eval.question_file import load_questions
from src.ingest.normalize import load_documents
from src.schemas import EvalQuestion

VERDICTS = ("accepted", "rejected")

REVIEW_INSTRUCTIONS = """\
# Audit of the retrieval pool. For each question, set `verdict` to accepted or
# rejected, and write a short `note` when rejecting. Reject when:
#   - the evidence does not support the answer
#   - the question could also be about another song, album or person
#   - the evidence or the answer is song lyrics
#   - it does not read like something a curious fan would ask
# Then run: python -m src.eval.audit_sample --report
"""


def to_review_entry(question: EvalQuestion, doc_titles: dict[str, str]) -> dict[str, str]:
    """One question as a reviewer needs to see it. The article title is
    included because "could this be about another song?" cannot be judged
    without knowing which song it came from."""
    return {
        "question_id": question.question_id,
        "article": doc_titles[question.doc_id],
        "section": question.section_heading,
        "question": question.question,
        "answer": question.answer,
        "evidence": question.evidence,
        "verdict": "",
        "note": "",
    }


@dataclass
class AuditResult:
    reviewed: int
    rejected: list[dict[str, str]]


def read_verdicts(entries: list[dict[str, str]]) -> AuditResult:
    """Raises if any verdict is missing or misspelled, listing the ids: a
    half-filled audit reported as complete would understate the problems."""
    unfinished = [entry["question_id"] for entry in entries if entry.get("verdict") not in VERDICTS]
    if unfinished:
        raise ValueError(f"{len(unfinished)} entries without a valid verdict: {', '.join(unfinished[:10])}")
    rejected = [entry for entry in entries if entry["verdict"] == "rejected"]
    return AuditResult(reviewed=len(entries), rejected=rejected)


def summary(result: AuditResult) -> str:
    """The rejected share with its 95% margin of error, then each rejection's
    note, so a repeated pattern is easy to spot."""
    rejected = len(result.rejected)
    lines = [f"{rejected} of {result.reviewed} rejected ({share_text(rejected, result.reviewed)})"]
    for entry in result.rejected:
        lines.append(f"  {entry['question_id']}: {entry.get('note') or '(no note)'}")
    return "\n".join(lines)


def write_review_file(path: Path, entries: list[dict[str, str]]) -> None:
    """Refuses to overwrite: the existing file may already hold hours of
    verdicts."""
    if path.exists():
        raise FileExistsError(f"{path} already exists and may hold verdicts; delete it by hand to draw again")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(REVIEW_INSTRUCTIONS + "\n" + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a random sample of the retrieval pool.")
    parser.add_argument("--report", action="store_true", help="read the filled-in verdicts and summarise them")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml"))
    settings = config.eval.audit_sample

    if args.report:
        entries = yaml.safe_load(settings.output.read_text(encoding="utf-8"))
        print(summary(read_verdicts(entries)))
        return

    questions = load_questions(config.eval.generate.output)
    doc_titles = {doc.doc_id: doc.title for doc in load_documents(config.paths.docs)}
    sample = seeded_sample(questions, settings.size, config.eval.split_seed)
    write_review_file(settings.output, [to_review_entry(question, doc_titles) for question in sample])
    print(f"{len(sample)} questions to review -> {settings.output}")


if __name__ == "__main__":
    main()
