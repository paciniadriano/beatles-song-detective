"""What code can check about one generated answer, so a person grades only
what code cannot (CLAUDE.md, "Evaluation: two separate sets").

A question the sources cannot answer passes on `declined` alone: no hand
grading. An answerable one gets two hand grades, `correct` and `faithful`,
and passes only if both hold and every [n] it cites exists and it has no
link. The other checks help the person grade faster (`evidence_cited`) or
feed the max_steps pre-check (`evidence_retrieved`)."""

from __future__ import annotations

import re

from pydantic import BaseModel

from src.eval.metrics import evidence_spans, first_hit_rank
from src.rag.tools import cited_numbers
from src.schemas import Chunk, EvalQuestion, Section

DECLINE_PHRASE = "The sources here don't say."

_LINK = re.compile(r"https?://|www\.", re.IGNORECASE)


class Checks(BaseModel):
    """The automatic checks of one answer. The two evidence checks are None
    for questions with no evidence (the ones that must decline)."""
    declined: bool
    cites: bool
    citations_exist: bool
    no_link: bool
    truncated: bool
    evidence_retrieved: bool | None = None
    evidence_cited: bool | None = None


class Grade(BaseModel):
    """The hand grades of an answerable question. None: not graded yet."""
    correct: bool | None = None
    faithful: bool | None = None


def comparable(text: str) -> str:
    """Case, typographic apostrophes and outer spaces do not make an answer
    different: "The sources here don’t say" is the decline phrase too."""
    return text.replace("’", "'").strip().rstrip(".").casefold()


def is_decline(answer: str) -> bool:
    return comparable(answer) == comparable(DECLINE_PHRASE)


def has_link(answer: str) -> bool:
    return _LINK.search(answer) is not None


def evidence_found(question: EvalQuestion, section: Section, chunks: list[Chunk]) -> bool:
    """True if one of the chunks covers the question's evidence (the same hit
    definition as the retrieval pool)."""
    spans = evidence_spans(question.evidence, section.text, section.char_start)
    return first_hit_rank(chunks, question.doc_id, spans) is not None


def check_answer(
    question: EvalQuestion,
    answer: str,
    sources: list[Chunk],
    truncated: bool,
    section: Section | None,
) -> Checks:
    """`sources` are the chunks shown to the model, [1] first; `section` is
    the question's evidence section, None if it has none."""
    numbers = cited_numbers(answer)
    cited = [sources[number - 1] for number in numbers if 1 <= number <= len(sources)]
    checks = Checks(
        declined=is_decline(answer),
        cites=bool(numbers),
        citations_exist=len(cited) == len(numbers),
        no_link=not has_link(answer),
        truncated=truncated,
    )
    if question.answerable and section is not None:
        checks.evidence_retrieved = evidence_found(question, section, sources)
        checks.evidence_cited = evidence_found(question, section, cited)
    return checks


def passes(answerable: bool, checks: Checks, grade: Grade) -> bool:
    """One pass/fail per question, the number two runs are compared on.
    Raises if an answerable question is not fully graded yet."""
    if not answerable:
        return checks.declined
    if grade.correct is None or grade.faithful is None:
        raise ValueError("an answerable question needs both hand grades")
    return grade.correct and grade.faithful and checks.citations_exist and checks.no_link
