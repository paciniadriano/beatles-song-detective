"""Tests for the rule that ties `answerable` to the anchor fields of an
EvalQuestion. A question that breaks it would score wrong without any error,
so the model refuses it when the file is loaded."""

import pytest
from pydantic import ValidationError

from src.schemas import EvalQuestion

ANCHORED = {
    "question_id": "wp_1__s2__q0",
    "question": "Who played the cowbell?",
    "answer": "Lennon.",
    "doc_id": "wp_1",
    "doc_type": "song",
    "section_id": "wp_1__s2",
    "section_heading": "Recording",
    "evidence": "Lennon beating a cowbell",
}

DECLINE = {
    "question_id": "manual__q1",
    "question": "Which technique did Geoff Emerick use most?",
    "answer": "The sources here don't say.",
    "answerable": False,
    "doc_type": "person",
}


def test_an_anchored_question_is_answerable_by_default() -> None:
    question = EvalQuestion(**ANCHORED)
    assert question.answerable is True
    assert question.previous_question is None


def test_an_answerable_question_without_evidence_is_refused() -> None:
    with pytest.raises(ValidationError):
        EvalQuestion(**{**ANCHORED, "evidence": None})


def test_a_question_that_is_not_answerable_needs_no_anchor() -> None:
    question = EvalQuestion(**DECLINE)
    assert question.answerable is False
    assert question.doc_id is None


def test_a_question_that_is_not_answerable_cannot_carry_evidence() -> None:
    with pytest.raises(ValidationError):
        EvalQuestion(**{**DECLINE, "evidence": "Emerick used ADT"})
