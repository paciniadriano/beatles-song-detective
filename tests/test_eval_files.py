"""Tests for the scripts that run after the pool is generated: loading and
saving question files, the dev/test split, and the audit sample. All pure
functions or tmp_path files; no API and no corpus needed."""

from pathlib import Path

import pytest

from src.eval.audit_sample import read_verdicts, summary, to_review_entry, write_review_file
from src.eval.metrics import proportion_margin, seeded_sample, share_text
from src.eval.question_file import load_questions, save_questions
from src.eval.split import split_by_doc_type
from src.schemas import EvalQuestion


def _question(index: int, doc_type: str = "song") -> EvalQuestion:
    return EvalQuestion(
        question_id=f"q{index}", question="q?", answer="a", doc_id=f"wp_{index}",
        doc_type=doc_type, section_id=f"wp_{index}__s0", section_heading="Recording",
        evidence="e",
    )


# --- question_file ----------------------------------------------------------

def test_questions_survive_a_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    questions = [_question(i) for i in range(3)]
    save_questions(path, questions)
    assert load_questions(path) == questions


def test_a_duplicate_question_id_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pool.jsonl"
    save_questions(path, [_question(1), _question(1)])
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


# --- split ------------------------------------------------------------------

def test_every_doc_type_is_split_by_the_dev_ratio() -> None:
    questions = [_question(i, "song") for i in range(100)] + [_question(i, "studio") for i in range(100, 110)]
    split = split_by_doc_type(questions, dev_ratio=0.7, seed=42)
    songs = [q for q in split if q.doc_type == "song"]
    studios = [q for q in split if q.doc_type == "studio"]
    assert sum(q.split == "dev" for q in songs) == 70
    assert sum(q.split == "dev" for q in studios) == 7
    assert all(q.split in ("dev", "test") for q in split)


def test_the_split_is_repeatable_and_keeps_the_file_order() -> None:
    questions = [_question(i) for i in range(20)]
    first = split_by_doc_type(questions, dev_ratio=0.7, seed=42)
    second = split_by_doc_type(questions, dev_ratio=0.7, seed=42)
    assert [q.split for q in first] == [q.split for q in second]
    assert [q.question_id for q in first] == [q.question_id for q in questions]


# --- audit_sample -----------------------------------------------------------

def test_a_seeded_sample_is_repeatable_and_has_the_requested_size() -> None:
    questions = [_question(i) for i in range(50)]
    first = seeded_sample(questions, 10, seed=42)
    assert [q.question_id for q in first] == [q.question_id for q in seeded_sample(questions, 10, seed=42)]
    assert len(first) == 10
    assert len(seeded_sample(questions, 100, seed=42)) == 50   # fewer items than asked: all of them


def test_a_review_entry_carries_the_article_title_and_starts_without_a_verdict() -> None:
    entry = to_review_entry(_question(1), {"wp_1": "Penny Lane"})
    assert entry["article"] == "Penny Lane"
    assert entry["verdict"] == "" and entry["note"] == ""


def test_an_unfinished_audit_cannot_be_reported() -> None:
    entries = [{"question_id": "q1", "verdict": "accepted"}, {"question_id": "q2", "verdict": ""}]
    with pytest.raises(ValueError, match="q2"):
        read_verdicts(entries)


def test_the_summary_counts_rejections_and_shows_their_notes() -> None:
    entries = [{"question_id": f"q{i}", "verdict": "accepted", "note": ""} for i in range(9)]
    entries.append({"question_id": "q9", "verdict": "rejected", "note": "ambiguous"})
    text = summary(read_verdicts(entries))
    assert text.startswith("1 of 10 rejected (10%)")
    assert "q9: ambiguous" in text


def test_the_review_file_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "audit.yaml"
    write_review_file(path, [to_review_entry(_question(1), {"wp_1": "Penny Lane"})])
    with pytest.raises(FileExistsError):
        write_review_file(path, [])


def test_margins_are_95_percent() -> None:
    """With 100 items and a 10% share, the 95% margin is about 6 points."""
    assert round(100 * proportion_margin(0.10, 100)) == 6


def test_a_margin_is_shown_only_with_at_least_five_of_each_outcome() -> None:
    assert share_text(8, 100) == "8% ±5"
    assert share_text(66, 73) == "90% ±7"
    assert share_text(71, 73) == "97%"      # only 2 failures: no "±4"
    assert share_text(3, 3) == "100%"       # never a false "±0"
