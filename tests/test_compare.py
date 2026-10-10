"""Tests for src/eval/compare.py: runs are compared question by question, and
the verdict follows the tie rule."""

import pytest

from src.eval.compare import format_comparison, hit, paired_values
from src.eval.run_eval import QuestionResult


def _run(ranks: list[int | None]) -> list[QuestionResult]:
    return [QuestionResult(question_id=f"q{i}", doc_type="song", topics=["recording"],
                           hit_rank=rank, doc_rank=1, context_tokens=100) for i, rank in enumerate(ranks)]


def test_runs_must_have_scored_the_same_questions() -> None:
    with pytest.raises(ValueError, match="different questions"):
        paired_values(_run([1, 2]), _run([1]), hit, 5, 5)


def test_values_are_aligned_by_question_not_by_order() -> None:
    a, b = _run([1, None]), list(reversed(_run([3, None])))
    assert paired_values(a, b, hit, 5, 5) == ([1.0, 0.0], [1.0, 0.0])
    assert paired_values(a, b, hit, 2, 2) == ([1.0, 0.0], [0.0, 0.0])


def test_a_clear_gain_wins_and_a_small_one_ties() -> None:
    base = [1] * 60 + [None] * 40
    better = [1] * 60 + [1] * 20 + [None] * 20          # +20 points
    slightly = [1] * 60 + [1] * 1 + [None] * 39          # +1 point
    assert "Verdict: B wins" in format_comparison("A", "B", _run(base), _run(better), 5, 5, 20, 2.0)
    report = format_comparison("A", "B", _run(base), _run(slightly), 5, 5, 20, 2.0)
    assert "Verdict: tie (the gap is under 2.0 points)" in report
    assert "only A got it 0 · only B 1" in report
    assert "chance" not in report          # the 2-point bar is the stricter one here


def test_chance_is_named_only_when_it_is_the_stricter_bar() -> None:
    base = [1] * 10 + [None] * 10
    swapped = [None] * 5 + [1] * 5 + [1] * 5 + [None] * 5   # same hit rate, half the answers change
    report = format_comparison("A", "B", _run(base), _run(swapped), 5, 5, 20, 2.0)
    assert "chance alone could produce" in report and "Verdict: tie" in report


def test_each_run_can_be_scored_at_its_own_k() -> None:
    """512 x 5 against 256 x 8, or one run at two values of k: A is scored at
    k_a and B at k_b, on the same questions."""
    run = _run([1, 6, 8, None])
    report = format_comparison("A", "B", run, run, 5, 8, 20, 2.0)
    assert "hit@5/8" in report
    assert "only A got it 0 · only B 2" in report
