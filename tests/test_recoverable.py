"""Tests for src/eval/recoverable.py: which questions a reranker could fix,
and how many it must fix to beat the tie rule's floor."""

from src.eval.recoverable import needed, recoverable_ids
from src.eval.run_eval import QuestionResult


def _result(question_id: str, hit_rank: int | None) -> QuestionResult:
    return QuestionResult(question_id=question_id, doc_type="song", topics=["recording"],
                          hit_rank=hit_rank, doc_rank=1, context_tokens=100)


def test_only_questions_found_below_the_top_k_but_within_the_pool_are_recoverable() -> None:
    results = [_result("hit", 3), _result("edge", 5), _result("low", 6), _result("deep", 20),
               _result("beyond", 21), _result("missing", None)]
    assert recoverable_ids(results, final_top_k=5, pool=20) == {"low", "deep"}


def test_a_win_needs_strictly_more_than_the_floor() -> None:
    assert needed(1240, 2.0) == 25   # 24.8 questions is the floor
    assert needed(1250, 2.0) == 26   # exactly 25 would be a tie, not a win
