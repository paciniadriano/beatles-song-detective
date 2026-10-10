"""The questions a reranker could fix, for a quick test before a full run.

A reranker only reorders the candidates search already found, so it can only
gain on questions whose evidence is in the list but below final_top_k: ranked
from final_top_k + 1 to 20 in a run without a reranker. To win by the tie
rule's floor over the whole split it must rescue at least `needed` of them.
Running only those questions measures the best case: if even that falls
short, no full run can win. If it reaches `needed`, the full run is still
required, because the reranker may also push down questions that were
already found.

Used by run_eval.py --recoverable-from. A trial, like --limit: nothing is saved."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol


class RankedResult(Protocol):
    question_id: str
    hit_rank: int | None


def recoverable_ids(results: Iterable[RankedResult], final_top_k: int, pool: int) -> set[str]:
    """Questions found within `pool` but not within `final_top_k`."""
    return {r.question_id for r in results if r.hit_rank is not None and final_top_k < r.hit_rank <= pool}


def needed(total_questions: int, min_gap_points: float) -> int:
    """Rescued questions needed to beat the tie rule's floor on the whole split."""
    return math.floor(total_questions * min_gap_points / 100) + 1
