"""Compares two saved runs question by question and applies the tie rule, so
a configuration is chosen only when it is ahead by more than chance and by
more than min_meaningful_gap_points.

Both runs must have scored the same questions (the same split of the same
pool). The deciding metric is hit@k, with k = final_top_k unless --k says
otherwise; MRR, doc_hit and hit at the largest k are shown to explain where
a difference comes from. Every run stores the rank of each question's first
hit, so any k up to the largest can be scored from it.

    python -m src.eval.compare <run A> <run B>
    python -m src.eval.compare <run A> <run B> --k 5
    python -m src.eval.compare <run A> <run B> --k 5 --k-b 8   (512 x 5 against 256 x 8)
    python -m src.eval.compare <run> <run> --k 5 --k-b 7       (one run, two values of k)"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable
from pathlib import Path

from src.config import Config
from src.eval.metrics import paired_difference, reciprocal_rank, verdict, within
from src.eval.run_eval import QuestionResult, load_results

# A metric for one question at a given k: hit, MRR term or doc_hit.
Metric = Callable[[QuestionResult, int], float]


def hit(result: QuestionResult, k: int) -> float:
    return float(within(result.hit_rank, k))


def mrr(result: QuestionResult, k: int) -> float:
    return reciprocal_rank(result.hit_rank, k)


def doc_hit(result: QuestionResult, k: int) -> float:
    return float(within(result.doc_rank, k))


def paired_values(
    a: list[QuestionResult], b: list[QuestionResult], metric: Metric, k_a: int, k_b: int
) -> tuple[list[float], list[float]]:
    """The metric for each question in both runs, A scored at k_a and B at
    k_b, aligned by question_id. Raises if the runs did not score the same
    questions."""
    by_id_b = {r.question_id: r for r in b}
    if {r.question_id for r in a} != set(by_id_b):
        raise ValueError("the two runs scored different questions; compare runs of the same split and pool")
    return ([float(metric(r, k_a)) for r in a],
            [float(metric(by_id_b[r.question_id], k_b)) for r in a])


def format_comparison(
    name_a: str, name_b: str, a: list[QuestionResult], b: list[QuestionResult],
    k_a: int, k_b: int, largest_k: int, min_gap_points: float,
) -> str:
    """The comparison report. k_a and k_b are the number of chunks each option
    sends; they differ only when that number is what is being compared."""
    k = str(k_a) if k_a == k_b else f"{k_a}/{k_b}"   # shown as "hit@5/8" when they differ
    rows = [
        (f"hit@{k}", hit, k_a, k_b, True),
        (f"MRR@{k}", mrr, k_a, k_b, False),
        (f"doc_hit@{k}", doc_hit, k_a, k_b, True),
        (f"hit@{largest_k}", hit, largest_k, largest_k, True),
    ]
    lines = [f"A: {name_a}", f"B: {name_b}", f"{len(a):,} questions, compared one by one", "",
             f"{'':14s}{'A':>8s}{'B':>8s}{'B - A':>10s}"]
    for label, metric, row_k_a, row_k_b, in_points in rows:
        values_a, values_b = paired_values(a, b, metric, row_k_a, row_k_b)
        mean_a, mean_b = statistics.fmean(values_a), statistics.fmean(values_b)
        gap = f"{100 * (mean_b - mean_a):+.1f} pts" if in_points else f"{mean_b - mean_a:+.3f}"
        marker = "   <- decides" if label == f"hit@{k}" else ""
        lines.append(f"  {label:12s}{mean_a:8.3f}{mean_b:8.3f}{gap:>10s}{marker}")
    tokens_a = statistics.fmean(r.context_tokens for r in a)
    tokens_b = statistics.fmean(r.context_tokens for r in b)
    lines.append(f"  {'context tok':12s}{tokens_a:8,.0f}{tokens_b:8,.0f}")

    hits_a, hits_b = paired_values(a, b, hit, k_a, k_b)
    only_a = sum(1 for x, y in zip(hits_a, hits_b) if x and not y)
    only_b = sum(1 for x, y in zip(hits_a, hits_b) if y and not x)
    both = sum(1 for x, y in zip(hits_a, hits_b) if x and y)
    lines += ["", f"hit@{k}: only A got it {only_a} · only B {only_b} · both {both} · neither {len(a) - only_a - only_b - both}"]

    difference, margin = paired_difference(hits_a, hits_b)
    result = verdict(difference, margin, min_gap_points / 100)
    leader = "B" if difference > 0 else "A"
    threshold = max(100 * margin, min_gap_points)
    lines.append(f"{leader} is {abs(100 * difference):.1f} points ahead.")
    # Chance is named only when it is the stricter bar. With this pool it
    # rarely is: it needs the runs to disagree on about 13% of the questions.
    if 100 * margin > min_gap_points:
        lines.append(f"Note: these runs disagree on many questions, so chance alone could produce a gap of "
                     f"up to {100 * margin:.1f} points; a win must beat that instead of {min_gap_points:.1f}.")
    if result == "tie":
        lines.append(f"Verdict: tie (the gap is under {threshold:.1f} points) -> keep the simpler configuration")
    else:
        lines.append(f"Verdict: {result} (by more than {threshold:.1f} points)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two saved runs with the tie rule.")
    parser.add_argument("run_a", help="run id (a folder under eval.runs_output)")
    parser.add_argument("run_b")
    parser.add_argument("--k", type=int, help="chunks sent, i.e. the k of hit@k (default: retrieval.final_top_k)")
    parser.add_argument("--k-b", type=int, help="B's k when the two options send a different number "
                                                "of chunks (default: the same as --k)")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml"))
    largest_k = max(config.eval.k_values)
    k_a = args.k or config.retrieval.final_top_k
    k_b = args.k_b or k_a
    if max(k_a, k_b) > largest_k:
        parser.error(f"k can be at most {largest_k}: runs store ranks only up to the largest k_values entry")
    runs = config.eval.runs_output
    a = load_results(runs / args.run_a / "per_question.jsonl")
    b = load_results(runs / args.run_b / "per_question.jsonl")
    print(format_comparison(args.run_a, args.run_b, a, b, k_a, k_b,
                            largest_k, config.eval.min_meaningful_gap_points))


if __name__ == "__main__":
    main()
