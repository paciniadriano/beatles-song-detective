"""Reads a graded run of the generation set and reports it; optionally
compares it with another run by the tie rule, and with a closed-book run.

    python -m src.eval.generation_report RUN_ID
    python -m src.eval.generation_report RUN_B --against RUN_A      # B minus A
    python -m src.eval.generation_report RUN_ID --closed-book CLOSED_RUN_ID

Refuses to report while any grade is missing: a half-graded run reported as
complete would hide exactly the answers not yet looked at."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.config import Config
from src.eval.generation_checks import Checks, Grade, passes
from src.eval.metrics import fewest_gains_to_win, paired_difference, share_text, verdict
from src.eval.run_generation import ANSWERS_FILE, read_answers


def is_graded_entry(entry: dict) -> bool:
    """Answerable questions carry hand grades; the others do not."""
    return "correct" in entry


def unfinished(entries: list[dict], keys: tuple[str, ...]) -> list[str]:
    """Ids of the entries that should have every key filled and do not."""
    return [
        entry["question_id"] for entry in entries
        if any(key in entry and entry[key] is None for key in keys)
    ]


def refuse_unfinished(entries: list[dict], keys: tuple[str, ...]) -> None:
    missing = unfinished(entries, keys)
    if missing:
        raise ValueError(f"{len(missing)} answers not graded yet: {', '.join(missing[:10])}")


def entry_passes(entry: dict) -> bool:
    checks = Checks.model_validate(entry["checks"])
    grade = Grade(correct=entry.get("correct"), faithful=entry.get("faithful"))
    return passes(is_graded_entry(entry), checks, grade)


def share_line(label: str, values: list[bool], with_margin: bool = True) -> str:
    """"faithful: 66/73 (90% ±7)": the count, the share and its 95% margin,
    when there are enough of both outcomes for one (metrics.share_text).
    Breakdown rows pass with_margin=False: with 3 to 38 questions a margin
    (±31 at 10) says nothing useful, so they are indicative counts, like the
    small rows of the retrieval pool."""
    if not values:
        return f"{label}: no questions"
    share = 100 * sum(values) / len(values)
    detail = share_text(sum(values), len(values)) if with_margin else f"{share:.0f}%"
    return f"{label}: {sum(values)}/{len(values)} ({detail})"


def summary(entries: list[dict]) -> str:
    """Pass overall and per kind (the number runs are compared on), then the
    parts it is made of."""
    lines = [share_line("pass", [entry_passes(entry) for entry in entries])]
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for entry in entries:
        by_kind[entry.get("kind") or "?"].append(entry_passes(entry))
    for kind in sorted(by_kind):
        lines.append("  " + share_line(kind, by_kind[kind], with_margin=False))

    graded = [entry for entry in entries if is_graded_entry(entry)]
    declining = [entry for entry in entries if not is_graded_entry(entry)]
    lines.append("Answerable questions:")
    for key in ("correct", "faithful"):
        lines.append("  " + share_line(key, [bool(entry[key]) for entry in graded]))
    for key in ("citations_exist", "no_link", "evidence_retrieved", "evidence_cited"):
        lines.append("  " + share_line(key, [bool(entry["checks"][key]) for entry in graded]))
    lines.append("Questions that must decline:")
    lines.append("  " + share_line("declined", [entry["checks"]["declined"] for entry in declining]))
    truncated = sum(entry["checks"]["truncated"] for entry in entries)
    lines.append(f"Answers cut off by the token limit: {truncated}")
    return "\n".join(lines)


def rescuable(entries: list[dict]) -> int:
    """Answerable questions that failed and whose evidence was not among the
    sources: the only ones searching again could turn into a pass. A question
    answered right from another article (the same fact in two places) has no
    evidence passage either, but it passed, so it does not count."""
    return sum(
        1 for entry in entries
        if is_graded_entry(entry) and entry["checks"]["evidence_retrieved"] is False and not entry_passes(entry)
    )


def max_steps_pre_check(entries: list[dict], min_gap: float) -> str:
    """Whether the 2-step arm is worth running at all: only if enough failed
    questions lacked their evidence for a second search to win by the tie
    rule (CLAUDE.md, "Experiment order", the max_steps stage)."""
    count = rescuable(entries)
    needed = fewest_gains_to_win(len(entries), min_gap)
    decision = "run the 2-step arm" if needed is not None and count >= needed else "1 step stays; no 2-step run"
    return (f"max_steps pre-check: {count} answerable questions failed without their evidence among the sources; "
            f"searching again can win only if at least {needed} -> {decision}")


def compare(a: list[dict], b: list[dict], min_gap: float) -> str:
    """B minus A on pass, paired question by question, with the tie rule."""
    a_by_id = {entry["question_id"]: entry for entry in a}
    b_by_id = {entry["question_id"]: entry for entry in b}
    if a_by_id.keys() != b_by_id.keys():
        raise ValueError("the two runs did not answer the same questions")
    ids = sorted(a_by_id)
    a_pass = [float(entry_passes(a_by_id[i])) for i in ids]
    b_pass = [float(entry_passes(b_by_id[i])) for i in ids]
    difference, margin = paired_difference(a_pass, b_pass)
    only_a = sum(1 for x, y in zip(a_pass, b_pass) if x > y)
    only_b = sum(1 for x, y in zip(a_pass, b_pass) if y > x)
    return (f"pass, B minus A: {100 * difference:+.1f} points (margin ±{100 * margin:.1f}, "
            f"floor {100 * min_gap:.0f}); passed only by A: {only_a}, only by B: {only_b} "
            f"-> {verdict(difference, margin, min_gap)}")


def closed_book_summary(answers: list[dict], closed: list[dict]) -> str:
    """How much the model knew unaided, and how the RAG run did on the
    questions it did not know: what retrieval actually adds."""
    graded = [entry for entry in closed if "closed_book_correct" in entry]
    unaided = [bool(entry["closed_book_correct"]) for entry in graded]
    missed_ids = {entry["question_id"] for entry in graded if not entry["closed_book_correct"]}
    rag_on_missed = [entry_passes(entry) for entry in answers if entry["question_id"] in missed_ids]
    return "\n".join([
        share_line("closed book, correct unaided", unaided),
        share_line("with RAG, pass on the questions it missed unaided", rag_on_missed),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Report a graded run of the generation set.")
    parser.add_argument("run_id")
    parser.add_argument("--against", metavar="RUN_A", help="compare: this run is B, RUN_A is A")
    parser.add_argument("--closed-book", metavar="CLOSED_RUN_ID", help="a graded closed-book run")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml"))
    runs = config.eval.generation.runs_output
    entries = read_answers(runs / args.run_id / ANSWERS_FILE)
    refuse_unfinished(entries, ("correct", "faithful"))
    print(f"Run {args.run_id}\n")
    print(summary(entries))
    print("\n" + max_steps_pre_check(entries, config.eval.min_meaningful_gap_points / 100))

    if args.against:
        earlier = read_answers(runs / args.against / ANSWERS_FILE)
        refuse_unfinished(earlier, ("correct", "faithful"))
        print(f"\nB = {args.run_id}, A = {args.against}")
        print(compare(earlier, entries, config.eval.min_meaningful_gap_points / 100))
    if args.closed_book:
        closed = read_answers(runs / args.closed_book / ANSWERS_FILE)
        refuse_unfinished(closed, ("closed_book_correct",))
        print()
        print(closed_book_summary(entries, closed))


if __name__ == "__main__":
    main()
