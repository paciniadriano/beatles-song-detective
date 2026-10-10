"""Evaluation arithmetic shared by the eval scripts: the constants that define
the metrics, hit and rank for one question, paired comparison of two runs,
margins of error and seeded sampling.

Every margin of error in the project is a 95% margin: Z_95 times the standard
error. One function computes it, so the audit and the retrieval results can
never use different conventions."""

from __future__ import annotations

import math
import random
import statistics
from typing import Protocol, TypeVar

Item = TypeVar("Item")

# 95% of a normal distribution lies within 1.96 standard errors of its mean.
# A fixed convention, not a setting: the docs and the README say "95%".
Z_95 = 1.96

# A retrieved chunk "hits" a question when it is in the question's document and
# covers at least this share of the evidence span (character offsets). Not
# containment: a quote cut by a chunk boundary would then miss in every chunk,
# which penalises small chunks for how the metric is measured. At one half, the
# larger piece of a cut quote always hits. Part of the metric's definition, so
# it stays fixed with the pool.
HIT_MIN_COVERAGE = 0.5


# The normal approximation behind a share's margin holds only with enough of
# both outcomes; the textbook rule is at least 5 of each. Below that (3 of 3,
# 71 of 73) it would print a false "±0" or a margin past 100%, so the share is
# shown as a plain count instead.
MIN_EACH_SIDE_FOR_MARGIN = 5


def proportion_margin(share: float, n: int) -> float:
    """The 95% margin of error of a share measured on n items (normal
    approximation): with 100 items and a 10% share, about ±6 points."""
    return Z_95 * math.sqrt(share * (1 - share) / n)


def share_text(successes: int, n: int) -> str:
    """"90% ±7", or just "97%" when there are fewer than
    MIN_EACH_SIDE_FOR_MARGIN of either outcome: the count already says how
    little a handful of items can show."""
    if n <= 0:
        raise ValueError("a share needs at least one item")
    share = successes / n
    if min(successes, n - successes) < MIN_EACH_SIDE_FOR_MARGIN:
        return f"{100 * share:.0f}%"
    return f"{100 * share:.0f}% ±{100 * proportion_margin(share, n):.0f}"


def seeded_sample(items: list[Item], size: int, seed: int) -> list[Item]:
    """A random subset of `size` items, the same one for a given seed, or all
    of them if there are not that many. Used for the pilot run and for the
    audit sample."""
    if size >= len(items):
        return list(items)
    return random.Random(seed).sample(items, size)


# --- retrieval metrics -------------------------------------------------------
# A retrieved chunk is anything with a document and a character range in it
# (src.schemas.Chunk); these functions read only those three fields.

class Located(Protocol):
    doc_id: str
    char_start: int
    char_end: int


Span = tuple[int, int]  # a character range in a Document's full_text


def evidence_spans(evidence: str, section_text: str, section_start: int) -> list[Span]:
    """Every place the evidence quote occurs in its section, as document-level
    character ranges. Usually one; a quote that occurs twice in the section is
    found by a chunk covering either occurrence. Raises if the quote is not in
    the section at all: the pool and the corpus no longer match, and every
    number measured from here would be wrong without any other sign."""
    spans = []
    start = section_text.find(evidence)
    while start != -1:
        spans.append((section_start + start, section_start + start + len(evidence)))
        start = section_text.find(evidence, start + 1)
    if not spans:
        raise ValueError(f"evidence not in its section; regenerate the pool or restore the corpus: {evidence[:60]!r}")
    return spans


def hits_span(chunk: Located, span: Span) -> bool:
    """True if the chunk overlaps at least HIT_MIN_COVERAGE of the evidence
    span."""
    overlap = min(chunk.char_end, span[1]) - max(chunk.char_start, span[0])
    return overlap >= HIT_MIN_COVERAGE * (span[1] - span[0])


def first_hit_rank(chunks: list[Located], doc_id: str, spans: list[Span]) -> int | None:
    """The 1-based position of the first chunk that hits the evidence, or None
    if no chunk in the list does."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.doc_id == doc_id and any(hits_span(chunk, span) for span in spans):
            return rank
    return None


def first_doc_rank(chunks: list[Located], doc_id: str) -> int | None:
    """The 1-based position of the first chunk from the right article, any
    passage: separates "wrong song" from "right song, wrong passage"."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.doc_id == doc_id:
            return rank
    return None


def within(rank: int | None, k: int) -> bool:
    """hit@k for one question, from the rank of its first hit."""
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None, k: int) -> float:
    """MRR@k's term for one question: 1/rank if the first hit is within the
    top k, else 0 (what MRR is when search returns k chunks)."""
    return 1.0 / rank if within(rank, k) else 0.0


# --- comparing two runs --------------------------------------------------------

def paired_difference(a: list[float], b: list[float]) -> tuple[float, float]:
    """B minus A over the same questions, and its 95% margin, both in the
    metric's own units (a share: 0.02 = 2 points). Paired: each question is
    compared with itself, so questions both runs get right, or both get wrong,
    add no noise; only the ones where they disagree do."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired runs need the same questions, at least two")
    differences = [y - x for x, y in zip(a, b)]
    return statistics.fmean(differences), Z_95 * statistics.stdev(differences) / math.sqrt(len(differences))


def verdict(difference: float, margin: float, min_gap: float) -> str:
    """The tie rule, fixed before any result: B wins, or A wins, only if the
    gap beats both the margin and `min_gap`; anything less is a tie, which
    goes to the simpler configuration."""
    if abs(difference) <= max(margin, min_gap):
        return "tie"
    return "B wins" if difference > 0 else "A wins"


def fewest_gains_to_win(n: int, min_gap: float) -> int | None:
    """The fewest questions B must gain over A, out of n, to win by the tie
    rule in the best case: every gain is a question A failed and B passed,
    and B loses nothing. Fewer gains than this can never win, whatever
    happens on the other questions; that is what a pre-check uses to decide
    whether a full run is worth it. None if even n gains would not win."""
    for gains in range(1, n + 1):
        a = [0.0] * n
        b = [1.0] * gains + [0.0] * (n - gains)
        difference, margin = paired_difference(a, b)
        if verdict(difference, margin, min_gap) == "B wins":
            return gains
    return None
