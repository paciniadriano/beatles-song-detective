"""Tests for the retrieval metrics in src/eval/metrics.py: what counts as a
hit, how ranks become hit@k and MRR, and the paired comparison with its tie
rule. Every number the chunking decision rests on goes through these."""

from dataclasses import dataclass

import pytest

from src.eval.metrics import (
    evidence_spans, first_doc_rank, first_hit_rank, hits_span, paired_difference,
    reciprocal_rank, verdict, within,
)


@dataclass
class _Chunk:
    doc_id: str
    char_start: int
    char_end: int


SECTION = "They recorded it in one take. Lennon sang it twice."


def test_evidence_spans_are_document_offsets_and_find_every_occurrence() -> None:
    assert evidence_spans("one take", SECTION, 100) == [(120, 128)]
    assert evidence_spans("it", "it and it", 0) == [(0, 2), (7, 9)]


def test_evidence_missing_from_its_section_stops_the_run() -> None:
    with pytest.raises(ValueError, match="not in its section"):
        evidence_spans("two takes", SECTION, 0)


def test_a_chunk_hits_when_it_covers_at_least_half_the_quote() -> None:
    span = (100, 140)
    assert hits_span(_Chunk("wp_1", 90, 150), span)       # whole quote
    assert hits_span(_Chunk("wp_1", 120, 200), span)      # exactly half
    assert not hits_span(_Chunk("wp_1", 121, 200), span)  # just under half
    assert not hits_span(_Chunk("wp_1", 0, 90), span)     # elsewhere


def test_the_first_hit_must_be_in_the_right_document() -> None:
    spans = [(100, 140)]
    chunks = [_Chunk("wp_2", 100, 140), _Chunk("wp_1", 0, 50), _Chunk("wp_1", 95, 145)]
    assert first_hit_rank(chunks, "wp_1", spans) == 3   # same offsets in wp_2 do not count
    assert first_doc_rank(chunks, "wp_1") == 2
    assert first_hit_rank(chunks[:2], "wp_1", spans) is None


def test_ranks_become_hit_at_k_and_mrr() -> None:
    assert within(3, 5) and not within(6, 5) and not within(None, 5)
    assert reciprocal_rank(4, k=5) == 0.25
    assert reciprocal_rank(4, k=3) == 0.0      # beyond the k chunks returned
    assert reciprocal_rank(None, k=5) == 0.0


def test_the_paired_margin_only_counts_questions_the_runs_disagree_on() -> None:
    same = [1.0] * 50 + [0.0] * 50
    difference, margin = paired_difference(same, same)
    assert difference == 0.0 and margin == 0.0
    a = [1.0] * 50 + [0.0] * 50
    b = [1.0] * 50 + [1.0] * 10 + [0.0] * 40   # B also gets 10 that A missed
    difference, margin = paired_difference(a, b)
    assert difference == pytest.approx(0.10)
    assert 0.05 < margin < 0.07


def test_a_win_must_beat_both_the_margin_and_the_minimum_gap() -> None:
    assert verdict(0.03, 0.01, 0.02) == "B wins"
    assert verdict(-0.03, 0.01, 0.02) == "A wins"
    assert verdict(0.015, 0.01, 0.02) == "tie"   # beats the margin, not the gap
    assert verdict(0.03, 0.04, 0.02) == "tie"    # beats the gap, not the margin
