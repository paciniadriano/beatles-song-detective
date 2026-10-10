"""Tests for the pure parts of src/rag/search.py: the filter translation and
the per-section cap. Querying the index itself is checked by eye with
`python -m src.rag.search` and measured by the evaluation."""

from pathlib import Path

import pytest
from llama_index.core.vector_stores.types import FilterOperator

from src.config import Config
from src.rag.search import build_filters, open_searcher, select_chunks
from src.schemas import Chunk


def _chunk(chunk_id: str, section_id: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="wp_1", section_id=section_id, doc_type="song",
        topics=["recording"], title="T", section_path="P", url="u", revid=1,
        header={"Song": "T"}, text="x", char_start=0, char_end=1, score=score,
    )


def test_no_filters_means_no_filter_object() -> None:
    assert build_filters(None) is None
    assert build_filters({}) is None


def test_single_value_is_equality_and_a_list_is_in() -> None:
    """ANY is not supported by Qdrant; IN maps to MatchAny (trap 6)."""
    result = build_filters({"doc_type": "song", "topics": ["recording", "personnel"]})
    operators = {f.key: f.operator for f in result.filters}
    assert operators == {"doc_type": FilterOperator.EQ, "topics": FilterOperator.IN}


def test_unknown_filter_key_raises() -> None:
    with pytest.raises(ValueError):
        build_filters({"decade": "1960s"})


def test_the_order_given_is_kept() -> None:
    """The list arrives best first (fused, or reranked); the scores of the two
    are not comparable, so select_chunks must not re-sort by score."""
    chunks = [_chunk("a", "s1", 0.2), _chunk("b", "s2", 0.9), _chunk("c", "s3", 0.5)]
    assert [c.chunk_id for c in select_chunks(chunks, 3, None)] == ["a", "b", "c"]


def test_cap_keeps_at_most_n_chunks_per_section() -> None:
    chunks = [_chunk("a", "s1", 0.9), _chunk("b", "s1", 0.8), _chunk("c", "s1", 0.7), _chunk("d", "s2", 0.1)]
    picked = select_chunks(chunks, 5, max_per_section=2)
    assert [c.chunk_id for c in picked] == ["a", "b", "d"]


def test_cap_of_none_disables_the_cap_and_top_k_still_applies() -> None:
    chunks = [_chunk("a", "s1", 0.9), _chunk("b", "s1", 0.8), _chunk("c", "s1", 0.7)]
    assert [c.chunk_id for c in select_chunks(chunks, 2, None)] == ["a", "b"]


def test_a_missing_collection_fails_at_once(tmp_path: Path) -> None:
    """Checked before any model loads: an empty Qdrant folder has no
    collections, so opening one must raise instead of searching nothing."""
    config = Config.load(Path(__file__).resolve().parent.parent / "config.yaml")
    with pytest.raises(ValueError, match="not in"):
        open_searcher(config, "corpus_v1__missing__nomic", tmp_path)
