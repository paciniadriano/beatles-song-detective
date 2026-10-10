"""Tests for the pure parts of src/rag/rerank.py and the text it reads. The
model itself is measured by the evaluation, not here."""

import pytest
from llama_index.core.schema import MetadataMode

from src.index.nodes import chunk_to_node, retrieval_text
from src.rag.rerank import reorder
from src.schemas import Chunk

NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


def _chunk(chunk_id: str, score: float = 0.5) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="wp_1", section_id="wp_1__s2", doc_type="song", topics=["recording"],
        title="Rain", section_path="Recording", url="u", revid=1,
        header={"Song": "Rain", "Album": "Single/Outtake", "Section": "Recording"},
        text="It was recorded in two takes.", char_start=0, char_end=29, score=score,
    )


def test_the_reranker_reads_what_the_embedder_reads() -> None:
    chunk = _chunk("a")
    assert retrieval_text(chunk) == chunk_to_node(chunk, NAMESPACE).get_content(MetadataMode.EMBED)
    assert retrieval_text(chunk).startswith("Song: Rain")


def test_reorder_sorts_by_the_reranker_score_and_keeps_it() -> None:
    chunks = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
    reordered = reorder(chunks, [-2.0, 3.5, 0.1])
    assert [c.chunk_id for c in reordered] == ["b", "c", "a"]
    assert [c.score for c in reordered] == [3.5, 0.1, -2.0]


def test_reorder_refuses_a_missing_score() -> None:
    with pytest.raises(ValueError):
        reorder([_chunk("a"), _chunk("b")], [1.0])
