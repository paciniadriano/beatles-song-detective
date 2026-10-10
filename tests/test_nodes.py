"""Tests for src/index/nodes.py: what goes into Qdrant must come back out as the
same Chunk, and only the header may leak into the embedded text."""

import uuid

from llama_index.core.schema import MetadataMode

from src.index.nodes import chunk_to_node, node_to_chunk, point_id
from src.schemas import Chunk

NAMESPACE = "6f1b7c2e-0000-4000-8000-000000000000"


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="wp_1__s2__c0", doc_id="wp_1", section_id="wp_1__s2", doc_type="song",
        topics=["recording"], title="Test Song (song)", section_path="Music > Recording",
        url="https://example.org/Test_Song", revid=10,
        header={"Song": "Test Song", "Album": "Single/Outtake", "Section": "Music > Recording"},
        text="The vocal was doubled.", char_start=100, char_end=122,
    )


def test_point_id_is_a_deterministic_uuid() -> None:
    """Qdrant rejects readable ids like 'wp_1__s2__c0' (trap 1)."""
    first = point_id(NAMESPACE, "wp_1__s2__c0")
    assert uuid.UUID(first)
    assert first == point_id(NAMESPACE, "wp_1__s2__c0")
    assert first != point_id(NAMESPACE, "wp_1__s2__c1")


def test_round_trip_returns_the_same_chunk() -> None:
    chunk = _chunk()
    assert node_to_chunk(chunk_to_node(chunk, NAMESPACE)) == chunk


def test_round_trip_keeps_the_header_order() -> None:
    restored = node_to_chunk(chunk_to_node(_chunk(), NAMESPACE))
    assert list(restored.header) == ["Song", "Album", "Section"]


def test_embedded_text_is_header_plus_text_and_nothing_else() -> None:
    """Ids and offsets are payload, not content: they must not be embedded."""
    embedded = chunk_to_node(_chunk(), NAMESPACE).get_content(MetadataMode.EMBED)
    assert embedded == (
        "Song: Test Song\nAlbum: Single/Outtake\nSection: Music > Recording\n\nThe vocal was doubled."
    )
