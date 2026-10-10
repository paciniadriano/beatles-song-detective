"""Tests for publish.py on small local Qdrant folders built here, with the
same layout as the real collections: a named dense vector and a named sparse
one (hybrid search)."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from src.index.collections import LOCK_FILE
from src.index.publish import publish_collection

DENSE = "text-dense"
SPARSE = "text-sparse"


def _build(path: Path, names: list[str], points: int = 3) -> None:
    client = QdrantClient(path=str(path))
    try:
        for name in names:
            client.create_collection(
                name,
                vectors_config={DENSE: models.VectorParams(size=4, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams()},
            )
            client.upsert(name, [
                models.PointStruct(
                    id=i,
                    vector={
                        DENSE: [0.1 * (i + 1), 0.2, 0.3, 0.4],
                        SPARSE: models.SparseVector(indices=[i, 10], values=[1.0, 0.5]),
                    },
                    payload={"chunk_id": f"{name}__c{i}"},
                )
                for i in range(points)
            ])
    finally:
        client.close()


def test_only_the_named_collection_is_copied_whole(tmp_path: Path) -> None:
    source, destination = tmp_path / "experiments", tmp_path / "store"
    _build(source, ["winner", "loser"])
    assert publish_collection(source, destination, "winner") == 3

    client = QdrantClient(path=str(destination))
    try:
        assert [c.name for c in client.get_collections().collections] == ["winner"]
        point = client.retrieve("winner", [1], with_vectors=True)[0]
        assert point.payload == {"chunk_id": "winner__c1"}
        assert point.vector[SPARSE].indices == [1, 10]
    finally:
        client.close()


def test_the_published_store_has_no_lock_file(tmp_path: Path) -> None:
    source, destination = tmp_path / "experiments", tmp_path / "store"
    _build(source, ["winner"])
    publish_collection(source, destination, "winner")
    assert not (destination / LOCK_FILE).exists()


def test_publishing_again_replaces_the_old_store(tmp_path: Path) -> None:
    source, destination = tmp_path / "experiments", tmp_path / "store"
    _build(source, ["winner"])
    _build(destination, ["stale"])
    publish_collection(source, destination, "winner")
    client = QdrantClient(path=str(destination))
    try:
        assert [c.name for c in client.get_collections().collections] == ["winner"]
    finally:
        client.close()


def test_a_missing_collection_is_refused_without_touching_the_old_store(tmp_path: Path) -> None:
    source, destination = tmp_path / "experiments", tmp_path / "store"
    _build(source, ["other"])
    _build(destination, ["previous"])
    with pytest.raises(ValueError, match="build it first"):
        publish_collection(source, destination, "winner")
    client = QdrantClient(path=str(destination))
    try:
        assert [c.name for c in client.get_collections().collections] == ["previous"]
    finally:
        client.close()


def test_source_and_destination_cannot_be_the_same_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        publish_collection(tmp_path, tmp_path, "winner")
