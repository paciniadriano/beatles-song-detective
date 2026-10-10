"""Checks that every built Qdrant collection matches the chunks file it was
built from. build.py embeds on the CPU for minutes and can be interrupted or
truncated (`--limit` builds a trial collection), and a collection with missing
points still answers queries: retrieval metrics would simply be lower and the
evaluation would blame the chunking strategy for it.

Like test_corpus.py, these read data that a previous step produced. data/ is
gitignored, so on a fresh clone they are skipped rather than failed. Local
Qdrant locks its storage folder (trap 8): close any running app or build
before running them."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.config import Config
from src.index.collections import collection_name
from src.index.chunking import chunks_path, read_chunks
from src.index.nodes import point_id

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = Config.load(REPO_ROOT / "config.yaml")
QDRANT_DIR = REPO_ROOT / CONFIG.paths.qdrant_experiments
CHUNKS_DIR = REPO_ROOT / CONFIG.paths.chunks / CONFIG.project.corpus_version

# How many offending point ids to show in a failure message.
MAX_REPORTED = 10


def _built_strategies() -> list[str]:
    """Strategies with both a chunks file and a full (non-trial) collection.
    Evaluated at collection time so each one becomes its own test case."""
    if not QDRANT_DIR.exists() or not CHUNKS_DIR.exists():
        return []
    built = []
    for file in sorted(CHUNKS_DIR.glob("*.jsonl")):
        expected = collection_name(CONFIG, file.stem)
        if (QDRANT_DIR / "collection" / expected).exists():
            built.append(file.stem)
    return built


@pytest.fixture(scope="module")
def client() -> QdrantClient:
    try:
        qdrant = QdrantClient(path=str(QDRANT_DIR))
    except RuntimeError as error:
        pytest.skip(f"Qdrant storage is locked by another process: {error}")
    yield qdrant
    qdrant.close()


def _stored_points(client: QdrantClient, name: str) -> dict[str, dict]:
    """Every point of a collection, keyed by point id, with its vectors."""
    points = {}
    offset = None
    while True:
        batch, offset = client.scroll(name, limit=256, offset=offset, with_vectors=True)
        for point in batch:
            points[str(point.id)] = point.vector
        if offset is None:
            return points


def _report(failures: list[str]) -> str:
    return f"{len(failures)} failing: {', '.join(failures[:MAX_REPORTED])}"


@pytest.mark.parametrize("strategy_id", _built_strategies())
def test_point_count_equals_chunk_count(client: QdrantClient, strategy_id: str) -> None:
    chunks = read_chunks(chunks_path(CONFIG, strategy_id))
    name = collection_name(CONFIG, strategy_id)
    assert client.count(name).count == len(chunks)


@pytest.mark.parametrize("strategy_id", _built_strategies())
def test_every_chunk_has_its_point(client: QdrantClient, strategy_id: str) -> None:
    """Equal counts could hide one missing and one stray point; comparing the
    ids catches that too. Ids are deterministic uuid5 values (trap 1)."""
    chunks = read_chunks(chunks_path(CONFIG, strategy_id))
    expected = {point_id(CONFIG.project.uuid_namespace, c.chunk_id) for c in chunks}
    stored = set(_stored_points(client, collection_name(CONFIG, strategy_id)))
    assert stored == expected, (
        f"missing: {_report(sorted(expected - stored))}; stray: {_report(sorted(stored - expected))}"
    )


@pytest.mark.parametrize("strategy_id", _built_strategies())
def test_every_point_has_dense_and_sparse_vectors(client: QdrantClient, strategy_id: str) -> None:
    """Hybrid search silently degrades to dense-only if a point lacks its
    sparse vector, and an all-zero dense vector is retrievable but meaningless."""
    failures = []
    for point_id_, vectors in _stored_points(client, collection_name(CONFIG, strategy_id)).items():
        dense = vectors.get("text-dense")
        sparse = vectors.get("text-sparse-new")
        if not dense or not any(dense) or sparse is None or not sparse.indices:
            failures.append(point_id_)
    assert not failures, _report(failures)

