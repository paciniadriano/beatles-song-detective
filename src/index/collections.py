"""The Qdrant collections: what each one is called and how it is opened.

Used when building (build.py), publishing (publish.py) and searching
(search.py, the app).
It lives apart from build.py on purpose: build.py needs the chunking and
ingest code, which the Space does not install (requirements-dev.txt), so the
app must be able to name and open a collection without importing it."""

from __future__ import annotations

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import Config

# The file local Qdrant creates in a store while a process has it open. A
# published copy must not carry it (CLAUDE.md, trap 8): publish.py removes it
# after writing the store, and the app after copying the downloaded one.
LOCK_FILE = ".lock"


def collection_name(config: Config, strategy_id: str) -> str:
    """{corpus_version}__{strategy_id}__{embedding}: one collection per
    strategy and embedding model."""
    return f"{config.project.corpus_version}__{strategy_id}__{config.embeddings.active}"


def open_vector_store(client: QdrantClient, name: str, sparse_model: str) -> QdrantVectorStore:
    """The store for one collection, dense and sparse. Always opened with the
    sparse model, even for dense-only queries: a hybrid collection stores
    named vectors, and the store must know their names to query either one.
    build.py and search.py both open it here, so they cannot disagree."""
    return QdrantVectorStore(
        collection_name=name,
        client=client,
        enable_hybrid=True,
        fastembed_sparse_model=sparse_model,
    )
