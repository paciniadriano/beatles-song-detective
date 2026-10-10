"""Copies the winning collection into data/qdrant_store, the folder the
Space downloads.

data/qdrant_experiments holds every collection the evaluation built (one per
strategy and embedding). The app needs only one: the active strategy with
the active embedding, the exact collection the final numbers were measured
on. Copying it, instead of embedding the corpus again, guarantees that what
ships is what was evaluated; a search run on both stores at the end checks
that they answer alike.

    python -m src.index.publish

The store is rebuilt from scratch on every run, so it never holds a stale
collection, and Qdrant's `.lock` file is removed: it belongs to the process
that had the folder open, and a published copy must not carry it
(CLAUDE.md, trap 8). Upload to the Hugging Face dataset repo is a separate
step."""

from __future__ import annotations

import shutil
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import Config
from src.index.collections import LOCK_FILE, collection_name
# The one index module that imports the retrieval layer: publish.py is a
# command-line step run after the evaluation, it checks the copy by searching
# both stores, and nothing imports it (CLAUDE.md, section 4).
from src.rag.search import open_searcher

# Any in-scope question works: the check is that both stores return the same
# chunks in the same order, not that the answer is right.
CHECK_QUERY = "How did they make John's voice sound distant on Tomorrow Never Knows?"


def publish_collection(source: Path, destination: Path, name: str) -> int:
    """Copies one collection from `source` into an empty `destination` and
    returns its number of points. Raises if the collection does not exist
    (before touching `destination`) or the copy does not match it."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must be different folders")

    source_client = QdrantClient(path=str(source))
    try:
        if not source_client.collection_exists(name):
            raise ValueError(f"{name} is not in {source}: build it first with src.index.build")
        if destination.exists():
            shutil.rmtree(destination)
        destination_client = QdrantClient(path=str(destination))
        try:
            source_client.migrate(destination_client, collection_names=[name])
            expected = source_client.count(name).count
            copied = destination_client.count(name).count
        finally:
            destination_client.close()
    finally:
        source_client.close()

    if copied != expected:
        raise ValueError(f"copied {copied} points of {expected}")
    (destination / LOCK_FILE).unlink(missing_ok=True)
    return copied


def chunk_ids(config: Config, name: str, store: Path, query: str) -> list[str]:
    searcher = open_searcher(config, name, store)
    try:
        return [chunk.chunk_id for chunk in searcher.search(query)]
    finally:
        searcher.close()


def main() -> None:
    config = Config.load(Path("config.yaml"))
    name = collection_name(config, config.chunking.active)
    points = publish_collection(config.paths.qdrant_experiments, config.paths.qdrant_store, name)
    print(f"{name}: {points} points -> {config.paths.qdrant_store}")

    evaluated = chunk_ids(config, name, config.paths.qdrant_experiments, CHECK_QUERY)
    published = chunk_ids(config, name, config.paths.qdrant_store, CHECK_QUERY)
    (config.paths.qdrant_store / LOCK_FILE).unlink(missing_ok=True)  # the check opened it again
    if published != evaluated:
        raise ValueError(f"the published store answers differently: {published} vs {evaluated}")
    print(f"Check passed: both stores return the same {len(published)} chunks for a test query")


if __name__ == "__main__":
    main()
