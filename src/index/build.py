"""Embeds the chunks of one strategy and stores them in a local Qdrant
collection, dense and sparse (BM25) side by side so that search can be hybrid.

One collection per strategy: {corpus_version}__{strategy_id}__{embedding}.
Collections are rebuilt from scratch on every run, never appended to, so a
collection always matches the chunks file it was built from."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from qdrant_client import QdrantClient

from src.config import Config
from src.index.chunking import chunks_path, read_chunks
from src.index.collections import collection_name, open_vector_store
from src.index.embeddings import build_embedding
from src.index.nodes import chunk_to_node
from src.schemas import Chunk

TRIAL_SUFFIX = "__trial"


def build_collection(
    chunks: list[Chunk],
    name: str,
    client: QdrantClient,
    embed_model: BaseEmbedding,
    config: Config,
) -> None:
    """Replaces the collection if it exists. Embedding runs on the CPU and
    takes minutes for a full strategy, so a half-built collection is never
    worth keeping: rerun it."""
    if client.collection_exists(name):
        client.delete_collection(name)

    vector_store = open_vector_store(client, name, config.retrieval.sparse_model)
    nodes = [chunk_to_node(chunk, config.project.uuid_namespace) for chunk in chunks]
    VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model,
        show_progress=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Qdrant collection from a chunks file.")
    parser.add_argument("--strategy", help="strategy id; default: chunking.active")
    parser.add_argument("--embedding", help="embedding candidate; default: embeddings.active")
    parser.add_argument("--limit", type=int, help="trial run on the first N chunks, in a separate collection")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml")).with_overrides(embedding=args.embedding)
    strategy_id = args.strategy or config.chunking.active
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(config.paths.models))

    chunks = read_chunks(chunks_path(config, strategy_id))
    name = collection_name(config, strategy_id)
    if args.limit:
        chunks = chunks[: args.limit]
        name += TRIAL_SUFFIX

    embed_model = build_embedding(config.embeddings, config.paths.models)
    config.paths.qdrant_experiments.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.paths.qdrant_experiments))
    try:
        build_collection(chunks, name, client, embed_model, config)
        print(f"{name}: {client.count(name).count} points")
    finally:
        client.close()


if __name__ == "__main__":
    main()
