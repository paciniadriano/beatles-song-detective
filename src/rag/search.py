"""The retrieval boundary: `search(query, filters, top_k) -> list[Chunk]`.

The agent, the app and the evaluation all call this and nothing else. What is
inside (LlamaIndex, Qdrant, hybrid fusion) is invisible to them, which is what
makes it possible to change the embedding model or the fusion without touching
anything above. The signature never changes.

The index, the embedder and the retrieval settings are handed to the Searcher
when it is built, by an entry point, rather than read from globals here."""

from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from qdrant_client import QdrantClient

from src.config import Config, RetrievalConfig
from src.index.collections import collection_name, open_vector_store
from src.index.embeddings import build_embedding
from src.index.nodes import CHUNK_KEYS, node_to_chunk
from src.rag.rerank import Reranker, build_reranker
from src.schemas import Chunk

Filters = dict[str, str | list[str]]

QUERY_MODES = {
    "hybrid": VectorStoreQueryMode.HYBRID,
    "dense": VectorStoreQueryMode.DEFAULT,
    "sparse": VectorStoreQueryMode.SPARSE,
}


def build_filters(filters: Filters | None) -> MetadataFilters | None:
    """A list value means "any of these" (IN, which Qdrant maps to MatchAny);
    a single value means equality, which on a list-valued key such as
    `topics` matches a chunk that has that value among its own.
    FilterOperator.ANY is not supported by Qdrant and raises (CLAUDE.md, trap 6). An unknown key raises too: it would
    otherwise match nothing and look like an empty index."""
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if key not in CHUNK_KEYS:
            raise ValueError(f"Cannot filter on {key!r}; filterable keys: {CHUNK_KEYS}")
        operator = FilterOperator.IN if isinstance(value, list) else FilterOperator.EQ
        conditions.append(MetadataFilter(key=key, value=value, operator=operator))
    return MetadataFilters(filters=conditions)


def select_chunks(chunks: list[Chunk], top_k: int, max_per_section: int | None) -> list[Chunk]:
    """Keeps the order it is given (best first), takes at most
    `max_per_section` per section, then the first `top_k`. Overlapping chunks
    of one section would otherwise fill the top-k, wasting the LLM's context
    and starving questions that need several sources (CLAUDE.md, trap 10).
    The cap runs last, on the final order (fused, or reranked)."""
    selected: list[Chunk] = []
    per_section: dict[str, int] = {}
    for chunk in chunks:
        taken = per_section.get(chunk.section_id, 0)
        if max_per_section is not None and taken >= max_per_section:
            continue
        per_section[chunk.section_id] = taken + 1
        selected.append(chunk)
        if len(selected) == top_k:
            break
    return selected


class Searcher:
    """One open collection plus the settings to query it."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embed_model: BaseEmbedding,
        settings: RetrievalConfig,
        reranker: Reranker | None = None,
    ) -> None:
        self._client = client
        self._embed_model = embed_model
        self._settings = settings
        self.reranker = reranker
        self._store = open_vector_store(client, collection, settings.sparse_model)

    def search(self, query: str, filters: Filters | None = None, top_k: int | None = None) -> list[Chunk]:
        """Returns up to `top_k` chunks, best first.

        The agent never passes `filters`: a wrong filter hides the answer, and
        nothing in the evaluation measures whether the model picks them well
        (CLAUDE.md, "Agent and app"). They serve main() below, for checking
        the index by doc_type or topic."""
        top_k = top_k or self._settings.final_top_k
        chunks = self._query(query, filters)
        if self.reranker is not None:
            chunks = self.reranker.rerank(query, chunks)
        return select_chunks(chunks, top_k, self._settings.max_per_section)

    def _query(self, query: str, filters: Filters | None) -> list[Chunk]:
        settings = self._settings
        result = self._store.query(
            VectorStoreQuery(
                query_str=query,
                query_embedding=self._embed_model.get_query_embedding(query),
                mode=QUERY_MODES[settings.mode],
                similarity_top_k=settings.similarity_top_k,
                sparse_top_k=settings.sparse_top_k,
                hybrid_top_k=settings.hybrid_top_k,
                alpha=settings.alpha,
                filters=build_filters(filters),
            )
        )
        scores = result.similarities or [None] * len(result.nodes)
        chunks = [node_to_chunk(node, score) for node, score in zip(result.nodes, scores)]
        return sorted(chunks, key=lambda chunk: chunk.score or 0.0, reverse=True)

    def close(self) -> None:
        """Local Qdrant locks its storage folder until the client is closed
        (CLAUDE.md, trap 8)."""
        self._client.close()


def open_searcher(config: Config, collection: str, store_path: Path) -> Searcher:
    """Raises at once if the store has no such collection. Qdrant itself
    would not: every search would come back empty, and the app would start
    fine and then fail on every question (a typo in chunking.active, or a
    config change deployed without publishing its collection)."""
    client = QdrantClient(path=str(store_path))
    if not client.collection_exists(collection):
        client.close()
        raise ValueError(f"{collection} is not in {store_path}: build or publish it first")
    embed_model = build_embedding(config.embeddings, config.paths.models)
    reranker = build_reranker(config.reranking, config.paths.models)
    return Searcher(client, collection, embed_model, config.retrieval, reranker)


def main() -> None:
    """Prints what search() returns for one question, for checking by eye."""
    parser = argparse.ArgumentParser(description="Ask the index a question and print the chunks.")
    parser.add_argument("query")
    parser.add_argument("--strategy", help="strategy id; default: chunking.active")
    parser.add_argument("--doc-type", help="filter on doc_type")
    parser.add_argument("--topic", help="filter on a topic (config.yaml, `topics`)")
    parser.add_argument("--reranker", help="reranker candidate, or none; default: reranking.active")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml")).with_overrides(reranker=args.reranker)
    strategy_id = args.strategy or config.chunking.active
    collection = collection_name(config, strategy_id)
    filters = {"doc_type": args.doc_type, "topics": args.topic}
    filters = {key: value for key, value in filters.items() if value}

    searcher = open_searcher(config, collection, config.paths.qdrant_experiments)
    try:
        for rank, chunk in enumerate(searcher.search(args.query, filters), start=1):
            print(f"\n{rank}. score {chunk.score:.3f}  {chunk.title} > {chunk.section_path}")
            print(f"   {chunk.text[:300]}")
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
