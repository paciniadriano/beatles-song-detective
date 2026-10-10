"""The embedding models, and the one place task prefixes are applied.

nomic-embed-text requires "search_query: " on queries and "search_document: "
on passages, and fastembed does not add them: its `query_embed()` and
`passage_embed()` both call `embed()` unchanged. Every prefix in this project
is applied here and nowhere else, so that a chunk cannot reach Qdrant with
the wrong one -- or with two of them."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from pydantic import Field

from src.config import EmbeddingsConfig


class NomicFastEmbed(FastEmbedEmbedding):
    """FastEmbedEmbedding with nomic's task prefixes applied.

    Only the two low-level hooks are overridden: `_get_text_embedding`
    delegates to `_get_text_embeddings`, and both async variants delegate to
    their sync counterparts, so prefixing here covers every path into the
    model. The prefixes deliberately do not reach the sparse encoder --
    BM25 receives chunk text through `node.get_content(MetadataMode.EMBED)`
    inside QdrantVectorStore, bypassing this class entirely, which is
    correct: BM25 must not index the literal string "search_document:"."""

    query_prefix: str = Field(default="")
    doc_prefix: str = Field(default="")

    def _get_query_embedding(self, query: str) -> list[float]:
        return super()._get_query_embedding(f"{self.query_prefix}{query}")

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return super()._get_text_embeddings([f"{self.doc_prefix}{text}" for text in texts])


def build_embedding(config: EmbeddingsConfig, cache_dir: Path) -> BaseEmbedding:
    """Builds the embedding model named by `embeddings.active`. Raises on an
    unknown provider rather than falling back to a default, because a silent
    fallback would embed the corpus with the wrong model and invalidate every
    number measured against it."""
    candidate = config.candidates[config.active]

    if candidate.provider == "fastembed":
        cache_dir.mkdir(parents=True, exist_ok=True)
        return NomicFastEmbed(
            model_name=candidate.model,
            cache_dir=str(cache_dir),
            query_prefix=candidate.query_prefix or "",
            doc_prefix=candidate.doc_prefix or "",
        )

    if candidate.provider == "openai":        
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=candidate.model)

    raise ValueError(
        f"Unknown embedding provider {candidate.provider!r} for candidate {config.active!r}"
    )
