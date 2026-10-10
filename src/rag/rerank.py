"""The reranker: a cross-encoder that reads the question together with each
candidate chunk and scores how well that chunk answers it.

Hybrid search compares the question with each chunk separately (a vector, a
bag of words), so it is fast but coarse. A cross-encoder reads both at once,
which orders better and costs one model pass per candidate; that is why it
only reorders the first `top_n` candidates, never the whole collection.

Runs locally through fastembed, like the embedder, so it never uses the
user's key.

Off in the app (`reranking.active: null`). In the reranker stage it gained
1.3 points, a tie. Checked again on the final setup: 34 questions could be
rescued and 25 were needed to win, but the best model rescued 20. It stays so
that result can be reproduced:
`python -m src.eval.run_eval --reranker minilm`."""

from __future__ import annotations

from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.config import RerankingConfig
from src.index.nodes import retrieval_text
from src.schemas import Chunk


def reorder(chunks: list[Chunk], scores: list[float]) -> list[Chunk]:
    """The chunks sorted by their reranker score, best first, each carrying
    that score. The fused score is dropped: the two scales are not comparable."""
    if len(chunks) != len(scores):
        raise ValueError(f"{len(chunks)} chunks but {len(scores)} scores")
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk.model_copy(update={"score": score}) for chunk, score in ranked]


class Reranker:
    def __init__(self, model_name: str, cache_dir: Path, top_n: int) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))
        self._top_n = top_n

    @property
    def max_tokens(self) -> int:
        """Where fastembed cuts each question + chunk pair. It comes from the
        model's tokenizer files, not from the model's description, so it is
        read here rather than trusted: a pair longer than this is scored on
        its first `max_tokens` tokens only."""
        return self._model.model.tokenizer.truncation["max_length"]

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Reorders the first `top_n` chunks; the rest follow in their fused
        order, after them."""
        head, tail = chunks[: self._top_n], chunks[self._top_n :]
        scores = list(self._model.rerank(query, [retrieval_text(chunk) for chunk in head]))
        return reorder(head, scores) + tail


def build_reranker(config: RerankingConfig, cache_dir: Path) -> Reranker | None:
    """The reranker named by `reranking.active`, or None when it is off."""
    if config.active is None:
        return None
    return Reranker(config.candidates[config.active], cache_dir, config.top_n)
