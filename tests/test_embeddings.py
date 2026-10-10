"""Build-time assertions 6 and 7: the two checks that prove the embedder is
doing what the whole evaluation assumes, and that fail without any error
message when it is not.

These load the real nomic model (about 520 MB, cached in data/models), so on a
fresh clone they are skipped rather than failed, like the corpus tests."""

from pathlib import Path

import pytest
from llama_index.core.schema import MetadataMode
from tokenizers import Tokenizer

from src.config import Config
from src.index.chunking import chunks_path, read_chunks
from src.index.embeddings import NomicFastEmbed, build_embedding
from src.index.nodes import chunk_to_node

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = Config.load(REPO_ROOT / "config.yaml")
MODELS_DIR = REPO_ROOT / CONFIG.paths.models


@pytest.fixture(scope="module")
def embedder() -> NomicFastEmbed:
    if not any(MODELS_DIR.glob("models--*")):
        pytest.skip(f"No embedding model cached in {MODELS_DIR}; run build.py once first")
    return build_embedding(CONFIG.embeddings, MODELS_DIR)


def test_query_and_document_embeddings_differ(embedder: NomicFastEmbed) -> None:
    """Assertion 6. If the task prefixes were missing, both calls would return
    the same vector and retrieval would quietly lose accuracy."""
    text = "Who played the flugelhorn on Penny Lane?"
    assert embedder.get_query_embedding(text) != embedder.get_text_embedding(text)


def test_each_prefix_is_applied_exactly_once(embedder: NomicFastEmbed) -> None:
    """A doubled prefix would also make the two vectors differ, so assertion 6
    alone cannot catch it. Compare against the raw model with the prefix typed
    by hand."""
    text = "The vocal was doubled."
    raw_model = embedder._model
    expected_doc = list(raw_model.embed([f"{embedder.doc_prefix}{text}"]))[0].tolist()
    expected_query = list(raw_model.embed([f"{embedder.query_prefix}{text}"]))[0].tolist()
    assert embedder.get_text_embedding(text) == pytest.approx(expected_doc, abs=1e-5)
    assert embedder.get_query_embedding(text) == pytest.approx(expected_query, abs=1e-5)


def test_embedder_truncation_covers_the_largest_chunk(embedder: NomicFastEmbed) -> None:
    """Assertion 7 (CLAUDE.md, trap 13). fastembed truncates silently at the
    tokenizer's configured maximum, which can sit below the model's real
    capacity. If it cut the largest chunk, that arm would be embedded from its
    first half and the evaluation would blame the chunk size for a bug.

    Measured with the embedder's own tokenizer, on the text it really embeds:
    header plus body, with the document prefix."""
    largest_strategy = max(CONFIG.chunking.strategies, key=lambda s: s.size)
    path = REPO_ROOT / chunks_path(CONFIG, largest_strategy.id)
    if not path.exists():
        pytest.skip(f"{path} not found; run chunking.py first")

    tokenizer = embedder._model.model.tokenizer
    limit = tokenizer.truncation["max_length"]
    untruncated = Tokenizer.from_str(tokenizer.to_str())
    untruncated.no_truncation()

    texts = []
    for chunk in read_chunks(path):
        node = chunk_to_node(chunk, CONFIG.project.uuid_namespace)
        texts.append(embedder.doc_prefix + node.get_content(MetadataMode.EMBED))
    longest = max(len(encoding.ids) for encoding in untruncated.encode_batch(texts))

    assert longest <= limit, f"largest chunk is {longest} tokens, embedder truncates at {limit}"
