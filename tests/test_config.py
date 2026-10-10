"""Tests for the command-line overrides of config.yaml (Config.with_overrides),
which build.py and run_eval.py use for their command-line flags."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Config

CONFIG = Config.load(Path(__file__).resolve().parent.parent / "config.yaml")


def test_an_override_changes_only_the_copy() -> None:
    copy = CONFIG.with_overrides(embedding="openai", alpha=1)
    assert (copy.embeddings.active, copy.retrieval.alpha) == ("openai", 1)
    assert (CONFIG.embeddings.active, CONFIG.retrieval.alpha) == ("nomic", 0.5)


def test_no_override_keeps_the_file_values() -> None:
    assert CONFIG.with_overrides() == CONFIG


def test_an_unknown_embedding_is_refused() -> None:
    with pytest.raises(ValidationError, match="embeddings.active"):
        CONFIG.with_overrides(embedding="bge")


def test_the_reranker_can_be_turned_on_and_off() -> None:
    on = CONFIG.with_overrides(reranker="minilm")
    assert on.reranking.active == "minilm"
    assert on.with_overrides(reranker="none").reranking.active is None


def test_an_unknown_reranker_is_refused() -> None:
    with pytest.raises(ValidationError, match="reranking.active"):
        CONFIG.with_overrides(reranker="cohere")


def test_top_k_and_the_cap_can_be_overridden_and_the_cap_removed() -> None:
    copy = CONFIG.with_overrides(final_top_k=7, max_per_section=3)
    assert (copy.retrieval.final_top_k, copy.retrieval.max_per_section) == (7, 3)
    assert CONFIG.with_overrides(max_per_section="none").retrieval.max_per_section is None


def test_a_final_top_k_outside_k_values_or_a_cap_below_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="final_top_k"):
        CONFIG.with_overrides(final_top_k=9)
    with pytest.raises(ValidationError):
        CONFIG.with_overrides(max_per_section=0)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_an_alpha_outside_zero_to_one_is_refused(alpha: float) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        CONFIG.with_overrides(alpha=alpha)
