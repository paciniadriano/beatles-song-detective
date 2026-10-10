"""Loads config.yaml into a typed object. Only entry points (fetch.py,
build.py, app.py, run_eval.py, ...) call Config.load(); everything below
receives the values it needs as parameters, never as a module-level global."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# The topic normalize.py gives the text before the first heading.
INTRODUCTION_TOPIC = "introduction"

# How "no reranker" and "no cap per section" are written on the command line
# and in experiments.csv.
NO_RERANKER = "none"
NO_CAP = "none"


class ProjectConfig(BaseModel):
    corpus_version: str
    uuid_namespace: str


class PathsConfig(BaseModel):
    raw: Path
    docs: Path
    chunks: Path
    qdrant_experiments: Path
    qdrant_store: Path
    models: Path
    tiktoken_cache: Path


# "(s)" or "(es)" right after a letter, at the end of a word: the only
# parentheses a heading phrase may carry (normalize._phrase_pattern).
_PLURAL_ENDING_RE = re.compile(r"(?<=[a-z])\((?:s|es)\)(?![\w(])")


def check_phrases(phrases: list[str]) -> list[str]:
    """Refuses a phrase with any other parenthesis ("origin (s)", "(ies)"): it
    would load fine and then silently match nothing."""
    for phrase in phrases:
        rest = _PLURAL_ENDING_RE.sub("", phrase.lower())
        if "(" in rest or ")" in rest:
            raise ValueError(f"{phrase!r}: only a plural ending such as 'origin(s)' or 'mix(es)' may use parentheses")
    return phrases


class WikipediaConfig(BaseModel):
    user_agent: str  # a template: fetch.py fills in the contact address
    sleep_seconds: float
    drop_sections: list[str]     # apparatus and lists, unless an in-scope topic is named
    private_sections: list[str]  # always dropped

    @field_validator("drop_sections", "private_sections")
    @classmethod
    def _phrases_are_well_formed(cls, phrases: list[str]) -> list[str]:
        return check_phrases(phrases)


class AuditConfig(BaseModel):
    short_document_max_tokens: int
    short_section_max_tokens: int


class TopicsConfig(BaseModel):
    exact_titles: dict[str, str]
    keywords: dict[str, list[str]]  # topic -> phrases matched as whole words
    fallback: str
    out_of_scope: list[str]

    @field_validator("keywords")
    @classmethod
    def _phrases_are_well_formed(cls, keywords: dict[str, list[str]]) -> dict[str, list[str]]:
        for phrases in keywords.values():
            check_phrases(phrases)
        return keywords

    def names(self) -> set[str]:
        """Every topic normalize.py can assign."""
        return set(self.keywords) | set(self.exact_titles.values()) | {self.fallback, INTRODUCTION_TOPIC}


class ChunkingStrategy(BaseModel):
    """One strategy of the chunking grid. Every strategy uses the same SentenceSplitter;
    `unit` decides what one LlamaIndex Document is ("article" or "section"),
    and the splitter never cuts across two Documents."""
    id: str
    unit: Literal["article", "section"]
    size: int
    overlap: int


class ChunkingConfig(BaseModel):
    tokenizer: str
    header_fields: dict[str, list[str]]
    strategies: list[ChunkingStrategy]
    active: str


class EmbeddingCandidate(BaseModel):
    provider: str
    model: str
    query_prefix: str | None = None
    doc_prefix: str | None = None


class EmbeddingsConfig(BaseModel):
    active: str
    candidates: dict[str, EmbeddingCandidate]


class RetrievalConfig(BaseModel):
    mode: str
    sparse_model: str
    similarity_top_k: int
    sparse_top_k: int
    hybrid_top_k: int
    alpha: float = Field(ge=0, le=1)
    final_top_k: int
    max_per_section: Annotated[int, Field(ge=1)] | None


class RerankingConfig(BaseModel):
    active: str | None
    top_n: int = Field(gt=0)
    candidates: dict[str, str]  # name -> fastembed model


class LLMConfig(BaseModel):
    base_url: str | None
    model: str
    api_key_env: str
    extra_params: dict[str, object]
    max_output_tokens: int
    max_steps: int = Field(ge=1)  # at most this many searches per question
    force_first_tool: bool
    max_question_chars: int
    history_turns: int = Field(ge=0)  # earlier questions and answers the model sees
    prompt_path: Path


class TokenPrices(BaseModel):
    """USD per million tokens for one model. Used for the cost shown under
    each answer and printed after a run, so a stale price misleads the
    estimate but never changes a result.
    `cached_input` is the price of input tokens served from OpenAI's prompt
    cache, a fraction of `input`."""
    input: float
    cached_input: float
    output: float


class EvalGenerateConfig(BaseModel):
    """Which sections generate.py writes questions from (one per eligible
    section), and where the pool and the trial runs are written."""
    model: str
    max_output_tokens: int
    output: Path
    pilot_output: Path
    mention_required_for_doc_types: list[str]
    mention_terms: list[str]


class EvalAuditSampleConfig(BaseModel):
    """The random sample of the pool that a person audits (audit_sample.py)."""
    size: int
    output: Path


class EvalGenerationConfig(BaseModel):
    """The generation set (run_generation.py): the frozen questions, where
    each run's answers are written for grading, and the prompt of the
    closed-book runs (the model answering with no retrieval)."""
    questions: Path
    runs_output: Path
    closed_book_prompt_path: Path


class EvalConfig(BaseModel):
    split_seed: int
    dev_ratio: float
    k_values: list[int]
    runs_output: Path
    experiments_output: Path
    min_meaningful_gap_points: float  # tie rule: a win must also beat this
    generate: EvalGenerateConfig
    audit_sample: EvalAuditSampleConfig
    generation: EvalGenerationConfig


class SpaceConfig(BaseModel):
    repo_id: str  # the Space itself, for src/app/deploy.py
    dataset_repo_env: str  # the environment variable naming the dataset repo
    qdrant_runtime_path: Path
    concurrency_limit: int
    queue_max_size: int


class AppConfig(BaseModel):
    """What the Gradio app shows (src/app/ui.py)."""
    title: str
    subtitle: str
    api_key_label: str
    api_key_help: str  # Markdown
    question_placeholder: str
    examples: list[str]
    footer: str  # Markdown
    css_path: Path
    logo_path: Path  # shown instead of the title text, which becomes its alt text
    font_path: Path  # Inter, embedded in the page so every visitor sees the same font


class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    wikipedia: WikipediaConfig
    audit: AuditConfig
    topics: TopicsConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    reranking: RerankingConfig
    llm: LLMConfig
    model_prices: dict[str, TokenPrices]  # model name -> prices, for cost reports
    eval: EvalConfig
    space: SpaceConfig
    app: AppConfig

    @model_validator(mode="after")
    def _names_refer_to_something(self) -> "Config":
        """A misspelt name here would not fail anywhere else: a topic that does
        not exist silently excludes nothing, and a candidate pool smaller than
        a k silently caps hit@k."""
        if self.embeddings.active not in self.embeddings.candidates:
            raise ValueError(f"embeddings.active: {self.embeddings.active!r} is not one of {sorted(self.embeddings.candidates)}")
        if self.reranking.active is not None and self.reranking.active not in self.reranking.candidates:
            raise ValueError(f"reranking.active: {self.reranking.active!r} is not one of {sorted(self.reranking.candidates)}")
        if self.reranking.top_n > self.retrieval.hybrid_top_k:
            raise ValueError(f"reranking.top_n ({self.reranking.top_n}) is above the {self.retrieval.hybrid_top_k} fused candidates")
        unknown = set(self.topics.out_of_scope) - self.topics.names()
        if unknown:
            raise ValueError(f"topics.out_of_scope: unknown topics {sorted(unknown)}")
        unknown = set(self.eval.generate.mention_required_for_doc_types) - set(self.chunking.header_fields)
        if unknown:
            raise ValueError(f"eval.generate.mention_required_for_doc_types: unknown doc types {sorted(unknown)}")
        for role, model in (("llm.model", self.llm.model), ("eval.generate.model", self.eval.generate.model)):
            if model not in self.model_prices:
                raise ValueError(f"{role}: {model!r} has no entry in model_prices")
        smallest_pool = min(self.retrieval.similarity_top_k, self.retrieval.sparse_top_k, self.retrieval.hybrid_top_k)
        if self.retrieval.final_top_k not in self.eval.k_values:
            raise ValueError(f"retrieval.final_top_k ({self.retrieval.final_top_k}) must be in eval.k_values: it is the hit@k that decides")
        if max(self.eval.k_values) > smallest_pool:
            raise ValueError(f"eval.k_values reaches {max(self.eval.k_values)}, above a retrieval pool of {smallest_pool}")
        return self

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Reads config.yaml once, after loading .env into the environment.
        Every value below this call is passed down explicitly, which is what
        makes an experiment 'change a YAML' rather than 'hunt for constants'
        scattered across modules."""
        _load_dotenv(path.parent / ".env")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def with_overrides(
        self,
        embedding: str | None = None,
        alpha: float | None = None,
        reranker: str | None = None,
        final_top_k: int | None = None,
        max_per_section: int | str | None = None,
        max_steps: int | None = None,
    ) -> "Config":
        """A copy with the command-line choices applied; None keeps the value in
        config.yaml, reranker "none" turns the reranker off and max_per_section
        "none" removes the cap. The copy goes through the same validators as the
        file, so an unknown name, an alpha outside 0-1 or a final_top_k missing
        from k_values fails before the run starts."""
        data = self.model_dump()
        if embedding is not None:
            data["embeddings"]["active"] = embedding
        if alpha is not None:
            data["retrieval"]["alpha"] = alpha
        if reranker is not None:
            data["reranking"]["active"] = None if reranker == NO_RERANKER else reranker
        if final_top_k is not None:
            data["retrieval"]["final_top_k"] = final_top_k
        if max_per_section is not None:
            data["retrieval"]["max_per_section"] = None if max_per_section == NO_CAP else max_per_section
        if max_steps is not None:
            data["llm"]["max_steps"] = max_steps
        return Config.model_validate(data)


def _load_dotenv(path: Path) -> None:
    """Loads simple KEY=VALUE entries without adding a runtime dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
