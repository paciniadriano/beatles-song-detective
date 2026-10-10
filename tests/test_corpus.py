"""Checks the invariants that chunking and evaluation rely on, over the real
normalized corpus in data/docs/. Unlike test_normalize.py, these tests do not
exercise functions: they read the data that ingestion already produced and
verify it is safe to build on.

The most important one is the offset check. Chunking adds each section's
char_start to the position of every chunk, and the evaluation compares those
document-level positions against evidence quotes. Wrong offsets would corrupt
every retrieval metric without raising a single error.

data/ is gitignored, so on a fresh clone these tests are skipped rather than
failed."""

from pathlib import Path

import pytest

from src.config import Config
from src.ingest.normalize import load_documents
from src.schemas import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = Config.load(REPO_ROOT / "config.yaml")
DOCS_DIR = REPO_ROOT / CONFIG.paths.docs

# How many offending section ids to show in a failure message. Enough to
# start debugging, short enough to read.
MAX_REPORTED = 10


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    """Loads and schema-validates every document once for the whole module.
    Loading ~270 documents per test would make the suite slow for no gain."""
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.json")):
        pytest.skip(f"No normalized corpus in {DOCS_DIR}; run normalize.py first")
    return load_documents(DOCS_DIR)


def _report(failures: list[str]) -> str:
    """Formats a readable failure message: the count, then the first few ids."""
    shown = ", ".join(failures[:MAX_REPORTED])
    return f"{len(failures)} failing: {shown}"


def test_section_offsets_match_full_text(documents: list[Document]) -> None:
    """full_text[char_start:char_end] must be exactly the section's text.
    This is what makes document-level chunk offsets trustworthy."""
    failures = []
    for doc in documents:
        for section in doc.sections:
            if doc.full_text[section.char_start:section.char_end] != section.text:
                failures.append(section.section_id)
    assert not failures, _report(failures)


def test_sections_are_ordered_and_do_not_overlap(documents: list[Document]) -> None:
    """Each section starts at or after the end of the previous one. Overlapping
    sections would let two sections claim the same evidence quote."""
    failures = []
    for doc in documents:
        previous_end = 0
        for section in doc.sections:
            if section.char_start < previous_end:
                failures.append(section.section_id)
            previous_end = section.char_end
    assert not failures, _report(failures)


def test_no_section_is_empty(documents: list[Document]) -> None:
    """An empty section would become an empty chunk: a vector with no content
    that can still be retrieved."""
    failures = []
    for doc in documents:
        for section in doc.sections:
            if not section.text.strip():
                failures.append(section.section_id)
    assert not failures, _report(failures)


def test_every_document_has_a_revid(documents: list[Document]) -> None:
    """revid freezes the corpus and backs every citation's provenance."""
    failures = [doc.doc_id for doc in documents if doc.revid is None]
    assert not failures, _report(failures)
