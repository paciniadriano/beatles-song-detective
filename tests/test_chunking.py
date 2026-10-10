"""Tests for the pure functions in src/index/chunking.py, on a small synthetic
article so that every expected value can be worked out by hand.

The real corpus is covered elsewhere: chunk_corpus() runs the offset and header
assertions over the whole corpus every time chunking runs."""

import os
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
import tiktoken

from src.config import ChunkingConfig, ChunkingStrategy, Config
from src.index.chunking import (
    check_chunks,
    chunk_document,
    header_fields_for,
    header_value,
    make_splitter,
    section_at,
)
from src.ingest.seeds import NON_ALBUM
from src.schemas import Document, DocumentStats, Section

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", str(REPO_ROOT / Config.load(REPO_ROOT / "config.yaml").paths.tiktoken_cache)
)

SECTION_SEPARATOR = "\n\n"

CONFIG = ChunkingConfig(
    tokenizer="cl100k_base",
    header_fields={"song": ["Song", "Album", "Section"], "album": ["Album", "Section"]},
    strategies=[],
    active="section",
)
SECTION_STRATEGY = ChunkingStrategy(id="section", unit="section", size=100, overlap=10)
ARTICLE_STRATEGY = ChunkingStrategy(id="article", unit="article", size=100, overlap=10)


def _paragraph(word: str, sentences: int) -> str:
    """Enough short sentences to be split into several chunks."""
    return " ".join(f"The {word} was recorded in one long session number {n}." for n in range(sentences))


def _make_document(doc_type: str = "song") -> Document:
    """Two sections: a long one that must split, and a short one that must not."""
    texts = [_paragraph("guitar", 30), _paragraph("piano", 3)]
    sections = []
    cursor = 0
    for order, text in enumerate(texts):
        sections.append(
            Section(
                section_id=f"wp_1__s{order}", title=f"Part {order}",
                path=f"Part {order}", topics=["recording"], text=text,
                char_start=cursor, char_end=cursor + len(text), token_count=len(text.split()),
            )
        )
        cursor += len(text) + len(SECTION_SEPARATOR)
    full_text = SECTION_SEPARATOR.join(texts)
    return Document(
        doc_id="wp_1", pageid=1, revid=10, revised_at=datetime(2026, 1, 1),
        title="Test Song (song)", url="https://example.org/Test_Song", license="CC BY-SA 4.0",
        doc_type=doc_type, entity="Test Song", album=NON_ALBUM if doc_type == "song" else None,
        sections=sections,
        full_text=full_text,
        stats=DocumentStats(section_count=2, token_count=0, char_count=len(full_text)),
    )


@pytest.fixture(scope="module")
def tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def test_article_unit_drops_the_section_field() -> None:
    fields = header_fields_for("article", "song", CONFIG.header_fields)
    assert fields == ["Song", "Album"]
    assert header_fields_for("section", "song", CONFIG.header_fields) == ["Song", "Album", "Section"]


def test_song_header_shows_its_album() -> None:
    assert header_value("Album", _make_document(), None) == NON_ALBUM


def test_album_document_uses_its_own_name() -> None:
    doc = _make_document(doc_type="album")
    assert header_value("Album", doc, None) == "Test Song"


def test_only_a_song_has_an_album() -> None:
    song = _make_document()
    with pytest.raises(ValidationError, match="a song needs an album"):
        Document.model_validate({**song.model_dump(), "album": None})
    with pytest.raises(ValidationError, match="only a song has one"):
        Document.model_validate({**song.model_dump(), "doc_type": "studio"})


def test_section_at_finds_the_section_where_an_offset_falls() -> None:
    doc = _make_document()
    second = doc.sections[1]
    assert section_at(doc, 0).section_id == "wp_1__s0"
    assert section_at(doc, second.char_start).section_id == "wp_1__s1"
    assert section_at(doc, second.char_end - 1).section_id == "wp_1__s1"


def test_section_unit_never_mixes_sections(tokenizer: tiktoken.Encoding) -> None:
    doc = _make_document()
    chunks = chunk_document(doc, SECTION_STRATEGY, CONFIG, make_splitter(SECTION_STRATEGY, tokenizer))
    sections = {s.section_id: s for s in doc.sections}
    assert len({c.section_id for c in chunks}) == 2
    for chunk in chunks:
        section = sections[chunk.section_id]
        assert section.char_start <= chunk.char_start
        assert chunk.char_end <= section.char_end


def test_offsets_are_document_level(tokenizer: tiktoken.Encoding) -> None:
    """The trap: in a section arm the splitter's offsets are relative to the
    section, so a chunk of the second section must start after the first."""
    doc = _make_document()
    chunks = chunk_document(doc, SECTION_STRATEGY, CONFIG, make_splitter(SECTION_STRATEGY, tokenizer))
    for chunk in chunks:
        assert doc.full_text[chunk.char_start:chunk.char_end] == chunk.text
    second_section_chunks = [c for c in chunks if c.section_id == "wp_1__s1"]
    assert second_section_chunks[0].char_start >= doc.sections[1].char_start


def test_long_section_is_split_and_ids_are_numbered_per_section(tokenizer: tiktoken.Encoding) -> None:
    doc = _make_document()
    chunks = chunk_document(doc, SECTION_STRATEGY, CONFIG, make_splitter(SECTION_STRATEGY, tokenizer))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "wp_1__s0__c0" and ids[1] == "wp_1__s0__c1"
    assert ids[-1] == "wp_1__s1__c0"


def test_article_unit_headers_have_no_section_and_start_section_is_recorded(
    tokenizer: tiktoken.Encoding,
) -> None:
    doc = _make_document()
    chunks = chunk_document(doc, ARTICLE_STRATEGY, CONFIG, make_splitter(ARTICLE_STRATEGY, tokenizer))
    assert all(list(c.header) == ["Song", "Album"] for c in chunks)
    assert chunks[0].chunk_id == "wp_1__c0"
    assert chunks[0].section_id == "wp_1__s0"
    assert chunks[-1].section_id == "wp_1__s1"


def test_check_chunks_rejects_wrong_offsets(tokenizer: tiktoken.Encoding) -> None:
    doc = _make_document()
    chunks = chunk_document(doc, SECTION_STRATEGY, CONFIG, make_splitter(SECTION_STRATEGY, tokenizer))
    fields = ["Song", "Album", "Section"]
    check_chunks(doc, chunks, fields)
    shifted = chunks[0].model_copy(update={"char_start": chunks[0].char_start + 1})
    with pytest.raises(AssertionError):
        check_chunks(doc, [shifted], fields)


def test_check_chunks_rejects_an_empty_header_field(tokenizer: tiktoken.Encoding) -> None:
    doc = _make_document()
    chunks = chunk_document(doc, SECTION_STRATEGY, CONFIG, make_splitter(SECTION_STRATEGY, tokenizer))
    blank = chunks[0].model_copy(update={"header": {"Song": "Test Song", "Album": " ", "Section": "Part 0"}})
    with pytest.raises(AssertionError):
        check_chunks(doc, [blank], ["Song", "Album", "Section"])
