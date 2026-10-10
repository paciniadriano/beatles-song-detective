"""Pydantic models shared by every layer. A Document is one normalized
Wikipedia article with no chunking decisions baked in: that is what lets any
chunking strategy in index/chunking.py be rebuilt without re-downloading
anything."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, model_validator

DocType = Literal["song", "album", "person", "studio", "technique", "equipment", "overview"]


PATH_SEPARATOR = " > "  # between a heading and its ancestors in Section.path
INTRODUCTION_TITLE = "Introduction"  # the text before the first heading


class Section(BaseModel):
    """One heading's text. `path` is the heading with its ancestors once it is
    nested ("Recording > Take 7"); `topics` are what it is about, for filtering
    and for the evaluation pool (config.yaml, `topics`); char_start/char_end
    slice `text` back out of the Document's full_text."""
    section_id: str
    title: str
    path: str
    topics: list[str]
    text: str
    char_start: int
    char_end: int
    token_count: int


class DocumentStats(BaseModel):
    section_count: int
    token_count: int
    char_count: int


class Document(BaseModel):
    """`revid` and `revised_at` identify the frozen revision every citation
    points to. `entity` is the name of the thing the article is about, and
    `album` the one a song's header shows (seeds.yaml): set for every song,
    None for every other doc_type."""
    doc_id: str
    pageid: int
    revid: int
    revised_at: datetime
    title: str
    url: str
    license: str
    doc_type: DocType
    entity: str
    album: str | None
    sections: list[Section]
    full_text: str
    stats: DocumentStats

    @model_validator(mode="after")
    def _only_songs_have_an_album(self) -> "Document":
        """A song without an album would give an empty header field; any
        other article with one would be caught by an album filter."""
        if (self.doc_type == "song") != (self.album is not None):
            raise ValueError(f"{self.doc_id}: a song needs an album, and only a song has one")
        return self


class Chunk(BaseModel):
    """One retrievable piece of a Document, for one chunking strategy.

    Everything the agent needs to cite it (title, url, revid, section) travels
    with the chunk, so nothing above search.py has to reload a Document.
    `header` holds the fields that get prepended to `text` when it is embedded;
    they are stored apart from `text` so that char_start/char_end always point
    at the article's own words. `score` is None until search() fills it."""
    chunk_id: str
    doc_id: str
    section_id: str
    doc_type: DocType
    topics: list[str]
    title: str
    section_path: str
    url: str
    revid: int
    header: dict[str, str]
    text: str
    char_start: int
    char_end: int
    score: float | None = None


class EvalQuestion(BaseModel):
    """One evaluation question (retrieval pool or generation set), anchored
    to content rather than to chunks.

    `doc_id`, `section_id` and the verbatim `evidence` quote are what let one
    frozen set score every chunking strategy: chunk ids change with each
    strategy, this anchor does not. `section_heading` is for human reading
    only. The labels below `evidence` are filled by later passes.

    Two kinds of question share this model:
    - answerable (the default): generated from one section, so the anchor
      fields are required.
    - not answerable: written for the generation set (with the help of AI,
      then curated by hand), such as counts and superlatives ("which
      technique did Emerick use most?"). No single passage answers them, so
      they have no anchor, and the right behaviour is to decline with "the
      sources here don't say". They measure whether the system declines
      instead of making something up."""
    question_id: str
    question: str
    answer: str
    answerable: bool = True
    doc_id: str | None = None
    doc_type: DocType
    section_id: str | None = None
    section_heading: str | None = None
    evidence: str | None = None
    split: Literal["dev", "test"] | None = None
    # Generation set only. `kind` groups the report (detail, credits, known,
    # unanswerable, out_of_scope, follow_up, adversarial). `previous_question`
    # makes it a second turn: the first is asked only to build the history.
    kind: str | None = None
    previous_question: str | None = None

    @model_validator(mode="after")
    def _anchor_matches_answerable(self) -> "EvalQuestion":
        """An answerable question without an anchor could never score a hit,
        and an unanswerable one with an evidence quote contradicts itself.
        Both are mistakes to catch when the file is loaded, not at scoring."""
        anchor = [self.doc_id, self.section_id, self.section_heading, self.evidence]
        if self.answerable and any(field is None for field in anchor):
            raise ValueError(
                f"{self.question_id}: an answerable question needs doc_id, "
                "section_id, section_heading and evidence"
            )
        if not self.answerable and self.evidence is not None:
            raise ValueError(
                f"{self.question_id}: a question that is not answerable has no evidence quote"
            )
        return self
