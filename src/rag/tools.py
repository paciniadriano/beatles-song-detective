"""The one tool the agent can call, `search`, and the numbered sources it
produces.

Three jobs:
- TOOLS: the definition the model sees. It is sent on every call and never
  changes, so it is part of the prompt prefix OpenAI caches.
- run_search(): runs one search and writes its result for the model, each
  chunk numbered so the answer can cite it as [1], [2]...
- Sources: which chunk each number stands for, for the whole question. The
  app turns [n] into a link (citation_url); the evaluation checks that every
  cited number exists and that the evidence chunk was among the sources.

Numbers belong to one question: the next question starts again at [1]. That
is why strip_citations() exists: an earlier answer goes into the history
without its [n], which would otherwise point at sources the model no longer
sees."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import quote

from src.schemas import INTRODUCTION_TITLE, PATH_SEPARATOR, Chunk

SEARCH_TOOL_NAME = "search"

SEARCH_TOOL: dict[str, object] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the English Wikipedia passages about the Beatles' songs, albums, "
            "people, studios, equipment and techniques. Returns numbered passages "
            "to cite as [1], [2]..."
        ),
        "strict": True,  # the API guarantees the arguments match the schema
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A search query in English, whatever the user's language. It must "
                        "make sense on its own: resolve 'it', 'that song' or 'he' from the "
                        "conversation, and name the song, album or person it is about."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

# Every tool the model may call: only search. The agent answers any other
# name with an error message instead of failing.
TOOLS = [SEARCH_TOOL]

NO_RESULTS = "No passages found."
EMPTY_QUERY = "The search query was empty. Search again with a query in English."

_CITATION = re.compile(r"\s*\[\d+\]")


class ChunkSearcher(Protocol):
    """What run_search() needs from search.py: only the query, since the
    tool has no filters and top_k comes from config."""

    def search(self, query: str) -> list[Chunk]: ...


@dataclass
class Sources:
    """The chunks shown to the model while answering one question, in the
    order they were first shown. Chunk number n is `chunks[n - 1]`, so a chunk
    that comes back in a later search keeps the number it already has."""
    chunks: list[Chunk] = field(default_factory=list)

    def number_of(self, chunk: Chunk) -> int | None:
        for number, known in enumerate(self.chunks, start=1):
            if known.chunk_id == chunk.chunk_id:
                return number
        return None

    def add(self, chunk: Chunk) -> int:
        """Gives a new chunk the next number; returns it."""
        self.chunks.append(chunk)
        return len(self.chunks)

    def get(self, number: int) -> Chunk | None:
        """The chunk behind [number], or None if the model cited a number it
        was never shown."""
        if 1 <= number <= len(self.chunks):
            return self.chunks[number - 1]
        return None


def format_source(number: int, chunk: Chunk) -> str:
    """One passage as the model reads it: its number, the same header the
    embedder saw (Song, Album, Section...), then the text. No URL, revid or
    score: they cost tokens and do not help answer."""
    header = " | ".join(f"{key}: {value}" for key, value in chunk.header.items())
    return f"[{number}] {header}\n{chunk.text}"


def run_search(query: str, searcher: ChunkSearcher, sources: Sources) -> str:
    """Runs one search and returns the text of the tool message. New chunks
    get the next numbers; a chunk already shown in this question is not sent
    again, only named, so a second search does not repeat context."""
    if not query.strip():
        return EMPTY_QUERY
    chunks = searcher.search(query)
    if not chunks:
        return NO_RESULTS
    blocks = []
    for chunk in chunks:
        known = sources.number_of(chunk)
        if known is not None:
            blocks.append(f"[{known}] (already shown above)")
        else:
            blocks.append(format_source(sources.add(chunk), chunk))
    return "\n\n".join(blocks)


def cited_numbers(answer: str) -> list[int]:
    """The distinct [n] in an answer, in order of first appearance."""
    numbers = [int(match) for match in re.findall(r"\[(\d+)\]", answer)]
    return list(dict.fromkeys(numbers))


def strip_citations(answer: str) -> str:
    """The answer without its [n], for the conversation history."""
    return _CITATION.sub("", answer)


def citation_url(chunk: Chunk) -> str:
    """A link to the exact revision the chunk was taken from, at its section,
    so the page the user opens says what the answer cited even if the article
    has changed since. The introduction has no anchor: it is the top."""
    url = f"https://en.wikipedia.org/w/index.php?oldid={chunk.revid}"
    heading = chunk.section_path.split(PATH_SEPARATOR)[-1]
    if heading == INTRODUCTION_TITLE:
        return url
    return url + "#" + quote(heading.replace(" ", "_"), safe="():,'!&-.")
