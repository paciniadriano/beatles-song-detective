"""Tests for tools.py: how search results are numbered and shown to the model,
and how citations are read back. The searcher is a fake that returns prepared
chunks, so no Qdrant collection is needed."""

from src.rag.tools import (
    EMPTY_QUERY,
    NO_RESULTS,
    SEARCH_TOOL,
    Sources,
    citation_url,
    cited_numbers,
    format_source,
    run_search,
    strip_citations,
)
from src.schemas import Chunk


def _chunk(chunk_id: str, text: str = "text", section_path: str = "Recording") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="wp_1", section_id="wp_1__s1", doc_type="song", topics=["recording"],
        title="Hey Jude", section_path=section_path, url="https://en.wikipedia.org/wiki/Hey_Jude",
        revid=123, header={"Song": "Hey Jude", "Section": section_path}, text=text,
        char_start=0, char_end=len(text),
    )


class _FakeSearcher:
    def __init__(self, *results: list[Chunk]) -> None:
        self._results = list(results)
        self.queries: list[str] = []

    def search(self, query: str) -> list[Chunk]:
        self.queries.append(query)
        return self._results.pop(0)


def test_a_source_shows_its_number_header_and_text() -> None:
    chunk = _chunk("c1", text="It was the first eight-track recording.")
    assert format_source(3, chunk) == (
        "[3] Song: Hey Jude | Section: Recording\nIt was the first eight-track recording."
    )


def test_the_first_search_numbers_its_chunks_from_one() -> None:
    sources = Sources()
    result = run_search("Hey Jude", _FakeSearcher([_chunk("a"), _chunk("b")]), sources)
    assert result.startswith("[1] ")
    assert "\n\n[2] " in result
    assert [c.chunk_id for c in sources.chunks] == ["a", "b"]


def test_a_second_search_continues_the_numbers_and_does_not_resend_a_known_chunk() -> None:
    sources = Sources()
    searcher = _FakeSearcher([_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")])
    run_search("first", searcher, sources)
    second = run_search("second", searcher, sources)
    assert second.startswith("[2] (already shown above)")
    assert "[3] Song: Hey Jude" in second
    assert [c.chunk_id for c in sources.chunks] == ["a", "b", "c"]


def test_no_results_and_an_empty_query_answer_the_model_instead_of_failing() -> None:
    sources = Sources()
    assert run_search("anything", _FakeSearcher([]), sources) == NO_RESULTS
    searcher = _FakeSearcher()
    assert run_search("   ", searcher, sources) == EMPTY_QUERY
    assert searcher.queries == []  # an empty query never reaches search()
    assert sources.chunks == []


def test_a_number_the_model_was_never_shown_has_no_chunk() -> None:
    sources = Sources([_chunk("a")])
    assert sources.get(1).chunk_id == "a"
    assert sources.get(0) is None
    assert sources.get(2) is None


def test_citations_are_read_in_order_once_each() -> None:
    assert cited_numbers("Emerick [2] did it [1], at Abbey Road [2][3].") == [2, 1, 3]
    assert cited_numbers("The sources here don't say.") == []


def test_citations_are_stripped_for_the_history() -> None:
    assert strip_citations("Emerick did it [1], at Abbey Road [2][3].") == "Emerick did it, at Abbey Road."


def test_the_link_points_at_the_exact_revision_and_section() -> None:
    assert citation_url(_chunk("a", section_path="Recording > Brass overdub")) == (
        "https://en.wikipedia.org/w/index.php?oldid=123#Brass_overdub"
    )
    assert citation_url(_chunk("a", section_path="Introduction")) == (
        "https://en.wikipedia.org/w/index.php?oldid=123"
    )


def test_the_tool_schema_is_strict_with_one_required_query() -> None:
    function = SEARCH_TOOL["function"]
    assert function["strict"] is True
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["additionalProperties"] is False
