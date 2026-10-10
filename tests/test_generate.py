"""Tests for the pure parts of src/eval/generate.py. The questions themselves
are judged by a person; what is checked here is that every eligible section
gets a question and that a quote which is not in its section can never get
through."""

import pytest

from src.config import EvalGenerateConfig, TokenPrices
from src.eval.generate import (
    GeneratedQuestion, all_candidates, build_prompt,
    cost_report, eligible_sections, find_evidence,
    sections_still_to_do, validate_candidate,
)
from src.rag.llm import TokenUsage
from src.schemas import Document, EvalQuestion, Section


def _section(index: int, topic: str, tokens: int = 500, text: str = "Alpha beta gamma.") -> Section:
    return Section(
        section_id=f"wp_1__s{index}", title=f"Heading {index}", path=f"Heading {index}",
        topics=topic.split("+"), text=text, char_start=0, char_end=len(text), token_count=tokens,
    )


def _doc(doc_id: str, doc_type: str, sections: list[Section]) -> Document:
    return Document.model_construct(doc_id=doc_id, doc_type=doc_type, title=doc_id, sections=sections)


OUT_OF_SCOPE = ["release", "reception", "cover_versions"]


def _settings() -> EvalGenerateConfig:
    return EvalGenerateConfig(
        model="m", max_output_tokens=700,
        output="out.jsonl", pilot_output="pilot.jsonl",
        mention_required_for_doc_types=["studio"], mention_terms=["Beatle", "Lennon"],
    )


def test_a_section_needs_one_in_scope_topic_whatever_its_length() -> None:
    doc = _doc("wp_1", "song", [
        _section(0, "recording", 12), _section(1, "release"), _section(2, "general", 3000),
        _section(3, "reception"), _section(4, "cover_versions"), _section(5, "recording+release"),
    ])
    assert [s.section_id for s in eligible_sections(doc, _settings(), OUT_OF_SCOPE)] == ["wp_1__s0", "wp_1__s2", "wp_1__s5"]


def test_every_eligible_section_of_every_document_gets_one_question() -> None:
    song = _doc("wp_1", "song", [_section(0, "recording"), _section(1, "recording"), _section(2, "release")])
    studio = _doc("wp_2", "studio", [_section(0, "general", text="The Beatles recorded here.")])
    chosen = all_candidates([song, studio], _settings(), OUT_OF_SCOPE)
    assert [(d.doc_id, s.section_id) for d, s in chosen] == [
        ("wp_1", "wp_1__s0"), ("wp_1", "wp_1__s1"), ("wp_2", "wp_1__s0"),
    ]


def test_articles_not_about_the_band_keep_only_sections_that_name_it() -> None:
    studio = _doc("wp_1", "studio", [
        _section(0, "general", text="Lennon recorded here in 1968."),
        _section(1, "writing", text="The building was redeveloped in 2015."),
        _section(2, "recording", text="A former Beatle came back in 1972."),
    ])
    assert [s.section_id for s in eligible_sections(studio, _settings(), OUT_OF_SCOPE)] == ["wp_1__s0", "wp_1__s2"]
    song = _doc("wp_2", "song", [_section(0, "writing", text="The melody descends by step.")])
    assert len(eligible_sections(song, _settings(), OUT_OF_SCOPE)) == 1   # a song needs no mention


def test_a_quote_not_in_the_section_is_rejected() -> None:
    section = _section(0, "recording", 500, text="They recorded it in one take.")
    good = GeneratedQuestion(question="q?", answer="a", evidence="in one take")
    bad = good.model_copy(update={"evidence": "in two takes"})
    assert validate_candidate(good, section).evidence == "in one take"
    with pytest.raises(ValueError, match="not in the section"):
        validate_candidate(bad, section)


def test_a_quote_with_look_alike_characters_is_stored_as_the_section_wrote_it() -> None:
    """Credit lists use en dashes and one line per person; the model tends to
    copy them with a hyphen and a space. The stored evidence must still be
    the section's own text, character for character."""
    text = "George Harrison – lead guitar (2nd solo)\nEric Clapton – lead guitar (1st solo)"
    section = _section(0, "personnel", 500, text=text)
    copied = GeneratedQuestion(
        question="q?", answer="a",
        evidence="(2nd solo) Eric Clapton - lead guitar",
    )
    anchored = validate_candidate(copied, section)
    assert anchored.evidence == "(2nd solo)\nEric Clapton – lead guitar"
    assert anchored.evidence in section.text


def test_a_full_stop_the_model_added_to_a_credit_line_is_ignored() -> None:
    text = "George Martin – piano\nRingo Starr – drums"
    assert find_evidence("George Martin – piano.", text) == "George Martin – piano"
    assert find_evidence("George Martin - piano.", text) == "George Martin – piano"
    assert find_evidence("George Martin – organ.", text) is None


def test_find_evidence_refuses_an_empty_quote() -> None:
    assert find_evidence("   ", "Some section text.") is None


def test_resume_skips_sections_that_already_have_a_question() -> None:
    docs = [_doc("wp_1", "song", [_section(0, "recording", 500), _section(1, "recording", 500)])]
    candidates = all_candidates(docs, _settings(), OUT_OF_SCOPE)
    done = [EvalQuestion(
        question_id="wp_1__s0__q0", question="q?", answer="a", doc_id="wp_1", doc_type="song",
        section_id="wp_1__s0", section_heading="h", evidence="e",
    )]
    assert [s.section_id for _, s in sections_still_to_do(candidates, done)] == ["wp_1__s1"]


def test_cost_report_projects_the_pilot_average_onto_the_whole_pool() -> None:
    usage = TokenUsage(input_tokens=30_000, output_tokens=6_000)  # 30 sections
    report = cost_report(usage, sections_sent=30, pool_size=1500, prices=TokenPrices(input=0.10, cached_input=0.01, output=0.50))
    assert "$0.0060" in report          # 0.003 input + 0.003 output
    assert "~$0.30" in report           # 0.0002 per section x 1500


def test_the_prompt_carries_the_article_the_section_path_and_its_text() -> None:
    doc = _doc("wp_1", "song", [])
    section = _section(0, "recording", 500, text="They recorded it in one take.")
    prompt = build_prompt(doc, section)
    for part in ("Article: wp_1 (song)", f"Section: {section.path}", section.text):
        assert part in prompt


def test_the_quote_is_the_first_field_the_model_writes() -> None:
    """Structured Outputs are written in schema order: with the quote first,
    the question and answer are written from it, not from the whole section."""
    assert list(GeneratedQuestion.model_fields) == ["evidence", "question", "answer"]
