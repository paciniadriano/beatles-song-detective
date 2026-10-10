"""Tests for the generation set's scripts: the automatic checks, what a run
writes, and the report. No API, no Qdrant: the agent is replaced by prepared
Done events, and files live in tmp_path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Config
from src.eval.generation_checks import Checks, Grade, check_answer, is_decline, passes
from src.eval.generation_report import compare, refuse_unfinished, rescuable, max_steps_pre_check, summary
from src.eval.metrics import fewest_gains_to_win
from src.eval.question_file import load_question_set
from src.eval.run_generation import (
    answer_entry, ask, make_run_id, read_answers, reuse_grades, write_answers,
)
from src.rag.agent import Done
from src.rag.llm import TokenUsage
from src.rag.tools import Sources
from src.schemas import Chunk, EvalQuestion, Section

ROOT = Path(__file__).resolve().parent.parent
TEXT = "They recorded it in one take. Lennon sang it twice."


def _section() -> Section:
    return Section(section_id="wp_1__s2", title="Recording", path="Recording", topics=["recording"],
                   text=TEXT, char_start=100, char_end=100 + len(TEXT), token_count=12)


def _question(**changes: object) -> EvalQuestion:
    fields = dict(question_id="gen_001", question="How was it recorded?", answer="In one take.", kind="detail",
                  doc_id="wp_1", doc_type="song", section_id="wp_1__s2", section_heading="Recording",
                  evidence="in one take")
    return EvalQuestion(**{**fields, **changes})


def _declining_question() -> EvalQuestion:
    return EvalQuestion(question_id="gen_050", question="Which song took the most takes?", kind="unanswerable",
                        answer="The sources here don't say.", answerable=False, doc_type="song")


def _chunk(chunk_id: str, start: int, end: int, doc_id: str = "wp_1") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, section_id="wp_1__s2", doc_type="song", topics=["recording"],
                 title="Hey Jude", section_path="Recording", url="u", revid=1, header={"Song": "Hey Jude"},
                 text="some text", char_start=start, char_end=end)


EVIDENCE_CHUNK = _chunk("hit", 100, 160)
OTHER_CHUNK = _chunk("miss", 0, 50, doc_id="wp_9")


# --- checks ------------------------------------------------------------------

def test_the_decline_phrase_is_recognised_whatever_its_apostrophe_or_case() -> None:
    assert is_decline("The sources here don't say.")
    assert is_decline("  the sources here don’t say ")
    assert not is_decline("The sources here don't say who played it, but [1] says...")


def test_checks_see_the_evidence_among_the_sources_and_among_the_citations() -> None:
    checks = check_answer(_question(), "In one take [2].", [OTHER_CHUNK, EVIDENCE_CHUNK], False, _section())
    assert checks.cites and checks.citations_exist and checks.no_link
    assert checks.evidence_retrieved is True
    assert checks.evidence_cited is True


def test_evidence_retrieved_but_not_cited() -> None:
    checks = check_answer(_question(), "In one take [1].", [OTHER_CHUNK, EVIDENCE_CHUNK], False, _section())
    assert checks.evidence_retrieved is True
    assert checks.evidence_cited is False


def test_a_citation_the_model_was_never_shown_and_a_link_are_caught() -> None:
    checks = check_answer(_question(), "See www.example.com [3].", [EVIDENCE_CHUNK], False, _section())
    assert not checks.citations_exist
    assert not checks.no_link


def test_a_question_that_must_decline_has_no_evidence_checks() -> None:
    checks = check_answer(_declining_question(), "The sources here don't say.", [OTHER_CHUNK], False, None)
    assert checks.declined
    assert checks.evidence_retrieved is None


def test_pass_needs_both_grades_valid_citations_and_no_link() -> None:
    good = Checks(declined=False, cites=True, citations_exist=True, no_link=True, truncated=False)
    assert passes(True, good, Grade(correct=True, faithful=True))
    assert not passes(True, good, Grade(correct=True, faithful=False))
    assert not passes(True, good.model_copy(update={"citations_exist": False}), Grade(correct=True, faithful=True))
    with pytest.raises(ValueError):
        passes(True, good, Grade(correct=True))


def test_a_question_that_must_decline_passes_on_declining_alone() -> None:
    declined = Checks(declined=True, cites=False, citations_exist=True, no_link=True, truncated=False)
    assert passes(False, declined, Grade())
    assert not passes(False, declined.model_copy(update={"declined": False}), Grade())


# --- the max_steps pre-check ------------------------------------------------

def test_fewest_gains_to_win_follows_the_tie_rule() -> None:
    assert fewest_gains_to_win(100, 0.02) == 4      # 4/100: +4.0 against a 3.9 margin
    assert fewest_gains_to_win(1240, 0.02) == 25    # the reranker's recoverable test


# --- what a run writes -------------------------------------------------------

def _done(answer: str, chunks: list[Chunk]) -> Done:
    return Done(answer, Sources(list(chunks)), ["Hey Jude recording"], TokenUsage(100, 10, 50), truncated=False)


def test_an_answerable_entry_carries_the_sources_checks_and_empty_grades() -> None:
    done = _done("In one take [1].", [EVIDENCE_CHUNK])
    checks = check_answer(_question(), done.answer, done.sources.chunks, False, _section())
    entry = answer_entry(_question(), {"wp_1": "Hey Jude"}, done, checks, done.usage, 1.234)
    assert entry["sources"] == ["[1] Hey Jude > Recording"]
    assert entry["cited_passages"] == {"[1] Hey Jude > Recording": "some text"}
    assert entry["checks"]["evidence_cited"] is True
    assert entry["tokens"] == {"input": 100, "cached": 50, "output": 10}
    assert (entry["correct"], entry["faithful"], entry["note"]) == (None, None, "")
    assert entry["seconds"] == 1.2


def test_only_cited_passages_are_kept_and_a_made_up_number_is_skipped() -> None:
    from src.eval.run_generation import cited_passages
    chunks = [OTHER_CHUNK, EVIDENCE_CHUNK]
    assert list(cited_passages("A [2], B [2], C [7].", chunks)) == ["[2] Hey Jude > Recording"]


def test_an_entry_that_must_decline_has_nothing_to_grade() -> None:
    done = _done("The sources here don't say.", [OTHER_CHUNK])
    checks = check_answer(_declining_question(), done.answer, done.sources.chunks, False, None)
    entry = answer_entry(_declining_question(), {}, done, checks, done.usage, 1.0)
    assert "correct" not in entry and "evidence" not in entry


def test_grades_are_reused_only_for_identical_answers() -> None:
    earlier = [{"question_id": "a", "answer": "X [1].", "correct": True, "faithful": True, "note": "ok"},
               {"question_id": "b", "answer": "Y [1].", "correct": False, "faithful": True, "note": ""}]
    entries = [{"question_id": "a", "answer": "X [1].", "correct": None, "faithful": None, "note": ""},
               {"question_id": "b", "answer": "Y, changed [1].", "correct": None, "faithful": None, "note": ""}]
    assert reuse_grades(entries, earlier) == 1
    assert entries[0]["correct"] is True and entries[0]["note"] == "ok"
    assert entries[1]["correct"] is None


def test_the_answers_file_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "run" / "answers.yaml"
    write_answers(path, [{"question_id": "a", "answer": "Ñandú – “quoted”"}], "# header\n")
    assert read_answers(path) == [{"question_id": "a", "answer": "Ñandú – “quoted”"}]
    with pytest.raises(FileExistsError):
        write_answers(path, [], "# header\n")


def test_the_run_id_names_the_settings_or_closed_book() -> None:
    from datetime import datetime
    config = Config.load(ROOT / "config.yaml").with_overrides(max_steps=2)
    now = datetime(2026, 10, 1, 10, 30)
    assert make_run_id(now, "section_t256_o0", config, closed_book=False) == "2026-10-01_1030_section_t256_o0_k8_steps2"
    assert make_run_id(now, "section_t256_o0", config, closed_book=True) == "2026-10-01_1030_closed_book"


def test_a_follow_up_asks_its_previous_question_first_and_counts_both_turns() -> None:
    """Driven by a scripted model: the first turn and the second each search
    once and answer, so four calls in all."""
    config = Config.load(ROOT / "config.yaml")

    def searches(query: str) -> list[SimpleNamespace]:
        function = SimpleNamespace(name="search", arguments=json.dumps({"query": query}))
        delta = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(index=0, id="c", function=function)])
        return [SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="tool_calls")], usage=None)]

    def answers(text: str) -> list[SimpleNamespace]:
        delta = SimpleNamespace(content=text, tool_calls=None)
        return [SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="stop")], usage=None)]

    replies = [searches("Mr Kite lyrics source"), answers("A circus poster [1]."),
               searches("Mr Kite poster bought"), answers("In Sevenoaks [1].")]
    requests: list[dict] = []

    def create(**request: object) -> list[SimpleNamespace]:
        requests.append(request)
        return replies.pop(0)

    model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    searcher = SimpleNamespace(search=lambda query: [EVIDENCE_CHUNK])
    question = _question(previous_question="Where did the lyrics come from?", question="Where did John get it?")
    done, _ = ask(question, searcher, model, config, "SYSTEM")
    assert done.answer == "In Sevenoaks [1]."
    assert len(requests) == 4
    history = [(m["role"], m["content"]) for m in requests[2]["messages"][:4]]
    assert history == [("system", "SYSTEM"), ("user", "Where did the lyrics come from?"),
                       ("assistant", "A circus poster."), ("user", "Where did John get it?")]


# --- the report --------------------------------------------------------------

def _entry(question_id: str, correct: bool | None, kind: str = "detail") -> dict:
    checks = Checks(declined=False, cites=True, citations_exist=True, no_link=True, truncated=False,
                    evidence_retrieved=True, evidence_cited=True).model_dump()
    return {"question_id": question_id, "kind": kind, "answer": "x [1].", "checks": checks,
            "correct": correct, "faithful": True, "note": ""}


def _declined_entry(question_id: str, declined: bool) -> dict:
    checks = Checks(declined=declined, cites=False, citations_exist=True, no_link=True, truncated=False).model_dump()
    return {"question_id": question_id, "kind": "unanswerable", "answer": "x", "checks": checks}


def test_the_report_refuses_a_run_that_is_not_fully_graded() -> None:
    with pytest.raises(ValueError, match="q2"):
        refuse_unfinished([_entry("q1", True), _entry("q2", None)], ("correct", "faithful"))


def test_the_summary_counts_pass_overall_and_per_kind() -> None:
    entries = [_entry("q1", True), _entry("q2", False), _declined_entry("q3", True)]
    text = summary(entries)
    assert "pass: 2/3" in text
    assert "detail: 1/2" in text and "unanswerable: 1/1" in text


def test_the_pre_check_counts_only_failed_answers_missing_their_evidence() -> None:
    failed_missing = _entry("q1", False)
    failed_missing["checks"]["evidence_retrieved"] = False
    right_from_another_article = _entry("q2", True)       # same fact, other article: passed
    right_from_another_article["checks"]["evidence_retrieved"] = False
    failed_with_evidence = _entry("q3", False)             # a second search would not help
    entries = [failed_missing, right_from_another_article, failed_with_evidence, _declined_entry("q4", False)]
    assert rescuable(entries) == 1


def test_the_pre_check_decides_with_the_fewest_gains_that_could_win() -> None:
    def run_with(rescuable_count: int) -> list[dict]:
        entries = [_entry(f"q{i}", True) for i in range(100)]
        for entry in entries[:rescuable_count]:
            entry["correct"] = False
            entry["checks"]["evidence_retrieved"] = False
        return entries
    assert max_steps_pre_check(run_with(3), 0.02).endswith("1 step stays; no 2-step run")
    assert max_steps_pre_check(run_with(4), 0.02).endswith("run the 2-step arm")


def test_two_runs_are_compared_by_the_tie_rule() -> None:
    a = [_entry(f"q{i}", i >= 10) for i in range(100)]      # A passes 90
    b = [_entry(f"q{i}", True) for i in range(100)]         # B passes all 100
    assert compare(a, b, 0.02).endswith("B wins")
    assert compare(a, a, 0.02).endswith("tie")
    with pytest.raises(ValueError):
        compare(a, b[:50], 0.02)


# --- the question set --------------------------------------------------------

def test_the_frozen_set_loads_with_its_kinds_and_follow_ups() -> None:
    config = Config.load(ROOT / "config.yaml")
    questions = load_question_set(ROOT / config.eval.generation.questions)
    assert len(questions) == 104
    kinds = {question.kind for question in questions}
    assert {"detail", "unanswerable", "follow_up", "adversarial"} <= kinds
    assert all(question.previous_question for question in questions if question.kind == "follow_up")
