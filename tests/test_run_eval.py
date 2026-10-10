"""Tests for the pure parts of src/eval/run_eval.py: scoring one question from
what search() returned, the summaries and the experiments file. The search
itself runs against a built collection and is not repeated here."""

from datetime import datetime
from pathlib import Path

import pytest

from src.config import Config
from src.eval.run_eval import (
    QuestionResult, _decimal, append_experiment, experiment_row, format_report, group_by, make_run_id, score_question,
    summarize,
)
from src.ingest.normalize import load_tokenizer
from src.schemas import Chunk, EvalQuestion, Section

TOKENIZER = load_tokenizer(Config.load(Path(__file__).resolve().parent.parent / "config.yaml"))
TEXT = "They recorded it in one take. Lennon sang it twice."


def _section() -> Section:
    return Section(section_id="wp_1__s2", title="Recording", path="Recording", topics=["recording"],
                   text=TEXT, char_start=100, char_end=100 + len(TEXT), token_count=12)


def _question() -> EvalQuestion:
    return EvalQuestion(question_id="wp_1__s2__q0", question="How was it recorded?", answer="In one take.",
                        doc_id="wp_1", doc_type="song", section_id="wp_1__s2", section_heading="Recording",
                        evidence="in one take")


def _chunk(doc_id: str, start: int, end: int) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}__c{start}", doc_id=doc_id, section_id="x", doc_type="song", topics=["recording"],
                 title="T", section_path="P", url="u", revid=1, header={"Song": "T"}, text="some text",
                 char_start=start, char_end=end)


def test_a_question_gets_the_rank_of_its_first_hit_and_of_its_article() -> None:
    chunks = [_chunk("wp_9", 0, 50), _chunk("wp_1", 0, 40), _chunk("wp_1", 100, 160)]
    result = score_question(_question(), _section(), chunks, final_top_k=2, tokenizer=TOKENIZER)
    assert (result.hit_rank, result.doc_rank) == (3, 2)
    assert result.topics == ["recording"]
    assert result.context_tokens == 2 * len(TOKENIZER.encode("Song: T\n\nsome text"))  # only the first 2


def _result(hit: int | None, doc: int | None, doc_type: str = "song", topics: list[str] | None = None) -> QuestionResult:
    return QuestionResult(question_id="q", doc_type=doc_type, topics=topics or ["recording"],
                          hit_rank=hit, doc_rank=doc, context_tokens=100)


def test_summaries_turn_ranks_into_hit_at_k_mrr_and_doc_hit() -> None:
    results = [_result(1, 1), _result(4, 1), _result(None, 6), _result(None, None)]
    summary = summarize(results, k_values=[1, 5])
    assert summary["hit@1"] == 0.25 and summary["hit@5"] == 0.5
    assert summary["doc_hit@1"] == 0.5 and summary["doc_hit@5"] == 0.5   # ranks 1, 1, 6, none
    assert summary["mrr@5"] == pytest.approx((1 + 0.25) / 4)
    assert summary["mrr@1"] == pytest.approx(1 / 4)                       # rank 4 is beyond k = 1
    assert summary["n"] == 4


def test_a_question_with_two_topics_counts_in_both_groups() -> None:
    results = [_result(1, 1, topics=["recording", "writing"]), _result(1, 1, topics=["writing"])]
    groups = group_by(results, "topic")
    assert list(groups) == ["writing", "recording"]           # largest first
    assert len(groups["writing"]) == 2 and len(groups["recording"]) == 1
    assert list(group_by(results, "doc_type")) == ["song"]


def test_the_report_names_the_hit_at_k_that_decides() -> None:
    report = format_report([_result(1, 1), _result(None, None, doc_type="album")], [1, 5], final_top_k=5)
    assert "All questions (2)" in report and "← decides" in report
    assert "By doc_type" in report and "By topic" in report


def test_experiments_keep_one_header_and_refuse_rows_that_do_not_fit(tmp_path: Path) -> None:
    path = tmp_path / "experiments.csv"
    append_experiment(path, {"run_id": "a", "hit@5": 0.7})
    append_experiment(path, {"run_id": "b", "hit@5": 0.8})
    assert path.read_text(encoding="utf-8").splitlines() == ["run_id,hit@5", "a,0.7", "b,0.8"]
    with pytest.raises(ValueError, match="Start a new file"):
        append_experiment(path, {"run_id": "c", "hit@7": 0.9})


def test_numbers_show_up_to_four_decimals_without_trailing_zeros() -> None:
    assert [_decimal(v, 0) for v in (1.0, 0.3, 0.92646, 0.0)] == ["1.0", "0.3", "0.9265", "0.0"]


def test_the_run_id_and_the_csv_row_name_every_setting_that_tells_runs_apart() -> None:
    config = Config.load(Path(__file__).resolve().parent.parent / "config.yaml").with_overrides(alpha=1, reranker="minilm")
    run_id = make_run_id(datetime(2026, 9, 29, 10, 54), "section_t512_o0", config)
    assert run_id == "2026-09-29_1054_section_t512_o0_nomic_a1_minilm_k8_cap2"
    row = experiment_row(run_id, "reranker", "section_t512_o0", "dev", config, {"n": 3, "hit@5": 0.5}, search_ms=41.6)
    assert (row["alpha"], row["reranker"], row["search_ms"]) == (1, "minilm", 42)
    assert make_run_id(datetime(2026, 9, 29, 10, 54), "x", config.with_overrides(reranker="none", final_top_k=7, max_per_section="none")).endswith("_none_k7_capnone")
    assert make_run_id(datetime(2026, 9, 29, 10, 54), "x", config, split="test").endswith("_k8_cap2_test")
