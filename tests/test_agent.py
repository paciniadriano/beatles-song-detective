"""Tests for agent.py. The model is a fake that plays a script of streamed
replies (search, or answer), and the searcher a fake that returns prepared
chunks, so no API key or Qdrant collection is needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Config, LLMConfig
from src.ingest.normalize import load_tokenizer
from src.rag.agent import Done, Searching, Turn, answer, build_messages, load_prompt, tool_choice_for
from src.rag.llm import TextDelta, TokenUsage
from src.rag.tools import TOOLS
from src.schemas import Chunk

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Config.load(ROOT / "config.yaml")


def _settings(max_steps: int = 1, history_turns: int = 3) -> LLMConfig:
    return CONFIG.llm.model_copy(update={"max_steps": max_steps, "history_turns": history_turns})


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="wp_1", section_id="wp_1__s1", doc_type="song", topics=["recording"],
        title="Hey Jude", section_path="Recording", url="https://en.wikipedia.org/wiki/Hey_Jude",
        revid=1, header={"Song": "Hey Jude", "Section": "Recording"}, text=f"text of {chunk_id}",
        char_start=0, char_end=10,
    )


class _FakeSearcher:
    def __init__(self, *results: list[Chunk]) -> None:
        self._results = list(results)

    def search(self, query: str) -> list[Chunk]:
        return self._results.pop(0)


# --- a scripted model --------------------------------------------------------

def _usage_chunk(prompt: int) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=10, prompt_tokens_details=None)
    return SimpleNamespace(choices=[], usage=usage)


def _searches(query: str, call_id: str = "call_1", name: str = "search") -> list[SimpleNamespace]:
    """A streamed reply that asks for one search."""
    function = SimpleNamespace(name=name, arguments=json.dumps({"query": query}))
    piece = SimpleNamespace(index=0, id=call_id, function=function)
    delta = SimpleNamespace(content=None, tool_calls=[piece])
    return [SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason="tool_calls")], usage=None),
            _usage_chunk(100)]


def _answers(*pieces: str, finish_reason: str = "stop") -> list[SimpleNamespace]:
    """A streamed reply that answers, one chunk per piece of text."""
    chunks = []
    for index, piece in enumerate(pieces):
        last = index == len(pieces) - 1
        delta = SimpleNamespace(content=piece, tool_calls=None)
        chunks.append(SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason if last else None)], usage=None,
        ))
    return chunks + [_usage_chunk(100)]


class _ScriptedModel:
    """Plays one prepared reply per call and records every request."""

    def __init__(self, *replies: list[SimpleNamespace]) -> None:
        self._replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request: object) -> list[SimpleNamespace]:
        self.requests.append(request)
        return self._replies.pop(0)


def _run(model: _ScriptedModel, searcher: _FakeSearcher, settings: LLMConfig,
         history: list[Turn] | None = None, question: str = "Where was Hey Jude recorded?") -> list:
    return list(answer(question, history or [], searcher, model, settings, "SYSTEM"))


# --- the loop ----------------------------------------------------------------

def test_one_step_searches_then_answers() -> None:
    model = _ScriptedModel(_searches("Hey Jude recording studio"), _answers("At Trident ", "Studios [1]."))
    events = _run(model, _FakeSearcher([_chunk("a")]), _settings(max_steps=1))
    assert events[0] == Searching("Hey Jude recording studio")
    assert events[1:3] == [TextDelta("At Trident "), TextDelta("Studios [1].")]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.answer == "At Trident Studios [1]."
    assert done.queries == ["Hey Jude recording studio"]
    assert [c.chunk_id for c in done.sources.chunks] == ["a"]
    assert done.usage == TokenUsage(input_tokens=200, output_tokens=20)
    assert not done.truncated
    assert [r["tool_choice"] for r in model.requests] == [
        {"type": "function", "function": {"name": "search"}}, "none",
    ]


def test_three_steps_can_search_again_before_answering() -> None:
    model = _ScriptedModel(_searches("first", "c1"), _searches("second", "c2"), _answers("Done [2]."))
    searcher = _FakeSearcher([_chunk("a")], [_chunk("b")])
    done = _run(model, searcher, _settings(max_steps=3))[-1]
    assert done.queries == ["first", "second"]
    assert [c.chunk_id for c in done.sources.chunks] == ["a", "b"]
    assert [r["tool_choice"] for r in model.requests][1:] == ["auto", "auto"]


def test_three_steps_stop_as_soon_as_the_model_answers() -> None:
    model = _ScriptedModel(_searches("first"), _answers("Done [1]."))
    done = _run(model, _FakeSearcher([_chunk("a")]), _settings(max_steps=3))[-1]
    assert done.answer == "Done [1]."
    assert len(model.requests) == 2


def test_the_model_is_never_allowed_more_searches_than_max_steps() -> None:
    assert [tool_choice_for(n, max_steps=3, force_first_tool=True) for n in (1, 2, 3, 4)] == [
        "search", "auto", "auto", "none",
    ]
    assert tool_choice_for(1, max_steps=1, force_first_tool=False) == "auto"


def test_a_model_that_ignores_the_last_tool_choice_raises() -> None:
    model = _ScriptedModel(_searches("first", "c1"), _searches("again", "c2"))
    with pytest.raises(RuntimeError):
        _run(model, _FakeSearcher([_chunk("a")], [_chunk("b")]), _settings(max_steps=1))


def test_an_unknown_tool_gets_an_error_message_not_a_crash() -> None:
    model = _ScriptedModel(_searches("x", name="lyrics"), _answers("The sources here don't say."))
    done = _run(model, _FakeSearcher(), _settings(max_steps=1))[-1]
    tool_message = model.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "no tool called 'lyrics'" in tool_message["content"]
    assert done.queries == []


def test_a_cut_off_answer_is_marked_truncated() -> None:
    model = _ScriptedModel(_searches("q"), _answers("The brass was", finish_reason="length"))
    assert _run(model, _FakeSearcher([_chunk("a")]), _settings())[-1].truncated


def test_a_question_that_is_too_long_is_refused_before_any_call() -> None:
    model = _ScriptedModel()
    too_long = "x" * (CONFIG.llm.max_question_chars + 1)
    with pytest.raises(ValueError):
        _run(model, _FakeSearcher(), _settings(), question=too_long)
    assert model.requests == []


# --- messages ----------------------------------------------------------------

def test_history_comes_after_the_system_prompt_without_citations() -> None:
    history = [Turn("Q1", "A1 [1]."), Turn("Q2", "A2 [2]."), Turn("Q3", "A3 [1][2].")]
    messages = build_messages("SYSTEM", history, "Q4", history_turns=2)
    assert [(m.role, m.content) for m in messages] == [
        ("system", "SYSTEM"),
        ("user", "Q2"), ("assistant", "A2."),
        ("user", "Q3"), ("assistant", "A3."),
        ("user", "Q4"),
    ]


def test_no_history_when_history_turns_is_zero() -> None:
    messages = build_messages("SYSTEM", [Turn("Q1", "A1")], "Q2", history_turns=0)
    assert [m.content for m in messages] == ["SYSTEM", "Q2"]


def test_the_search_and_its_result_follow_the_question() -> None:
    model = _ScriptedModel(_searches("Hey Jude studio"), _answers("A [1]."))
    _run(model, _FakeSearcher([_chunk("a")]), _settings())
    roles = [m["role"] for m in model.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]


# --- the prompt --------------------------------------------------------------

def test_the_static_prefix_is_long_enough_to_be_cached() -> None:
    """OpenAI caches a prompt prefix only from 1,024 tokens. The prefix is the
    system prompt plus the tool definitions; a margin covers the difference
    between this tokenizer and the model's own."""
    prompt = load_prompt(ROOT / CONFIG.llm.prompt_path)
    tokenizer = load_tokenizer(CONFIG)
    prefix_tokens = len(tokenizer.encode(prompt)) + len(tokenizer.encode(json.dumps(TOOLS)))
    assert prefix_tokens >= 1_100
