"""Tests for the parts of llm.py that do not need the network. The client is a
small fake that returns prepared responses (or prepared stream chunks), so no
API key is used."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.rag.llm import (
    TOOL_CHOICE_NONE,
    Msg,
    TextDelta,
    TokenUsage,
    ToolCall,
    TurnEnd,
    _to_api,
    complete_structured,
    stream_turn,
)


class Answer(BaseModel):
    text: str


def _fake_client(parsed: Answer | None, usage: SimpleNamespace | None = None) -> SimpleNamespace:
    """Mimics client.chat.completions.parse(...) returning one choice."""
    message = SimpleNamespace(parsed=parsed)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
    completions = SimpleNamespace(parse=lambda **kwargs: response)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_complete_structured_returns_the_parsed_object_and_its_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=200)
    client = _fake_client(Answer(text="Phil Spector"), usage)
    result, tokens = complete_structured(client, "m", [Msg("user", "q")], Answer, 100, {})
    assert result == Answer(text="Phil Spector")
    assert tokens == TokenUsage(input_tokens=900, output_tokens=200)


def test_a_server_that_reports_no_usage_counts_as_zero() -> None:
    client = _fake_client(Answer(text="x"), usage=None)
    _, tokens = complete_structured(client, "m", [Msg("user", "q")], Answer, 100, {})
    assert tokens == TokenUsage()


def test_token_usage_adds_up_and_prices_per_million() -> None:
    total = TokenUsage()
    total.add(TokenUsage(input_tokens=600_000, output_tokens=100_000))
    total.add(TokenUsage(input_tokens=400_000, output_tokens=100_000))
    assert total == TokenUsage(input_tokens=1_000_000, output_tokens=200_000)
    cost = total.cost_usd(input_per_million=0.10, output_per_million=0.50, cached_input_per_million=0.01)
    assert cost == pytest.approx(0.20)


def test_cached_input_tokens_are_priced_at_the_cached_rate() -> None:
    # 1M input, of which 800k cached: 200k x 0.10 + 800k x 0.01 = 0.028
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=800_000)
    cost = usage.cost_usd(input_per_million=0.10, output_per_million=0.50, cached_input_per_million=0.01)
    assert cost == pytest.approx(0.028)


def test_complete_structured_raises_when_nothing_was_parsed() -> None:
    client = _fake_client(None)
    with pytest.raises(ValueError):
        complete_structured(client, "m", [Msg("user", "q")], Answer, 100, {})


# --- stream_turn -------------------------------------------------------------

def _text_chunk(text: str, finish_reason: str | None = None) -> SimpleNamespace:
    delta = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


def _tool_chunk(
    index: int, call_id: str | None, name: str | None, arguments: str, finish_reason: str | None = None
) -> SimpleNamespace:
    piece = SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    delta = SimpleNamespace(content=None, tool_calls=[piece])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)], usage=None)


def _usage_chunk(prompt: int, completion: int, cached: int | None) -> SimpleNamespace:
    details = None if cached is None else SimpleNamespace(cached_tokens=cached)
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=details)
    return SimpleNamespace(choices=[], usage=usage)


class _FakeStreamingClient:
    """Mimics client.chat.completions.create(stream=True) and remembers the
    request it was sent."""

    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self.request: dict[str, object] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._chunks = chunks

    def _create(self, **request: object) -> list[SimpleNamespace]:
        self.request = request
        return self._chunks


SEARCH_TOOL = {"type": "function", "function": {"name": "search", "parameters": {}}}


def _run(client: _FakeStreamingClient, tool_choice: str = "search") -> list[TextDelta | TurnEnd]:
    return list(stream_turn(client, "m", [Msg("user", "q")], [SEARCH_TOOL], tool_choice, 100, {}))


def test_text_arrives_piece_by_piece_and_whole_at_the_end() -> None:
    client = _FakeStreamingClient([
        _text_chunk("Emerick "),
        _text_chunk("did it [1]."),
        _text_chunk("", finish_reason="stop"),
        _usage_chunk(1500, 12, cached=1024),
    ])
    events = _run(client, tool_choice=TOOL_CHOICE_NONE)
    assert events[:2] == [TextDelta("Emerick "), TextDelta("did it [1].")]
    end = events[-1]
    assert end == TurnEnd(
        text="Emerick did it [1].",
        tool_calls=[],
        finish_reason="stop",
        usage=TokenUsage(input_tokens=1500, output_tokens=12, cached_input_tokens=1024),
    )


def test_a_tool_call_split_across_chunks_is_joined_and_parsed() -> None:
    client = _FakeStreamingClient([
        _tool_chunk(0, "call_1", "search", ""),
        _tool_chunk(0, None, None, '{"que'),
        _tool_chunk(0, None, None, 'ry": "Hey Jude eight-track"}', finish_reason="tool_calls"),
        _usage_chunk(1200, 20, cached=0),
    ])
    events = _run(client)
    assert events == [TurnEnd(
        text="",
        tool_calls=[ToolCall("call_1", "search", {"query": "Hey Jude eight-track"})],
        finish_reason="tool_calls",
        usage=TokenUsage(input_tokens=1200, output_tokens=20),
    )]


def test_a_cut_off_answer_is_reported_not_raised() -> None:
    client = _FakeStreamingClient([_text_chunk("The brass was", finish_reason="length")])
    end = _run(client, tool_choice=TOOL_CHOICE_NONE)[-1]
    assert end.finish_reason == "length"
    assert end.text == "The brass was"


def test_a_server_without_usage_or_cache_details_counts_as_zero() -> None:
    no_usage = _FakeStreamingClient([_text_chunk("x", finish_reason="stop")])
    assert _run(no_usage)[-1].usage == TokenUsage()
    no_details = _FakeStreamingClient([_text_chunk("x", finish_reason="stop"), _usage_chunk(10, 1, cached=None)])
    assert _run(no_details)[-1].usage == TokenUsage(input_tokens=10, output_tokens=1)


def test_invalid_tool_arguments_raise() -> None:
    client = _FakeStreamingClient([_tool_chunk(0, "call_1", "search", '{"query": ', finish_reason="tool_calls")])
    with pytest.raises(ValueError):
        _run(client)


def test_the_request_forces_the_named_tool_one_call_at_a_time() -> None:
    client = _FakeStreamingClient([_text_chunk("x", finish_reason="stop")])
    _run(client, tool_choice="search")
    assert client.request["tool_choice"] == {"type": "function", "function": {"name": "search"}}
    assert client.request["parallel_tool_calls"] is False
    assert client.request["stream_options"] == {"include_usage": True}


def test_tools_are_sent_even_when_the_model_must_answer() -> None:
    """They are part of the cached prompt prefix."""
    client = _FakeStreamingClient([_text_chunk("x", finish_reason="stop")])
    _run(client, tool_choice=TOOL_CHOICE_NONE)
    assert client.request["tools"] == [SEARCH_TOOL]
    assert client.request["tool_choice"] == "none"


def test_without_tools_no_tool_parameters_are_sent() -> None:
    client = _FakeStreamingClient([_text_chunk("x", finish_reason="stop")])
    list(stream_turn(client, "m", [Msg("user", "q")], [], TOOL_CHOICE_NONE, 100, {}))
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & client.request.keys()


def test_history_with_a_tool_call_translates_to_the_api_format() -> None:
    history = [
        Msg("user", "Where was Hey Jude recorded?"),
        Msg("assistant", tool_calls=[ToolCall("call_1", "search", {"query": "Hey Jude recording studio"})]),
        Msg("tool", "[1] Song: Hey Jude ...", tool_call_id="call_1"),
    ]
    assert _to_api(history) == [
        {"role": "user", "content": "Where was Hey Jude recorded?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "Hey Jude recording studio"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "[1] Song: Hey Jude ..."},
    ]


@pytest.mark.parametrize("status, sdk_error", [(403, "PermissionDeniedError"), (404, "NotFoundError")])
def test_a_key_without_access_to_the_model_gets_its_own_error(status: int, sdk_error: str) -> None:
    """Retrying never fixes a key that cannot use the model, so the app must
    not tell the user to try again."""
    import httpx
    import openai

    from src.rag.llm import ModelUnavailableError

    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    error = getattr(openai, sdk_error)("no access", response=response, body=None)

    def create(**request: object) -> None:
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(ModelUnavailableError):
        list(stream_turn(client, "m", [Msg("user", "q")], [], TOOL_CHOICE_NONE, 100, {}))


def test_a_rejected_key_becomes_this_module_s_own_error() -> None:
    """The app catches InvalidKeyError without importing openai."""
    import httpx
    import openai

    from src.rag.llm import InvalidKeyError

    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    error = openai.AuthenticationError("bad key", response=response, body=None)

    def create(**request: object) -> None:
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(InvalidKeyError):
        list(stream_turn(client, "m", [Msg("user", "q")], [], TOOL_CHOICE_NONE, 100, {}))
