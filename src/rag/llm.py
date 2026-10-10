"""The only module that imports `openai` (CLAUDE.md, rule 2). It translates
between the project's own dataclasses (`Msg`, `ToolCall`) and the SDK, so
conversation history never holds raw SDK objects and swapping the SDK touches
this file alone.

Three ways to call the model, chosen by who reads the reply:
- stream_turn(): one step of the agent. Streams the answer as it is written,
  or returns the searches the model asked for.
- complete(): free text in one piece, no tools (the closed-book runs).
- complete_structured(): a reply forced into a Pydantic schema, for code to
  read (generated questions)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TypeVar

import openai
from openai import ContentFilterFinishReasonError, LengthFinishReasonError, OpenAI
from pydantic import BaseModel

# Callers annotate with this alias, so none of them needs to import openai.
LLMClient = OpenAI

Schema = TypeVar("Schema", bound=BaseModel)

# Values for stream_turn()'s `tool_choice`. Any other value is the name of the
# one tool the model must call.
TOOL_CHOICE_AUTO = "auto"  # the model decides: call a tool or answer
TOOL_CHOICE_NONE = "none"  # the model must answer with text


class LLMError(Exception):
    """A failure talking to the model that the app should explain to the user
    in plain words. The app catches these, so it never has to import openai
    (CLAUDE.md, rule 2)."""


class InvalidKeyError(LLMError):
    """The provider rejected the API key."""


class RateLimitedError(LLMError):
    """Too many requests, or the key has no quota left."""


class ConnectionFailedError(LLMError):
    """The provider could not be reached, or failed on its side."""


class ModelUnavailableError(LLMError):
    """The key is valid, but it is not allowed to use the configured model
    (or the model does not exist for it). Trying again never helps, so the
    app says so instead of "try again in a moment"."""


@dataclass
class ToolCall:
    """One tool call the model asked for, with its arguments already parsed."""
    id: str
    name: str
    arguments: dict[str, object]


@dataclass
class Msg:
    """One message of the conversation.

    - system / user: `content` only.
    - assistant: `content`, or `tool_calls` when it asked for tools instead.
    - tool: `content` is the tool's result, `tool_call_id` the call it answers."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class TokenUsage:
    """Tokens consumed by one or more calls, as the API reported them. Kept so
    a run can say what it actually cost instead of guessing from prompt
    lengths.

    `cached_input_tokens` is the part of `input_tokens` that OpenAI served from
    its prompt cache (the same prefix sent recently): billed at a lower price,
    and the evidence that prompt caching works."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens

    def cost_usd(
        self,
        input_per_million: float,
        output_per_million: float,
        cached_input_per_million: float,
    ) -> float:
        uncached_input = self.input_tokens - self.cached_input_tokens
        return (
            uncached_input * input_per_million
            + self.cached_input_tokens * cached_input_per_million
            + self.output_tokens * output_per_million
        ) / 1_000_000


@dataclass
class TextDelta:
    """A piece of the answer, to show as soon as it arrives."""
    text: str


@dataclass
class TurnEnd:
    """The last event of a stream_turn(): what the model did in this step.

    - `text`: the whole answer written in this step ("" if it only called tools).
    - `tool_calls`: the tools it asked for (empty if it answered).
    - `finish_reason`: "stop", "tool_calls", or "length" when the answer was
      cut off by the output-token limit. Reported, not raised: the user has
      already seen the part that arrived.
    - `usage`: tokens of this call."""
    text: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: TokenUsage


def api_key_from_env(name: str) -> str:
    """Local runs only: the Space takes the key pasted in the UI instead."""
    api_key = os.getenv(name)
    if not api_key:
        raise ValueError(f"{name} is not set; put it in .env or the environment")
    return api_key


def make_client(base_url: str | None, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _to_api(messages: list[Msg]) -> list[dict[str, object]]:
    return [_message_to_api(msg) for msg in messages]


def _message_to_api(msg: Msg) -> dict[str, object]:
    if msg.role == "tool":
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    if msg.tool_calls:
        return {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in msg.tool_calls
            ],
        }
    return {"role": msg.role, "content": msg.content}


def _tool_choice_to_api(tool_choice: str) -> object:
    if tool_choice in (TOOL_CHOICE_AUTO, TOOL_CHOICE_NONE):
        return tool_choice
    return {"type": "function", "function": {"name": tool_choice}}


def _usage_of(response: object) -> TokenUsage:
    """Some OpenAI-compatible servers omit `usage`, or its cached-token detail.
    Then it counts as zero, rather than stopping a run over bookkeeping."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0
    return TokenUsage(usage.prompt_tokens, usage.completion_tokens, cached)


def _parse_arguments(name: str, raw: str) -> dict[str, object]:
    """The arguments arrive as a JSON string. Invalid JSON raises: a search
    with a garbled query is worse than a clear error."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"the model sent invalid arguments for {name}: {raw!r}") from error


@dataclass
class _PartialToolCall:
    """A tool call while its pieces are still arriving."""
    id: str = ""
    name: str = ""
    arguments: str = ""


def stream_turn(
    client: OpenAI,
    model: str,
    messages: list[Msg],
    tools: list[dict[str, object]],
    tool_choice: str,
    max_output_tokens: int,
    extra_params: dict[str, object],
) -> Iterator[TextDelta | TurnEnd]:
    """One call to the model, streamed. Yields a TextDelta for each piece of
    answer text, then exactly one TurnEnd.

    `tool_choice` is TOOL_CHOICE_AUTO, TOOL_CHOICE_NONE, or the name of the
    tool the model must call. `tools` is sent on every call, even with
    TOOL_CHOICE_NONE: the tool definitions are part of the prompt prefix
    OpenAI caches, so leaving them out would break the cache.

    At most one tool call per step (parallel_tool_calls=False): otherwise one
    step could run several searches at once, and "max_steps" would no longer
    mean "at most that many searches".

    While streaming, a tool call arrives in pieces (its id and name first,
    then the JSON arguments a few characters at a time), keyed by `index`.
    They are joined here and parsed once the stream ends."""
    request: dict[str, object] = {
        "model": model,
        "messages": _to_api(messages),
        "max_completion_tokens": max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},  # usage arrives in the last chunk
        **extra_params,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = _tool_choice_to_api(tool_choice)
        request["parallel_tool_calls"] = False

    text_parts: list[str] = []
    partial_calls: dict[int, _PartialToolCall] = {}
    finish_reason: str | None = None
    usage = TokenUsage()

    for chunk in _api_chunks(client, request):
        if getattr(chunk, "usage", None) is not None:
            usage = _usage_of(chunk)
        if not chunk.choices:  # the usage-only chunk at the end
            continue
        choice = chunk.choices[0]
        if choice.finish_reason is not None:
            finish_reason = choice.finish_reason
        delta = choice.delta
        if delta.content:
            text_parts.append(delta.content)
            yield TextDelta(delta.content)
        for piece in delta.tool_calls or []:
            call = partial_calls.setdefault(piece.index, _PartialToolCall())
            if piece.id:
                call.id = piece.id
            if piece.function is not None:
                call.name += piece.function.name or ""
                call.arguments += piece.function.arguments or ""

    tool_calls = [
        ToolCall(call.id, call.name, _parse_arguments(call.name, call.arguments))
        for _, call in sorted(partial_calls.items())
    ]
    yield TurnEnd("".join(text_parts), tool_calls, finish_reason, usage)


def _api_chunks(client: OpenAI, request: dict[str, object]) -> Iterator[object]:
    """The streamed chunks of one request, with the SDK's errors translated
    into this module's own. An error can come at the start or halfway through
    the stream, so both are inside the same try."""
    try:
        yield from client.chat.completions.create(**request)
    except openai.AuthenticationError as error:
        raise InvalidKeyError("the API key was rejected") from error
    except openai.RateLimitError as error:
        raise RateLimitedError("rate limit or quota reached") from error
    except (openai.PermissionDeniedError, openai.NotFoundError) as error:
        raise ModelUnavailableError(f"the key cannot use {request.get('model')}") from error
    except (openai.APIConnectionError, openai.InternalServerError) as error:
        raise ConnectionFailedError("the model provider could not be reached") from error


def complete(
    client: OpenAI,
    model: str,
    messages: list[Msg],
    max_output_tokens: int,
    extra_params: dict[str, object],
) -> tuple[str, TokenUsage]:
    """One non-streaming completion without tools, and the tokens it used. An
    empty reply raises: a silent empty string would look like a valid, if
    useless, answer."""
    response = client.chat.completions.create(
        model=model,
        messages=_to_api(messages),
        max_completion_tokens=max_output_tokens,
        **extra_params,
    )
    text = response.choices[0].message.content
    if not text:
        raise ValueError(f"{model} returned an empty completion")
    return text, _usage_of(response)


def complete_structured(
    client: OpenAI,
    model: str,
    messages: list[Msg],
    schema: type[Schema],
    max_output_tokens: int,
    extra_params: dict[str, object],
) -> tuple[Schema, TokenUsage]:
    """One completion whose reply the API guarantees to match `schema`
    (Structured Outputs): exact keys, correct types, and Literal fields limited
    to their allowed values. Returns the validated object (not text) and the
    tokens the call used.

    The schema cannot check meaning, only shape: whether a quote really comes
    from the source text is still the caller's job.

    A bad reply (cut off by the token limit, filtered, or refused) raises
    ValueError, so callers can skip that one item and carry on. Those rare
    calls are paid for but not counted in any TokenUsage. Network and API
    errors are not caught: those should stop the run."""
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=_to_api(messages),
            response_format=schema,
            max_completion_tokens=max_output_tokens,
            **extra_params,
        )
    except (LengthFinishReasonError, ContentFilterFinishReasonError) as error:
        raise ValueError(f"{model} reply was cut off or filtered: {error}") from error
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"{model} returned no {schema.__name__} (refusal or truncated reply)")
    return parsed, _usage_of(response)
