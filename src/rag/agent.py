"""The agent: turns one user question into a cited answer.

The loop, for `max_steps` = N (the most searches allowed per question):

    call 1          the model must search (force_first_tool)
    calls 2..N      the model chooses: search again, or answer
    call N + 1      the model must answer (no more searches)

With N = 1 that is two calls: search, then answer. If the model answers
early, the loop ends there. The same prompt serves every N: the loop, not the
prompt, decides when searching stops (the max_steps stage chose N = 1: a pre-check
showed N = 2 could not win, so it never ran).

The messages always keep the same order, so the start of every request is the
same and OpenAI can cache it (CLAUDE.md, trap 11):

    system prompt -> earlier turns -> this question -> searches and results

answer() yields events as they happen: Searching when a search starts,
TextDelta for each piece of the answer, and one Done at the end. The app shows
them; the evaluation reads Done."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from src.config import LLMConfig
from src.rag.llm import (
    TOOL_CHOICE_AUTO,
    TOOL_CHOICE_NONE,
    LLMClient,
    Msg,
    TextDelta,
    TokenUsage,
    ToolCall,
    TurnEnd,
    stream_turn,
)
from src.rag.tools import SEARCH_TOOL_NAME, TOOLS, ChunkSearcher, Sources, run_search, strip_citations


@dataclass
class Turn:
    """One earlier question and the answer it got, for the history."""
    question: str
    answer: str


@dataclass
class Searching:
    """A search is starting; the app can show its query."""
    query: str


@dataclass
class Done:
    """The last event of answer().

    - `answer`: the final answer, with its [n] citations.
    - `sources`: the chunk behind each [n].
    - `queries`: every search made, in order.
    - `usage`: tokens of all the calls together.
    - `truncated`: the answer was cut off by the output-token limit."""
    answer: str
    sources: Sources
    queries: list[str]
    usage: TokenUsage
    truncated: bool


AgentEvent = Searching | TextDelta | Done


def load_prompt(path: Path) -> str:
    """Read once at the entry point (app.py, the evaluation) and passed down."""
    return path.read_text(encoding="utf-8")


def build_messages(system_prompt: str, history: list[Turn], question: str, history_turns: int) -> list[Msg]:
    """The system prompt, the last `history_turns` turns, then the question.
    Earlier answers go in without their [n]: those numbers pointed at sources
    the model no longer sees, and this question's sources start again at [1].
    Earlier searches and their results are left out too: about 1,600 tokens
    each, and the question and answer are enough to understand a follow-up."""
    recent = history[-history_turns:] if history_turns > 0 else []
    messages = [Msg("system", system_prompt)]
    for turn in recent:
        messages.append(Msg("user", turn.question))
        messages.append(Msg("assistant", strip_citations(turn.answer)))
    messages.append(Msg("user", question))
    return messages


def tool_choice_for(call_number: int, max_steps: int, force_first_tool: bool) -> str:
    """What the model may do on this call (see the table at the top)."""
    if call_number > max_steps:
        return TOOL_CHOICE_NONE
    if call_number == 1 and force_first_tool:
        return SEARCH_TOOL_NAME
    return TOOL_CHOICE_AUTO


def unknown_tool_message(name: str) -> str:
    return f"There is no tool called {name!r}. The only tool is {SEARCH_TOOL_NAME!r}."


def answer(
    question: str,
    history: list[Turn],
    searcher: ChunkSearcher,
    client: LLMClient,
    settings: LLMConfig,
    system_prompt: str,
) -> Iterator[AgentEvent]:
    """Answers one question; see the module docstring for the loop.

    A question longer than `max_question_chars` raises ValueError before any
    call is made, so the limit holds for the app and the evaluation alike.
    Network and API errors are not caught: the caller shows them."""
    if len(question) > settings.max_question_chars:
        raise ValueError(f"The question is longer than {settings.max_question_chars} characters")

    messages = build_messages(system_prompt, history, question, settings.history_turns)
    sources = Sources()
    queries: list[str] = []
    usage = TokenUsage()

    for call_number in range(1, settings.max_steps + 2):
        tool_choice = tool_choice_for(call_number, settings.max_steps, settings.force_first_tool)
        end: TurnEnd | None = None
        for event in stream_turn(
            client, settings.model, messages, TOOLS, tool_choice,
            settings.max_output_tokens, settings.extra_params,
        ):
            if isinstance(event, TextDelta):
                yield event
            else:
                end = event
        assert end is not None  # stream_turn always ends with a TurnEnd
        usage.add(end.usage)

        if not end.tool_calls:
            yield Done(end.text, sources, queries, usage, truncated=end.finish_reason == "length")
            return

        messages.append(Msg("assistant", end.text, tool_calls=end.tool_calls))
        for call in end.tool_calls:
            if call.name == SEARCH_TOOL_NAME:
                query = _query_of(call)
                queries.append(query)
                yield Searching(query)
                result = run_search(query, searcher, sources)
            else:
                result = unknown_tool_message(call.name)
            messages.append(Msg("tool", result, tool_call_id=call.id))

    # The last call has tool_choice "none", so the model cannot get here by
    # the rules; a server that ignored them would.
    raise RuntimeError(f"{settings.model} kept calling tools after {settings.max_steps} searches")


def _query_of(call: ToolCall) -> str:
    query = call.arguments.get("query", "")
    return query if isinstance(query, str) else ""
