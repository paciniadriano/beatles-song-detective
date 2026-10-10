"""One chat turn of the app: the question in, the chat messages out, updated
as the answer streams.

The API key comes from the page with every question; locally, an empty box
falls back to the key in `.env` (`fallback_api_key`, always empty on the
Space). A client is made for that request and dropped after it: the key is
never stored, logged or put in the session state (CLAUDE.md, "Operating constraints").

Every failure the user can fix (no key, a key OpenAI rejects, no credit
left, a key without access to the model, a question too long) becomes a message in the chat, not an error
dialog. llm.py turns the API's errors into its own, so this module never
imports openai."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from src.app import render
from src.config import LLMConfig, TokenPrices
from src.rag.agent import Done, Searching, Turn, answer
from src.rag.llm import (
    ConnectionFailedError,
    InvalidKeyError,
    LLMClient,
    LLMError,
    ModelUnavailableError,
    RateLimitedError,
    TextDelta,
)
from src.rag.tools import ChunkSearcher
from src.schemas import Chunk

logger = logging.getLogger(__name__)

# What the user reads for each failure. Plain words, and what to do next.
MISSING_KEY = "Enter your OpenAI API key above to ask a question."
QUESTION_TOO_LONG = "Please keep the question under {limit} characters."
ERROR_MESSAGES: dict[type[LLMError], str] = {
    InvalidKeyError: "OpenAI rejected this API key. Check that it is complete and still active.",
    RateLimitedError: "OpenAI says this key has reached its rate limit or has no credit left. "
                      "Wait a minute, or check the key's billing.",
    ConnectionFailedError: "OpenAI could not be reached. Try again in a moment.",
    ModelUnavailableError: "This API key cannot use the model the app needs. Check the key's "
                           "project permissions in your OpenAI account, or try another key.",
}
UNEXPECTED_ERROR = "Something went wrong while answering. Try again in a moment."

# A chat message as gr.Chatbot takes it: {"role": "user" | "assistant", "content": str}.
Message = dict[str, str]


class LockedSearcher:
    """Lets one search run at a time. Gradio answers up to
    `space.concurrency_limit` questions at once, each in its own thread, and
    the local Qdrant store and the embedding models are not meant to be used
    from two threads together. A search takes well under a second; the
    answer's streaming, the slow part, is not locked."""

    def __init__(self, searcher: ChunkSearcher) -> None:
        self._searcher = searcher
        self._lock = threading.Lock()

    def search(self, query: str) -> list[Chunk]:
        with self._lock:
            return self._searcher.search(query)


@dataclass
class ChatContext:
    """What every turn needs and that is the same for every user: loaded once
    by app.py. `make_client` turns a key into a client (llm.make_client with
    the configured base URL); the tests pass a fake. `fallback_api_key` is
    used when the key box is empty: the `.env` key locally, "" on the Space."""
    searcher: ChunkSearcher
    settings: LLMConfig
    prices: TokenPrices
    system_prompt: str
    make_client: Callable[[str], LLMClient]
    fallback_api_key: str = ""


def respond(
    question: str,
    api_key: str,
    messages: list[Message],
    turns: list[Turn],
    ctx: ChatContext,
) -> Iterator[tuple[str, list[Message], list[Turn]]]:
    """Yields (question box, chat messages, turns) each time the chat should
    change: the box is emptied at once, then the assistant message grows.

    `turns` is the history the model sees; only answered questions join it.
    Both lists are copied, never changed in place, so an error halfway leaves
    the session as it was."""
    question = question.strip()
    if not question:
        yield question, messages, turns
        return

    shown = [*messages, {"role": "user", "content": render.escape_text(question)}]

    def with_reply(content: str) -> list[Message]:
        return [*shown, {"role": "assistant", "content": content}]

    key = api_key.strip() or ctx.fallback_api_key  # a pasted key always wins
    if not key:
        yield "", with_reply(render.notice(MISSING_KEY)), turns
        return
    if len(question) > ctx.settings.max_question_chars:
        limit = f"{ctx.settings.max_question_chars:,}"
        yield "", with_reply(render.notice(QUESTION_TOO_LONG.format(limit=limit))), turns
        return

    yield "", with_reply(render.status()), turns
    started = time.monotonic()
    client = ctx.make_client(key)
    text = ""
    try:
        for event in answer(question, turns, ctx.searcher, client, ctx.settings, ctx.system_prompt):
            if isinstance(event, Searching):
                yield "", with_reply(render.status(event.query)), turns
            elif isinstance(event, TextDelta):
                text += event.text
                yield "", with_reply(render.escape_text(text)), turns
            elif isinstance(event, Done):
                seconds = time.monotonic() - started
                cost = event.usage.cost_usd(ctx.prices.input, ctx.prices.output, ctx.prices.cached_input)
                final = render.final_answer(event, seconds, cost)
                yield "", with_reply(final), [*turns, Turn(question, event.answer)]
    except LLMError as error:
        yield "", with_reply(render.notice(ERROR_MESSAGES.get(type(error), UNEXPECTED_ERROR))), turns
    except Exception:
        # Logged for the Space's logs; the key is not part of the message.
        logger.exception("unexpected error while answering")
        yield "", with_reply(render.notice(UNEXPECTED_ERROR)), turns
