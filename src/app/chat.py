"""One chat turn of the app: the question in, the chat messages out, updated
as the answer streams.

The page sends a password with every question, never a key. On the Space it
must be the regular password or the admin one (config `access`); the answer
is then paid with the owner's key (`ChatContext.api_key`). The regular
password is limited to `max_questions` per user in each window; the admin one
is not. A local run sets no password (an empty `ctx.password`). A client is
made for that request and dropped after it: the key is never logged or put in
the session state.

On the Space each answered question is saved (`ctx.log_question`, see
question_log.py): the question, the answer and the searches, never the IP or
the password. A local run saves nothing.

Every failure the user can fix (wrong password, no questions left, a
question too long) or that is not theirs (OpenAI errors) becomes a message in
the chat, not an error dialog. llm.py turns the API's errors into its own, so
this module never imports openai."""

from __future__ import annotations

import hmac
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

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
WRONG_PASSWORD = "Wrong password. Enter the password above to ask a question."
RATE_LIMITED = ("You have used your {limit} questions. "
                "Try again in {minutes} minute(s).")
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


class RateLimiter:
    """Allows each user `max_questions` in a window that starts at their first
    question. The counts live here, on the server, so reloading the page does
    not give a user a fresh quota (they are lost when the Space restarts)."""

    def __init__(self, max_questions: int, window_seconds: int) -> None:
        self.max_questions = max_questions
        self._window_seconds = window_seconds
        self._windows: dict[str, tuple[float, int]] = {}  # user -> (window start, questions asked)
        self._lock = threading.Lock()

    def use_one(self, user: str) -> int:
        """Counts a question and returns 0, or returns the seconds until the
        user's window ends when they have none left (and counts nothing)."""
        now = time.monotonic()
        with self._lock:
            start, used = self._windows.get(user, (now, 0))
            if now - start >= self._window_seconds:
                start, used = now, 0
            if used >= self.max_questions:
                return int(start + self._window_seconds - now) + 1
            self._windows[user] = (start, used + 1)
            return 0


@dataclass
class ChatContext:
    """What every turn needs and that is the same for every user: loaded once
    by app.py. `make_client` turns a key into a client (llm.make_client with
    the configured base URL); the tests pass a fake. `api_key` is the owner's.
    An empty `password` means no password is asked (a local run); an empty
    `unlimited_password` never matches."""
    searcher: ChunkSearcher
    settings: LLMConfig
    prices: TokenPrices
    system_prompt: str
    make_client: Callable[[str], LLMClient]
    api_key: str
    limiter: RateLimiter
    password: str = ""
    unlimited_password: str = ""
    log_question: Callable[[dict], None] | None = None  # None: nothing is saved (local runs)


def matches(typed: str, expected: str) -> bool:
    """Constant-time comparison, so response times do not leak the password."""
    return bool(expected) and hmac.compare_digest(typed.encode("utf-8"), expected.encode("utf-8"))


def respond(
    question: str,
    password: str,
    messages: list[Message],
    turns: list[Turn],
    ctx: ChatContext,
    user: str = "",
) -> Iterator[tuple[str, list[Message], list[Turn]]]:
    """Yields (question box, chat messages, turns) each time the chat should
    change: the box is emptied at once, then the assistant message grows.
    `user` identifies who asks, for the rate limit (app.py passes the IP).

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

    typed = password.strip()
    unlimited = matches(typed, ctx.unlimited_password)
    if ctx.password and not unlimited and not matches(typed, ctx.password):
        yield "", with_reply(render.notice(WRONG_PASSWORD)), turns
        return
    if len(question) > ctx.settings.max_question_chars:
        limit = f"{ctx.settings.max_question_chars:,}"
        yield "", with_reply(render.notice(QUESTION_TOO_LONG.format(limit=limit))), turns
        return
    wait = 0 if unlimited else ctx.limiter.use_one(user)
    if wait:
        message = RATE_LIMITED.format(limit=ctx.limiter.max_questions, minutes=-(-wait // 60))
        yield "", with_reply(render.notice(message)), turns
        return

    yield "", with_reply(render.status()), turns
    started = time.monotonic()
    client = ctx.make_client(ctx.api_key)
    text = ""
    record = new_record(question)
    try:
        for event in answer(question, turns, ctx.searcher, client, ctx.settings, ctx.system_prompt):
            if isinstance(event, Searching):
                record["queries"].append(event.query)
                yield "", with_reply(render.status(event.query)), turns
            elif isinstance(event, TextDelta):
                text += event.text
                yield "", with_reply(render.escape_text(text)), turns
            elif isinstance(event, Done):
                seconds = time.monotonic() - started
                cost = event.usage.cost_usd(ctx.prices.input, ctx.prices.output, ctx.prices.cached_input)
                record.update(queries=event.queries, cost_usd=round(cost, 6), completed=True)
                text = event.answer
                final = render.final_answer(event, seconds, cost)
                yield "", with_reply(final), [*turns, Turn(question, event.answer)]
    except LLMError as error:
        yield "", with_reply(render.notice(ERROR_MESSAGES.get(type(error), UNEXPECTED_ERROR))), turns
    except Exception:
        # Logged for the Space's logs; the key is not part of the message.
        logger.exception("unexpected error while answering")
        yield "", with_reply(render.notice(UNEXPECTED_ERROR)), turns
    finally:
        # Runs even when the visitor closes the page mid-answer: Gradio then
        # closes this generator, and the partial answer is still saved.
        record.update(answer=text, seconds=round(time.monotonic() - started, 2))
        save_record(ctx, record)


def new_record(question: str) -> dict:
    """One line of the question log; `completed` turns true only on Done."""
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "answer": "",
        "queries": [],
        "cost_usd": None,
        "seconds": None,
        "completed": False,
    }


def save_record(ctx: ChatContext, record: dict) -> None:
    """Saving a question must never break the answer the visitor sees."""
    if ctx.log_question is None:
        return
    try:
        ctx.log_question(record)
    except Exception:
        logger.exception("could not save the question")
