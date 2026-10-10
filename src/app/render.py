"""Turns the agent's output into what the chat shows. Pure functions: no
Gradio, no network, so every rule here is tested.

The model's text is shown as text, never as HTML or Markdown. The chat
renders Markdown, so the text is escaped twice:
1. HTML: `<script>` arrives as the characters "<script>", not a tag.
2. Markdown: every ASCII punctuation mark gets a backslash, so "**", "[x](y)",
   "# " or a bare URL show as typed instead of becoming bold text, a link or
   a heading. (`&` and `;` are left alone: step 1 wrote them as entities.)

The only HTML in a message is the HTML written here: the citation links, the
sources block and the meta line. Every link is built from a Chunk the search
returned (tools.citation_url), never from the model's words, so an answer
cannot point the user anywhere else (CLAUDE.md, agent and app)."""

from __future__ import annotations

import html
import re
import string

from src.rag.agent import Done
from src.rag.tools import Sources, cited_numbers, citation_url

# Step 2 above. `&` and `;` are excluded: they are part of the entities that
# step 1 wrote ("&lt;"), and a backslash would show them as typed.
_MARKDOWN_PUNCTUATION = set(string.punctuation) - {"&", ";"}

# A citation after escaping: "[3]" has become "\[3\]".
_ESCAPED_CITATION = re.compile(r"\\\[(\d+)\\\]")

SOURCES_TITLE = "Sources"
TRUNCATED_NOTICE = "The answer was cut off at its length limit."


def escape_text(text: str) -> str:
    """Model or user text, to show exactly as written."""
    escaped = html.escape(text, quote=False)
    return "".join("\\" + char if char in _MARKDOWN_PUNCTUATION else char for char in escaped)


def _link(url: str, label: str, css_class: str) -> str:
    """An HTML link that opens in a new tab and does not tell Wikipedia which
    page sent the user."""
    return (
        f'<a class="{css_class}" href="{html.escape(url)}" '
        f'target="_blank" rel="noopener noreferrer">{label}</a>'
    )


def link_citations(escaped_answer: str, sources: Sources) -> str:
    """Turns each [n] of an escaped answer into a link to its source. A
    number the model was never shown stays as plain text: there is nothing
    to link it to. The brackets are written as entities so the Markdown
    renderer does not read them as a link of its own."""
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        chunk = sources.get(number)
        if chunk is None:
            return match.group(0)
        return _link(citation_url(chunk), f"&#91;{number}&#93;", "beatle_rag_citation")

    return _ESCAPED_CITATION.sub(replace, escaped_answer)


def sources_block(answer: str, sources: Sources) -> str:
    """The sources the answer cites, in the order it cites them, each with
    its article and section. Empty when it cites none (a decline). Written on
    one line: a Markdown HTML block ends at the first blank line."""
    rows = []
    for number in cited_numbers(answer):
        chunk = sources.get(number)
        if chunk is None:
            continue
        title = _link(citation_url(chunk), html.escape(chunk.title), "beatle_rag_source_link")
        section = html.escape(chunk.section_path)
        rows.append(
            f'<div class="beatle_rag_source">'
            f'<span class="beatle_rag_source_number">[{number}]</span> '
            f'{title} <span class="beatle_rag_source_section">· {section}</span></div>'
        )
    if not rows:
        return ""
    return (
        f'<div class="beatle_rag_sources">'
        f'<div class="beatle_rag_sources_title">{SOURCES_TITLE}</div>{"".join(rows)}</div>'
    )


def meta_line(seconds: float, cost_usd: float) -> str:
    """How long the answer took and roughly what it cost the user's key."""
    return f'<div class="beatle_rag_answer_meta">{seconds:.1f} s · ~${cost_usd:.4f}</div>'


def notice(text: str) -> str:
    """A message from the app itself (missing key, error...), not from the
    model. `text` is written by the app, so it is trusted as is."""
    return f'<div class="beatle_rag_notice">{text}</div>'


def status(query: str | None = None) -> str:
    """Shown while a search runs: with its query once the model has written
    it. The query is the model's, so it is escaped."""
    if query is None:
        return '<div class="beatle_rag_status">Searching Wikipedia…</div>'
    return f'<div class="beatle_rag_status">Searching Wikipedia: {html.escape(query)}…</div>'


def final_answer(done: Done, seconds: float, cost_usd: float) -> str:
    """The whole message once the answer is complete: the answer with linked
    citations, the sources it cites, a notice if it was cut off, and the meta
    line. Blank lines keep each HTML block apart from the answer's text."""
    parts = [link_citations(escape_text(done.answer), done.sources)]
    sources = sources_block(done.answer, done.sources)
    if sources:
        parts.append(sources)
    if done.truncated:
        parts.append(notice(TRUNCATED_NOTICE))
    parts.append(meta_line(seconds, cost_usd))
    return "\n\n".join(parts)
