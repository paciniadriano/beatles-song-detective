"""Tests for the app's own logic: what render.py writes into a message, what
chat.respond() shows for each event and each failure, and where startup.py
finds the store. The agent is replaced by a fake that yields prepared
events, so no key, model or Qdrant store is needed. Gradio itself is not
tested here: run `python app.py` and look."""

from pathlib import Path

import pytest

from src.app import chat, render, startup
from src.app.chat import ChatContext, LockedSearcher, respond
from src.config import Config, TokenPrices
from src.rag.agent import Done, Searching, Turn
from src.rag.llm import InvalidKeyError, ModelUnavailableError, RateLimitedError, TextDelta, TokenUsage
from src.rag.tools import Sources
from src.schemas import Chunk

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Config.load(ROOT / "config.yaml")


def _chunk(chunk_id: str, title: str = "Hey Jude", section: str = "Recording") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="wp_1", section_id="wp_1__s1", doc_type="song", topics=["recording"],
        title=title, section_path=section, url="https://en.wikipedia.org/wiki/Hey_Jude",
        revid=123, header={"Song": title}, text="text", char_start=0, char_end=4,
    )


def _sources(*chunks: Chunk) -> Sources:
    sources = Sources()
    for chunk in chunks:
        sources.add(chunk)
    return sources


# --- render: the model's text is shown as text ------------------------------

@pytest.mark.parametrize("text", [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
])
def test_html_in_the_answer_is_shown_not_run(text: str) -> None:
    escaped = render.escape_text(text)
    assert "<" not in escaped and ">" not in escaped


@pytest.mark.parametrize("text", [
    "[click here](https://evil.example)",
    "**bold** and _italic_",
    "# A heading",
    "Visit https://evil.example now",
    "![image](https://evil.example/x.png)",
])
def test_markdown_in_the_answer_cannot_make_links_or_formatting(text: str) -> None:
    escaped = render.escape_text(text)
    for char in "[]()*_#:/!":
        assert char not in escaped.replace("\\" + char, "")


def test_an_ampersand_is_shown_once_not_as_an_entity() -> None:
    assert render.escape_text("Lennon & McCartney") == "Lennon &amp; McCartney"


def test_citations_link_to_their_source_and_unknown_numbers_stay_text() -> None:
    sources = _sources(_chunk("c1"), _chunk("c2", section="Personnel"))
    html = render.link_citations(render.escape_text("Paul sang [1] and George played [2], see [7]."), sources)
    assert 'href="https://en.wikipedia.org/w/index.php?oldid=123#Recording"' in html
    assert 'href="https://en.wikipedia.org/w/index.php?oldid=123#Personnel"' in html
    assert html.count('class="beatle_rag_citation"') == 2
    assert "\\[7\\]" in html  # never shown to the model: no link


def test_the_sources_block_lists_only_cited_sources_in_citation_order() -> None:
    sources = _sources(_chunk("c1", title="Hey Jude"), _chunk("c2", title="Let It Be"), _chunk("c3", title="Help!"))
    block = render.sources_block("First [3], then [1] and [3] again.", sources)
    assert block.index("Help!") < block.index("Hey Jude")
    assert "Let It Be" not in block
    assert block.count('class="beatle_rag_source"') == 2
    assert "\n" not in block  # one line: a blank line would end the HTML block


def test_a_decline_has_no_sources_block() -> None:
    assert render.sources_block("The sources here don't say.", _sources(_chunk("c1"))) == ""


def test_titles_from_the_corpus_are_escaped_too() -> None:
    block = render.sources_block("x [1]", _sources(_chunk("c1", title="A <b>title</b>")))
    assert "<b>" not in block


def test_the_final_answer_carries_the_meta_line_and_a_cut_off_notice() -> None:
    done = Done("Yes [1].", _sources(_chunk("c1")), ["q"], TokenUsage(), truncated=True)
    message = render.final_answer(done, seconds=2.06, cost_usd=0.00091)
    assert "2.1 s · ~$0.0009" in message
    assert render.TRUNCATED_NOTICE in message
    assert 'class="beatle_rag_sources"' in message


# --- chat.respond --------------------------------------------------------------

PRICES = TokenPrices(input=0.10, cached_input=0.01, output=0.50)


def _ctx(fallback_api_key: str = "") -> ChatContext:
    return ChatContext(
        searcher=LockedSearcher(searcher=None),  # the fake agent never searches
        settings=CONFIG.llm,
        prices=PRICES,
        system_prompt="prompt",
        make_client=lambda key: f"client for {key}",
        fallback_api_key=fallback_api_key,
    )


def _fake_agent(*events: object, error: Exception | None = None):
    def fake_answer(question, history, searcher, client, settings, system_prompt):
        yield from events
        if error is not None:
            raise error
    return fake_answer


def _last(question: str, api_key: str = "sk-test", turns: list[Turn] | None = None):
    return list(respond(question, api_key, [], turns or [], _ctx()))[-1]


def test_an_answer_streams_then_joins_the_history(monkeypatch: pytest.MonkeyPatch) -> None:
    done = Done("Paul [1].", _sources(_chunk("c1")), ["who sang"], TokenUsage(input_tokens=1000), truncated=False)
    monkeypatch.setattr(chat, "answer", _fake_agent(Searching("who sang"), TextDelta("Paul "), TextDelta("[1]."), done))
    steps = list(respond("Who sang?", "sk-test", [], [], _ctx()))

    contents = [messages[-1]["content"] for _, messages, _ in steps]
    assert "Searching Wikipedia" in contents[0]
    assert "who sang" in contents[1]
    assert contents[2] == "Paul "
    assert 'class="beatle_rag_citation"' in contents[-1]

    box, messages, turns = steps[-1]
    assert box == ""
    assert messages[0] == {"role": "user", "content": render.escape_text("Who sang?")}
    assert turns == [Turn("Who sang?", "Paul [1].")]


def test_without_a_key_nothing_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "answer", _fake_agent(error=AssertionError("the agent must not run")))
    _, messages, turns = _last("Who sang?", api_key="  ")
    assert chat.MISSING_KEY in messages[-1]["content"]
    assert turns == []


def _client_used(monkeypatch: pytest.MonkeyPatch, api_key: str, fallback_api_key: str) -> str:
    """Which client the agent was given, for a key box and a fallback key."""
    used: list[str] = []

    def fake_answer(question, history, searcher, client, settings, system_prompt):
        used.append(client)
        yield Done("x", Sources(), [], TokenUsage(), False)

    monkeypatch.setattr(chat, "answer", fake_answer)
    list(respond("Who sang?", api_key, [], [], _ctx(fallback_api_key)))
    return used[0]


def test_locally_an_empty_box_uses_the_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _client_used(monkeypatch, api_key="", fallback_api_key="sk-env") == "client for sk-env"


def test_a_pasted_key_wins_over_the_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _client_used(monkeypatch, api_key=" sk-pasted ", fallback_api_key="sk-env") == "client for sk-pasted"


def test_a_question_over_the_limit_is_refused_before_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "answer", _fake_agent(error=AssertionError("the agent must not run")))
    _, messages, _ = _last("x" * (CONFIG.llm.max_question_chars + 1))
    assert 'class="beatle_rag_notice"' in messages[-1]["content"]


def test_an_empty_question_changes_nothing() -> None:
    assert list(respond("   ", "sk-test", [], [], _ctx())) == [("", [], [])]


@pytest.mark.parametrize("error, expected", [
    (InvalidKeyError("x"), chat.ERROR_MESSAGES[InvalidKeyError]),
    (RateLimitedError("x"), chat.ERROR_MESSAGES[RateLimitedError]),
    (ModelUnavailableError("x"), chat.ERROR_MESSAGES[ModelUnavailableError]),
    (RuntimeError("x"), chat.UNEXPECTED_ERROR),
])
def test_errors_become_a_message_and_leave_the_history_alone(
    monkeypatch: pytest.MonkeyPatch, error: Exception, expected: str,
) -> None:
    earlier = [Turn("Earlier?", "Earlier answer.")]
    monkeypatch.setattr(chat, "answer", _fake_agent(TextDelta("Half an ans"), error=error))
    _, messages, turns = _last("Who sang?", turns=earlier)
    assert expected in messages[-1]["content"]
    assert turns == earlier


def test_the_user_s_question_is_escaped_in_the_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "answer", _fake_agent(Done("x", Sources(), [], TokenUsage(), False)))
    _, messages, _ = _last("<b>Who</b>?")
    assert "<b>" not in messages[0]["content"]


# --- startup -----------------------------------------------------------------

def test_only_the_dataset_repo_variable_means_the_space(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG.space.dataset_repo_env, raising=False)
    assert not startup.runs_on_space(CONFIG)
    monkeypatch.setenv(CONFIG.space.dataset_repo_env, "someone/beatles-store")
    assert startup.runs_on_space(CONFIG)


def test_locally_the_published_store_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(CONFIG.space.dataset_repo_env, raising=False)
    config = CONFIG.model_copy(update={"paths": CONFIG.paths.model_copy(update={"qdrant_store": tmp_path})})
    assert startup.prepare_store(config) == tmp_path


def test_on_the_space_the_store_is_downloaded_and_copied_without_its_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    download = tmp_path / "download"
    (download / startup.STORE_FOLDER).mkdir(parents=True)
    (download / startup.STORE_FOLDER / "meta.json").write_text("{}")
    (download / startup.STORE_FOLDER / ".lock").write_text("")
    requested: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        requested.update(kwargs)
        return str(download)

    monkeypatch.setenv(CONFIG.space.dataset_repo_env, "someone/beatles-store")
    monkeypatch.setattr(startup, "snapshot_download", fake_snapshot_download)
    runtime = tmp_path / "runtime"
    config = CONFIG.model_copy(update={"space": CONFIG.space.model_copy(update={"qdrant_runtime_path": runtime})})

    assert startup.prepare_store(config) == runtime
    assert requested["repo_id"] == "someone/beatles-store"
    assert requested["repo_type"] == "dataset"
    assert (runtime / "meta.json").exists()
    assert not (runtime / ".lock").exists()


# --- what the Space installs --------------------------------------------------

def test_the_app_imports_nothing_the_space_does_not_install() -> None:
    """The Space installs requirements.txt only. The ingest and chunking code
    needs packages from requirements-dev.txt (mwparserfromhell), so if app.py
    reached it through some import, the Space would crash at startup. The
    offline scripts (build, publish, the evaluation) must not load either:
    nothing the app runs should depend on them. Run in a fresh interpreter:
    this test session has already imported everything."""
    import subprocess
    import sys

    check = (
        "import sys, app; "
        "loaded = [m for m in sys.modules if m.startswith(('src.ingest', 'src.index.chunking', 'src.index.build', 'src.index.publish', 'src.eval', 'mwparserfromhell'))]; "
        "print(loaded)"
    )
    result = subprocess.run([sys.executable, "-c", check], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
