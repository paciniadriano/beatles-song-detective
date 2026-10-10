"""Tests for the guards in src/ingest/fetch.py that keep a wrong page out of
the corpus. The download itself is not tested: it needs the network."""

import pytest

from src.ingest.fetch import page_problem, user_agent

VALID = {"title": "Help! (song)", "pageid": 1, "revisions": [{"revid": 10}]}


def test_a_valid_page_has_no_problem() -> None:
    assert page_problem(VALID) is None


def test_a_missing_page_is_refused() -> None:
    assert page_problem({"title": "Nope", "missing": ""}) == "does not exist"


def test_a_disambiguation_page_is_refused() -> None:
    page = {**VALID, "pageprops": {"disambiguation": ""}}
    assert "disambiguation" in page_problem(page)


def test_a_page_without_revid_is_refused() -> None:
    assert page_problem({**VALID, "revisions": [{}]}) == "has no revid"


def test_the_contact_address_is_required_only_to_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WIKIPEDIA_CONTACT_EMAIL", raising=False)
    with pytest.raises(ValueError, match="WIKIPEDIA_CONTACT_EMAIL"):
        user_agent("Bot (${WIKIPEDIA_CONTACT_EMAIL})")
    monkeypatch.setenv("WIKIPEDIA_CONTACT_EMAIL", "me@example.org")
    assert user_agent("Bot (${WIKIPEDIA_CONTACT_EMAIL})") == "Bot (me@example.org)"
