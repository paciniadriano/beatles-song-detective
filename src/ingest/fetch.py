"""Downloads the Wikipedia articles listed in seeds.yaml that are not yet in
data/raw/wikipedia/, one untouched API response per page, as {pageid}.json.

A page already downloaded is never downloaded again: its revision is the one
the corpus is frozen at, and every citation points to it. Nothing here parses
wikitext -- that is normalize.py's job.

    python -m src.ingest.fetch"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from src.config import Config
from src.ingest.seeds import load_seeds

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
BATCH_SIZE = 50  # the MediaWiki API's max titles per query
EMAIL_PLACEHOLDER = "${WIKIPEDIA_CONTACT_EMAIL}"


def user_agent(template: str) -> str:
    """Wikipedia asks API clients for a contact address. It is filled in here,
    the only place that talks to Wikipedia, so nothing else needs it."""
    email = os.getenv("WIKIPEDIA_CONTACT_EMAIL")
    if not email:
        raise ValueError("WIKIPEDIA_CONTACT_EMAIL is required in .env or the environment to fetch")
    return template.replace(EMAIL_PLACEHOLDER, email)


def downloaded_titles(raw_dir: Path) -> set[str]:
    return {
        json.loads(path.read_text(encoding="utf-8"))["title"]
        for path in raw_dir.glob("*.json")
    }


def page_problem(page: dict) -> str | None:
    """Why a page returned by the API must not enter the corpus, or None."""
    if "missing" in page or "invalid" in page:
        return "does not exist"
    if "disambiguation" in page.get("pageprops", {}):
        return "is a disambiguation page; list the specific article instead"
    if page["revisions"][0].get("revid") is None:
        return "has no revid"
    return None


def fetch_pages(titles: list[str], out_dir: Path, session: requests.Session, agent: str, sleep: float) -> list[str]:
    """Writes each valid page and returns the titles written. Titles that are
    redirects, missing or disambiguation pages are collected and reported
    together at the end, after the valid ones are saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    problems: list[str] = []
    for batch in _batched(titles, BATCH_SIZE):
        data = _get(session, {
            "action": "query",
            "prop": "revisions|pageprops",
            "rvprop": "content|ids|timestamp",
            "rvslots": "main",
            "ppprop": "disambiguation",
            "titles": "|".join(batch),
            "redirects": "1",
            "format": "json",
        }, agent, sleep)
        query = data["query"]
        # A seed must be the canonical title, because normalize.py looks the
        # downloaded page up in seeds.yaml by the title Wikipedia returns. A
        # seed that redirects is reported, and the page it led to is not
        # saved (unless that page is also a seed in this batch).
        moves = query.get("normalized", []) + query.get("redirects", [])
        for move in moves:
            problems.append(f"{move['from']!r} redirects: use {move['to']!r} in seeds.yaml")
        reached_by_redirect = {move["to"] for move in moves} - set(batch)
        for page in query["pages"].values():
            if page["title"] in reached_by_redirect:
                continue
            problem = page_problem(page)
            if problem:
                problems.append(f"{page['title']!r} {problem}")
                continue
            record = {"pageid": page["pageid"], "title": page["title"], "raw": page}
            path = out_dir / f"{page['pageid']}.json"
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(page["title"])
    if problems:
        raise RuntimeError(
            "Fix these titles in src/ingest/seeds.yaml and run fetch again "
            f"({len(written)} valid pages were saved):\n" + "\n".join(f"  - {p}" for p in problems)
        )
    return written


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _get(session: requests.Session, params: dict, agent: str, sleep: float) -> dict:
    time.sleep(sleep)
    response = session.get(WIKIPEDIA_API, params=params, headers={"User-Agent": agent}, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    config = Config.load(Path("config.yaml"))
    out_dir = config.paths.raw
    already = downloaded_titles(out_dir) if out_dir.exists() else set()
    to_fetch = [title for title in load_seeds() if title not in already]
    print(f"{len(already)} pages already downloaded, {len(to_fetch)} to fetch")
    if not to_fetch:
        return
    agent = user_agent(config.wikipedia.user_agent)
    with requests.Session() as session:
        written = fetch_pages(to_fetch, out_dir, session, agent, config.wikipedia.sleep_seconds)
    print(f"Fetched {len(written)} pages into {out_dir}:")
    for title in written:
        print(f"  {title}")


if __name__ == "__main__":
    main()
