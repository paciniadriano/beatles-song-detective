"""Reads src/ingest/seeds.yaml: which articles make up the corpus, the doc_type
of each, each song's album, and the few per-article rules. fetch.py uses it to
know what to download and normalize.py to know how to read what was
downloaded, so both read it through this module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

SEEDS_PATH = Path(__file__).parent / "seeds.yaml"
RULE_KEYS = {"title", "keep_sections", "name"}
# The album group of a song that is on none of the corpus's album articles.
NON_ALBUM = "Single/Outtake"


@dataclass(frozen=True)
class Seed:
    title: str
    doc_type: str
    keep_sections: tuple[str, ...] | None = None  # None: keep every section
    album: str | None = None  # songs only: the album group they are listed under
    name: str | None = None   # overrides the name taken from the title


def entity_from_title(title: str) -> str:
    """Strips a trailing disambiguator: "Revolver (Beatles album)" -> "Revolver"."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def load_seeds(path: Path = SEEDS_PATH) -> dict[str, Seed]:
    """Returns every seed keyed by title. Songs are listed under their album,
    every other doc_type as a plain list. Raises on a title listed twice (its
    doc_type or album would depend on the order of the file), on an unknown
    rule, and on a song album that is not an album of the corpus."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    seeds: dict[str, Seed] = {}
    for doc_type, entries in raw.items():
        if doc_type == "song":
            listed = [(entry, album) for album, group in entries.items() for entry in group]
        else:
            listed = [(entry, None) for entry in entries]
        for entry, album in listed:
            seed = _to_seed(entry, doc_type, album)
            if seed.title in seeds:
                raise ValueError(f"{seed.title!r} is listed twice in {path.name}")
            seeds[seed.title] = seed
    _check_song_albums(seeds, path)
    return seeds


def _check_song_albums(seeds: dict[str, Seed], path: Path) -> None:
    """Every song album must be one of the corpus's album articles (or
    NON_ALBUM), so the Album header and filter only ever hold a value that has
    an article, spelt exactly as that article's name."""
    albums = {entity_from_title(seed.title) for seed in seeds.values() if seed.doc_type == "album"}
    unknown = {seed.album for seed in seeds.values() if seed.doc_type == "song"} - albums - {NON_ALBUM}
    if unknown:
        raise ValueError(f"song albums in {path.name} with no album article: {sorted(unknown)}")


def _to_seed(entry: str | dict, doc_type: str, album: str | None) -> Seed:
    if isinstance(entry, str):
        return Seed(title=entry, doc_type=doc_type, album=album)
    unknown = set(entry) - RULE_KEYS
    if unknown:
        raise ValueError(f"{entry.get('title')!r}: unknown keys {sorted(unknown)}")
    keep = entry.get("keep_sections")
    return Seed(
        title=entry["title"],
        doc_type=doc_type,
        keep_sections=tuple(keep) if keep is not None else None,
        album=album,
        name=entry.get("name"),
    )
