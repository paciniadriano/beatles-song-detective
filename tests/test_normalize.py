"""Tests for normalize.py's pure functions: how headings are given topics, which
templates keep their text, which sections survive, and the offsets that
chunking and evaluation rely on. The topic and drop cases run against the
real lists in config.yaml."""

import json
from pathlib import Path

import mwparserfromhell
import pytest
import tiktoken

from src.config import Config, TopicsConfig
from src.ingest.normalize import (
    heading_topics,
    _remove_non_prose,
    build_sections,
    is_dropped,
    names_any,
    path_topics,
    read_raw_records,
    section_paths,
    select_sections,
    split_headings,
    unwrap_templates,
)
from src.ingest.seeds import NON_ALBUM, Seed, load_seeds

CONFIG = Config.load(Path(__file__).resolve().parent.parent / "config.yaml")
TOPICS = CONFIG.topics
WIKIPEDIA = CONFIG.wikipedia

HEADING_CASES = [
    ("Music", ["writing"]),                          # exact title
    ("Recording and release", ["recording", "release"]),  # a mixed heading gets both
    ("Background and recording", ["recording", "writing"]),
    ("Music video", ["media"]),
    ("Cover art", ["media"]),
    ("Cover versions", ["cover_versions"]),
    ("Musicianship", ["musicianship"]),              # whole words: not "musician"
    ("Personnel", ["personnel"]),
    ("Musical structure", ["writing"]),
    ("The Beatles' version", ["recording"]),
    ("The Beatles’ version", ["recording"]),         # typographic apostrophe
    ("McCartney live performances", ["live_performance"]),
    ("Early life", ["biography"]),
    ("Critical reception", ["reception"]),
    ("Unreleased material", ["recording"]),          # whole words: not "release"
    ("Early influences", ["writing"]),               # what shaped them; "influence" alone is reception
    ("Formation", ["general"]),                      # whole words: not "formats"
    ("Description", ["general"]),                    # a building, a guitar, a technique: not writing
    ("11 August 1969", ["general"]),
]


@pytest.mark.parametrize("title, expected", HEADING_CASES)
def test_a_heading_gets_every_topic_it_names(title: str, expected: list[str]) -> None:
    assert path_topics(title, TOPICS) == expected


PATH_CASES = [
    ("Recording > Take 7", ["recording"]),            # no signal: the ancestor decides
    ("Recording > Cover versions", ["cover_versions"]),  # a heading with signal is never overridden
    ("Releases > Continental Europe", ["release"]),
    ("Songs > Side one > \"Taxman\"", ["general"]),
    ("Personnel > The Beatles", ["personnel"]),       # a bare "The Beatles" inherits its parent
]


@pytest.mark.parametrize("path, expected", PATH_CASES)
def test_a_heading_without_signal_inherits_from_its_nearest_ancestor(path: str, expected: list[str]) -> None:
    assert path_topics(path, TOPICS) == expected


def test_names_any_matches_whole_words_only() -> None:
    assert names_any("Notes and references", ["notes"])
    assert not names_any("Denotes", ["notes"])
    assert names_any("Career > The Beatles", ["Career > The Beatles"])
    assert not names_any("Beatlesque", ["Beatles"])


def test_a_plural_ending_in_parentheses_is_optional() -> None:
    assert names_any("Origin", ["origin(s)"])
    assert names_any("Origins and inspiration", ["origin(s)"])
    assert not names_any("Originals", ["origin(s)"])
    assert names_any("Mixes", ["mix(es)"])
    assert names_any("Live performances", ["live performance(s)"])


def test_a_phrase_without_parentheses_matches_only_its_own_form() -> None:
    """"influence" is reception and "influences" is writing: the notation is
    what keeps them apart."""
    assert names_any("Influence and legacy", ["influence"])
    assert not names_any("Early influences", ["influence"])


def test_other_parentheses_are_literal() -> None:
    assert names_any("Career (1958–2009)", ["Career (1958–2009)"])
    assert not names_any("Career", ["Career (1958–2009)"])


def test_an_exact_title_is_compared_like_every_other_heading_rule() -> None:
    """Case and apostrophe style do not matter, as for the keywords; the
    exact title still has to be the whole heading."""
    topics = TopicsConfig(
        exact_titles={"The Beatles' version": "recording"},
        keywords={}, fallback="general", out_of_scope=[],
    )
    assert heading_topics("The Beatles’ version", topics) == {"recording"}
    assert heading_topics("THE BEATLES' VERSION", topics) == {"recording"}
    assert heading_topics("The Beatles' version live", topics) == set()


@pytest.mark.parametrize("phrase", ["origin (s)", "documentar(ies)", "(s)", "origin(s", "take(s)s"])
def test_a_malformed_plural_ending_is_refused_when_the_config_loads(phrase: str) -> None:
    with pytest.raises(ValueError, match="plural ending"):
        TopicsConfig(exact_titles={}, keywords={"writing": [phrase]}, fallback="general", out_of_scope=[])


DROP_CASES = [
    ("References", True),
    ("Charts and certifications", True),       # apparatus naming only an out-of-scope topic
    ("Personal life and death", True),         # private: always
    ("Family and personal life", True),
    ("Cultural references", False),            # names media, an in-scope topic
    ("Track listing and recording", False),
    ("Recording", False),
]


@pytest.mark.parametrize("title, dropped", DROP_CASES)
def test_private_headings_always_go_and_apparatus_goes_unless_it_names_an_in_scope_topic(title: str, dropped: bool) -> None:
    assert is_dropped(title, WIKIPEDIA, TOPICS) is dropped


def test_every_out_of_scope_topic_exists() -> None:
    assert set(TOPICS.out_of_scope) <= TOPICS.names()


TEMPLATE_CASES = [
    ("chorus (B, F{{music|sharp}}, E)", "chorus (B, F♯, E)"),
    ("in {{music|time|5|4}} time", "in 5/4 time"),
    ("a key of B{{flat}} major", "a key of B♭ major"),
    ("He said: {{quote box|quote=It was fun.|source=Lennon}}", "He said: It was fun."),
    ("He said: {{Blockquote|text=It was fun.}}", "He said: It was fun."),
    ("a {{convert|15|in|mm}} drum", "a 15 in drum"),
    ("Lennon{{'s}} song{{sfn|MacDonald|2007|p=1}}", "Lennon's song"),
    ("{{nowrap|pop rock}}", "pop rock"),
]


@pytest.mark.parametrize("wikitext, expected", TEMPLATE_CASES)
def test_templates_keep_their_text(wikitext: str, expected: str) -> None:
    code = mwparserfromhell.parse(wikitext)
    unwrap_templates(code)
    assert str(code) == expected


def test_footnotes_categories_and_images_are_not_prose() -> None:
    code = mwparserfromhell.parse(
        'Smith (1923 – 2008)<ref group="nb">A long note. [http://x.org site]</ref> was an engineer.'
        "[[File:Smith.jpg|thumb|A caption]][[Category:Engineers]]"
    )
    _remove_non_prose(code)
    assert code.strip_code().strip() == "Smith (1923 – 2008) was an engineer."


def test_headings_lose_their_markup() -> None:
    wikitext = 'x\n== <span class="anchor" id="a"></span> Early years ==\ny\n=== With [[Wings (band)|Wings]] ===\nz\n'
    assert [title for _, title, _ in split_headings(wikitext)] == ["Introduction", "Early years", "With Wings"]


WIKITEXT = """Intro text.
== Background ==
Background text.
== Recording ==
=== Take 7 ===
Take text.
== The Beatles' version ==
Version text.
== References ==
=== Books ===
Book list.
"""


def test_headings_become_paths_and_dropped_headings_take_their_subsections() -> None:
    sections = section_paths(split_headings(WIKITEXT), WIKIPEDIA, TOPICS)
    paths = [path for _, path, _ in sections]
    assert paths == ["Introduction", "Background", "Recording", "Recording > Take 7", "The Beatles' version"]


def test_keep_sections_selects_by_path_and_judges_each_section_on_its_own() -> None:
    sections = section_paths(split_headings(WIKITEXT), WIKIPEDIA, TOPICS)
    kept = select_sections(sections, ("Introduction", "Take 7", "Beatles"), "Test")
    assert [path for _, path, _ in kept] == ["Introduction", "Recording > Take 7", "The Beatles' version"]


def test_keep_sections_fails_on_an_entry_that_matches_nothing() -> None:
    sections = section_paths(split_headings(WIKITEXT), WIKIPEDIA, TOPICS)
    with pytest.raises(ValueError, match="match no section"):
        select_sections(sections, ("Introduction", "Mixing"), "Test")


def test_keep_sections_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="is empty"):
        select_sections(section_paths(split_headings(WIKITEXT), WIKIPEDIA, TOPICS), (), "Test")


def test_sections_without_text_are_dropped_and_offsets_slice_the_text() -> None:
    tokenizer = tiktoken.get_encoding(CONFIG.chunking.tokenizer)
    sections = section_paths(split_headings(WIKITEXT), WIKIPEDIA, TOPICS)
    built, full_text = build_sections(sections, "wp_1", TOPICS, tokenizer)
    assert [section.title for section in built] == ["Introduction", "Background", "Take 7", "The Beatles' version"]
    assert [section.section_id for section in built] == [f"wp_1__s{i}" for i in range(4)]
    assert built[0].topics == ["introduction"]
    assert built[2].topics == ["recording"]   # "Take 7" inherits from "Recording"
    for section in built:
        assert full_text[section.char_start:section.char_end] == section.text


def _write_raw(raw_dir: Path, pageid: int, title: str) -> None:
    (raw_dir / f"{pageid}.json").write_text(json.dumps({"pageid": pageid, "title": title, "raw": {}}), encoding="utf-8")


def test_a_raw_file_not_in_seeds_is_refused(tmp_path: Path) -> None:
    _write_raw(tmp_path, 1, "Listed")
    _write_raw(tmp_path, 2, "Norman Smith")
    with pytest.raises(ValueError, match="not in seeds.yaml"):
        read_raw_records(tmp_path, {"Listed": Seed("Listed", "song")})


def test_a_seed_without_a_raw_file_is_refused(tmp_path: Path) -> None:
    _write_raw(tmp_path, 1, "Listed")
    seeds = {"Listed": Seed("Listed", "song"), "Not fetched": Seed("Not fetched", "song")}
    with pytest.raises(ValueError, match="run `python -m src.ingest.fetch`"):
        read_raw_records(tmp_path, seeds)


def test_seeds_refuse_a_duplicate_title_and_a_song_album_with_no_article(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "song:\n  Help!: [Help! (song)]\nalbum: [Help!]\noverview: [Help! (song)]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="listed twice"):
        load_seeds(duplicate)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("song:\n  Help:  [Help! (song)]\nalbum: [Help!]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no album article: \\['Help'\\]"):
        load_seeds(unknown)


def test_a_song_takes_the_album_it_is_listed_under(tmp_path: Path) -> None:
    path = tmp_path / "seeds.yaml"
    path.write_text(
        "song:\n"
        "  Revolver:\n"
        "    - Doctor Robert\n"
        "    - title: Got to Get You into My Life\n"
        "      keep_sections: [Introduction]\n"
        "  Single/Outtake: [Hey Jude]\n"
        "album: [Revolver (Beatles album)]\n",
        encoding="utf-8",
    )
    seeds = load_seeds(path)
    assert seeds["Doctor Robert"].album == "Revolver"
    assert seeds["Got to Get You into My Life"].album == "Revolver"
    assert seeds["Hey Jude"].album == NON_ALBUM
    assert seeds["Revolver (Beatles album)"].album is None


def test_the_real_seeds_load() -> None:
    seeds = load_seeds()
    assert seeds["Let It Be (1970 film)"].doc_type == "overview"
    assert seeds["Maggie May (folk song)"].name == "Maggie Mae"
    # Songs whose infobox names no album, or another release, before seeds held the album.
    assert seeds["Doctor Robert"].album == "Revolver"
    assert seeds["I Am the Walrus"].album == "Magical Mystery Tour"
    assert seeds["Twist and Shout"].album == "Please Please Me"
    assert seeds["Real Love (Beatles song)"].album == "Anthology 2"
    assert seeds["Hey Jude"].album == NON_ALBUM
