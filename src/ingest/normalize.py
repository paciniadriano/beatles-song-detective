"""Turns each raw Wikipedia API response (data/raw/wikipedia/{pageid}.json)
into one normalized Document (data/docs/{doc_id}.json): wikitext reduced to
plain text, split into sections at == headings ==, each section given its
topics, and full_text assembled with document-level char offsets. No
chunking decisions happen here -- that is index/chunking.py's job.

What each article is (its doc_type) and its per-article rules come from
seeds.yaml, looked up by the page title Wikipedia returned.

    python -m src.ingest.normalize"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

import mwparserfromhell
import tiktoken
from mwparserfromhell.nodes import Template
from mwparserfromhell.wikicode import Wikicode
from pydantic import ValidationError

from src.config import INTRODUCTION_TOPIC, Config, TopicsConfig, WikipediaConfig
from src.ingest.seeds import Seed, entity_from_title, load_seeds
from src.schemas import INTRODUCTION_TITLE, PATH_SEPARATOR, Document, DocumentStats, Section

LICENSE = "CC BY-SA 4.0"
SECTION_SEPARATOR = "\n\n"


class Heading(NamedTuple):
    """One heading as split_headings() finds it, with the wikitext under it."""
    level: int
    title: str
    body: str


class RawSection(NamedTuple):
    """A heading that survived the drop lists, with its full path."""
    title: str
    path: str
    body: str

_HEADING_RE = re.compile(r"^(={1,6})\s*(.+?)\s*\1[ \t]*$", re.MULTILINE)
_WIKI_QUOTE_MARKUP_RE = re.compile(r"'{2,}")
_EMPTY_PARENS_RE = re.compile(r"\(\s*[,;]?\s*\)")
_REPEATED_SPACES_RE = re.compile(r"[ \t]{2,}")


# --- templates -----------------------------------------------------------------
# strip_code() deletes every {{template}}. Most are citations or layout and should
# go, but a few carry text a reader needs: sharps and flats, time signatures,
# quotations. Those are replaced by their text first; every other template is
# still deleted.

MUSIC_SYMBOLS = {"flat": "♭", "b": "♭", "sharp": "♯", "#": "♯", "natural": "♮", "n": "♮"}
QUOTE_TEMPLATES = {"quote", "blockquote", "cquote", "quotation", "quote box", "rquote", "quote frame"}
# Templates whose first positional argument is the text itself.
TEXT_WRAPPERS = {"nowrap", "nobr", "small", "big", "em", "strong", "mono", "abbr", "sic", "not a typo", "smallcaps"}
# Templates whose second positional argument is the text ({{lang|fr|...}}).
SECOND_ARG_WRAPPERS = {"lang", "color", "colour"}
CHARACTERS = {
    "'s": "'s", "'": "'", "nbsp": " ", "snd": " – ", "spaced ndash": " – ",
    "ndash": "–", "mdash": "—", "!": "|",
}


def _positional(template: Template) -> list[str]:
    return [str(param.value).strip() for param in template.params if not param.showkey]


def template_text(template: Template) -> str:
    """The wikitext that should stand in for one template, "" to delete it."""
    name = template.name.strip().lower()
    args = _positional(template)
    if name in CHARACTERS:
        return CHARACTERS[name]
    if name in MUSIC_SYMBOLS:  # {{flat}}, {{sharp}}, {{natural}}
        return MUSIC_SYMBOLS[name]
    if name == "music" and args:  # {{music|flat}}, {{music|time|4|4}}
        if args[0] == "time" and len(args) >= 3:
            return f"{args[1]}/{args[2]}"
        return MUSIC_SYMBOLS.get(args[0].lower(), "")
    if name in QUOTE_TEMPLATES:
        for key in ("quote", "text"):
            if template.has(key):
                return str(template.get(key).value).strip()
        return args[0] if args else ""
    if name in TEXT_WRAPPERS:
        return args[0] if args else ""
    if name in SECOND_ARG_WRAPPERS:
        return args[1] if len(args) > 1 else ""
    if name == "convert" and len(args) >= 2:  # {{convert|15|in|mm}} -> "15 in"
        if len(args) >= 4 and args[1] in {"-", "–", "to", "and"}:
            return f"{args[0]}–{args[2]} {args[3]}"
        return f"{args[0]} {args[1]}"
    return ""


def unwrap_templates(wikicode: Wikicode) -> None:
    """Replaces every template in place by template_text(). Inner templates go
    first (reverse document order), so an outer one reads text, not markup."""
    for template in reversed(wikicode.filter_templates(recursive=True)):
        wikicode.replace(template, template_text(template))


# --- wikitext ------------------------------------------------------------------

def _remove_non_prose(wikicode: Wikicode) -> None:
    """Removes what is not the article's prose: footnotes (<ref> tags, whose
    text strip_code() would otherwise leave in the middle of a sentence),
    category links and images with their captions."""
    for tag in wikicode.filter_tags(recursive=True, matches=lambda node: node.tag.strip().lower() == "ref"):
        try:
            wikicode.remove(tag)
        except ValueError:  # already removed along with an enclosing tag
            pass
    for link in wikicode.filter_wikilinks():
        title = str(link.title).strip().lower()
        if title.startswith(("category:", "file:", "image:")):
            wikicode.remove(link)


def _clean_title(title: str) -> str:
    """A heading as plain text: links, HTML anchors and italic/bold markup
    removed ("''Love'' remix" -> "Love remix", "With [[Wings (band)|Wings]]" ->
    "With Wings")."""
    plain = mwparserfromhell.parse(title).strip_code()
    return _WIKI_QUOTE_MARKUP_RE.sub("", plain).strip()


def _plain_text(body_wikitext: str) -> str:
    """A section's body as plain text, without the gaps that deleted templates
    leave behind: empty parentheses and runs of spaces."""
    text = mwparserfromhell.parse(body_wikitext).strip_code(collapse=True)
    text = _EMPTY_PARENS_RE.sub("", text)
    text = _REPEATED_SPACES_RE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# --- sections ------------------------------------------------------------------

def split_headings(wikitext: str) -> list[Heading]:
    """Every heading in document order, starting with the text before the
    first heading as a level-1 "Introduction"."""
    matches = list(_HEADING_RE.finditer(wikitext))
    intro_end = matches[0].start() if matches else len(wikitext)
    headings = [Heading(1, INTRODUCTION_TITLE, wikitext[:intro_end])]
    for i, match in enumerate(matches):
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        headings.append(Heading(len(match.group(1)), _clean_title(match.group(2)), wikitext[match.end():body_end]))
    return headings


@functools.lru_cache(maxsize=None)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """A phrase as a whole-word pattern. "(s)" or "(es)" at the end of a word
    makes that ending optional: "origin(s)" is "origin" or "origins", "mix(es)"
    is "mix" or "mixes". Any other parenthesis is literal."""
    body = re.escape(comparable(phrase))
    body = body.replace(r"\(s\)", "(?:s)?").replace(r"\(es\)", "(?:es)?")
    return re.compile(r"(?<!\w)" + body + r"(?!\w)")


def comparable(text: str) -> str:
    """Text as every heading rule compares it: lower case, and a typographic
    apostrophe as a plain one ("The Beatles’ version" = "the beatles' version")."""
    return text.lower().replace("’", "'")


def names_any(text: str, phrases: Iterable[str]) -> bool:
    """True if one of the phrases appears in the text as whole words, ignoring
    case: "notes" is in "Notes and references" but not in "Denotes", and
    "musician" is not in "Musicianship". Every heading rule in config.yaml and
    seeds.yaml is matched this way (the "(s)" notation: _phrase_pattern)."""
    return any(_phrase_pattern(phrase).search(comparable(text)) for phrase in phrases)


def mentions_band(text: str, terms: list[str]) -> bool:
    """True if the text names the band or one of them (config.yaml,
    `mention_terms`). Unlike names_any, a plain case-sensitive search: "Beatle"
    must also match "Beatles", and "Harrison" and "Starr" are names here, which
    a lower-case match would also find in ordinary words."""
    return any(term in text for term in terms)


def heading_topics(title: str, config: TopicsConfig) -> set[str]:
    """Every topic whose phrases the heading names; an exact title gives its
    one topic. Empty when the heading carries no signal ("Take 7")."""
    for exact_title, topic in config.exact_titles.items():
        if comparable(title) == comparable(exact_title):
            return {topic}
    return {topic for topic, phrases in config.keywords.items() if names_any(title, phrases)}


def path_topics(path: str, config: TopicsConfig) -> list[str]:
    """The topics of the heading itself or, if it has none, of its nearest
    ancestor that has some ("Recording > Take 7" is recording)."""
    for title in reversed(path.split(PATH_SEPARATOR)):
        topics = heading_topics(title, config)
        if topics:
            return sorted(topics)
    return [config.fallback]


def is_dropped(title: str, wikipedia: WikipediaConfig, topics: TopicsConfig) -> bool:
    """Private headings always go. Apparatus and lists go unless the heading
    also names an in-scope topic: "Cultural references" is media and stays,
    "Charts and certifications" is only release and goes."""
    if names_any(title, wikipedia.private_sections):
        return True
    in_scope = heading_topics(title, topics) - set(topics.out_of_scope)
    return names_any(title, wikipedia.drop_sections) and not in_scope


def section_paths(
    headings: list[Heading], wikipedia: WikipediaConfig, topics: TopicsConfig
) -> list[RawSection]:
    """Every section not dropped; a dropped heading takes its subsections
    with it. The path is the title alone
    for a top-level heading, and the chain of ancestors once a heading is
    nested two levels deep ("Recording > Take 7")."""
    kept: list[RawSection] = []
    stack: list[tuple[int, str, bool]] = []  # (level, title, dropped)
    for level, title, body in headings:
        while stack and stack[-1][0] >= level:
            stack.pop()
        dropped = (bool(stack) and stack[-1][2]) or is_dropped(title, wikipedia, topics)
        ancestors = [entry[1] for entry in stack if entry[0] > 1]
        path = PATH_SEPARATOR.join(ancestors + [title]) if level > 2 and ancestors else title
        stack.append((level, title, dropped))
        if not dropped:
            kept.append(RawSection(title, path, body))
    return kept


def select_sections(sections: list[RawSection], keep: tuple[str, ...], doc_title: str) -> list[RawSection]:
    """Keeps the sections whose path names one of the `keep` phrases. Each
    section is judged on its own path, so a subsection can be kept under a
    parent that is not. Raises if the list is empty or if a phrase matches no
    section: a renamed heading or a wrong article would otherwise shrink the
    corpus without a word."""
    if not keep:
        raise ValueError(f"{doc_title!r}: keep_sections is empty in seeds.yaml; list the headings to keep")
    unmatched = [phrase for phrase in keep if not any(names_any(section.path, [phrase]) for section in sections)]
    if unmatched:
        raise ValueError(f"{doc_title!r}: keep_sections entries match no section: {unmatched}")
    return [section for section in sections if names_any(section.path, keep)]


def build_sections(
    sections: list[RawSection],
    doc_id: str,
    topics: TopicsConfig,
    tokenizer: tiktoken.Encoding,
) -> tuple[list[Section], str]:
    """Converts each section to plain text, drops the ones left with no text
    (a heading followed only by subsections), and assembles full_text so each
    section's char_start/char_end slice it back out exactly."""
    result: list[Section] = []
    parts: list[str] = []
    cursor = 0
    for title, path, body in sections:
        text = _plain_text(body)
        if not text:
            continue
        section_topics = [INTRODUCTION_TOPIC] if path == INTRODUCTION_TITLE else path_topics(path, topics)
        result.append(Section(
            section_id=f"{doc_id}__s{len(result)}",
            title=title,
            path=path,
            topics=section_topics,
            text=text,
            char_start=cursor,
            char_end=cursor + len(text),
            token_count=len(tokenizer.encode(text)),
        ))
        parts.append(text)
        cursor += len(text) + len(SECTION_SEPARATOR)
    return result, SECTION_SEPARATOR.join(parts)


# --- documents -----------------------------------------------------------------

def _wikipedia_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="():,'!&")


def normalize_document(record: dict, seed: Seed, config: Config, tokenizer: tiktoken.Encoding) -> Document:
    """Builds the Document for one raw record. Raises with the doc_id and the
    cause if it does not hold together."""
    revision = record["raw"]["revisions"][0]
    doc_id = f"wp_{record['pageid']}"

    wikicode = mwparserfromhell.parse(wikitext_of(record))
    _remove_non_prose(wikicode)
    unwrap_templates(wikicode)

    sections = section_paths(split_headings(str(wikicode)), config.wikipedia, config.topics)
    if seed.keep_sections is not None:
        sections = select_sections(sections, seed.keep_sections, seed.title)
    built, full_text = build_sections(sections, doc_id, config.topics, tokenizer)

    for section in built:
        if full_text[section.char_start:section.char_end] != section.text:
            raise AssertionError(f"{doc_id}: char offsets do not match text for {section.section_id!r}")
    try:
        return Document(
            doc_id=doc_id,
            pageid=record["pageid"],
            revid=revision["revid"],
            revised_at=revision["timestamp"],
            title=record["title"],
            url=_wikipedia_url(record["title"]),
            license=LICENSE,
            doc_type=seed.doc_type,
            entity=seed.name or entity_from_title(record["title"]),
            album=seed.album,
            sections=built,
            full_text=full_text,
            stats=DocumentStats(
                section_count=len(built),
                token_count=sum(section.token_count for section in built),
                char_count=len(full_text),
            ),
        )
    except ValidationError as exc:
        raise ValueError(f"{doc_id}: does not validate against the Document schema: {exc}") from exc


def wikitext_of(record: dict) -> str:
    """The article's wikitext inside a raw API record (fetch.py's format)."""
    return record["raw"]["revisions"][0]["slots"]["main"]["*"]


def load_tokenizer(config: Config) -> tiktoken.Encoding:
    """The tokenizer that counts section and chunk tokens. tiktoken downloads
    its files once, into the project's cache folder."""
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(config.paths.tiktoken_cache))
    config.paths.tiktoken_cache.mkdir(parents=True, exist_ok=True)
    return tiktoken.get_encoding(config.chunking.tokenizer)


def read_raw_records(raw_dir: Path, seeds: dict[str, Seed]) -> list[dict]:
    """Every raw record, checked against seeds.yaml in both directions: a raw
    file nobody listed (a page removed from the corpus) must be deleted, and a
    listed page with no raw file must be fetched first."""
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_dir.glob("*.json"))]
    titles = {record["title"] for record in records}
    unlisted = sorted(titles - set(seeds))
    if unlisted:
        raise ValueError(
            f"These raw files are not in seeds.yaml; delete them from {raw_dir}:\n"
            + "\n".join(f"  - {record['pageid']}.json  {record['title']}" for record in records if record["title"] in unlisted)
        )
    missing = sorted(set(seeds) - titles)
    if missing:
        raise ValueError("These seeds have no raw file; run `python -m src.ingest.fetch`:\n"
                         + "\n".join(f"  - {title}" for title in missing))
    return records


def normalize_all(raw_dir: Path, docs_dir: Path, config: Config) -> int:
    """Writes data/docs/{doc_id}.json for every raw file. Fails loudly on the
    first document that does not hold together rather than skipping it."""
    tokenizer = load_tokenizer(config)
    seeds = load_seeds()

    records = read_raw_records(raw_dir, seeds)
    # Documents are derived data: rebuilt whole, so none survives from a page
    # that has left the corpus.
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*.json"):
        stale.unlink()
    for record in records:
        document = normalize_document(record, seeds[record["title"]], config, tokenizer)
        out_path = docs_dir / f"{document.doc_id}.json"
        out_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    return len(records)


def load_documents(docs_dir: Path) -> list[Document]:
    """Reads back every normalized document, validating it against the schema:
    a stale or hand-edited file is caught here, before anything is built on it."""
    return [
        Document.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(docs_dir.glob("*.json"))
    ]


def main() -> None:
    config = Config.load(Path("config.yaml"))
    written = normalize_all(config.paths.raw, config.paths.docs, config)
    print(f"Normalized {written} documents into {config.paths.docs}")


if __name__ == "__main__":
    main()
