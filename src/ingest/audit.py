"""Prints the report to read once after normalizing, before anything is built on
the corpus: what it contains, how its sections are labelled, how their sizes
compare with the chunk sizes, and the signals of a wrong article or of text
lost in conversion. It flags; it never changes or drops anything.

    python -m src.ingest.audit"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import mwparserfromhell

from src.config import Config
from src.ingest.normalize import load_documents, mentions_band, template_text, wikitext_of
from src.schemas import Document

TOP_N = 30
# "within , rather": a word followed by a space and punctuation is where a
# deleted template left a hole in a sentence.
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\w [,.;:](?=\s)")


def topic_table(documents: list[Document]) -> tuple[list[str], dict[str, Counter]]:
    """Section counts per doc_type and topic (a section with two topics counts
    in both), for a look at whether the topics make sense for each kind of
    article."""
    table: dict[str, Counter] = {}
    for doc in documents:
        counts = table.setdefault(doc.doc_type, Counter())
        for section in doc.sections:
            counts.update(section.topics)
    columns = sorted({topic for counts in table.values() for topic in counts})
    return columns, table


def topic_combinations(documents: list[Document], n: int) -> list[tuple[str, int]]:
    """The most frequent multi-topic sections: where mixed headings land."""
    counts: Counter[str] = Counter(
        " + ".join(section.topics) for doc in documents for section in doc.sections if len(section.topics) > 1
    )
    return counts.most_common(n)


def general_titles(documents: list[Document], fallback: str, n: int) -> list[tuple[str, int]]:
    """The headings that fell to the fallback, most frequent first: where a
    missing keyword would show up."""
    counts: Counter[str] = Counter(
        section.title for doc in documents for section in doc.sections if section.topics == [fallback]
    )
    return counts.most_common(n)


def share_over(sizes: list[int], limit: int) -> float:
    return 100 * sum(1 for size in sizes if size > limit) / len(sizes) if sizes else 0.0


def wrong_article_signals(documents: list[Document], terms: list[str], short_max: int) -> dict[str, list[str]]:
    """Three signs that a document may not be the article meant: it never names
    the band, it is a song whose introduction does not, or it is very short."""
    return {
        "never names the band": [doc.title for doc in documents if not mentions_band(doc.full_text, terms)],
        "song introduction without the band": [
            doc.title for doc in documents
            if doc.doc_type == "song" and doc.sections and not mentions_band(doc.sections[0].text, terms)
        ],
        f"at most {short_max} tokens": [
            f"{doc.title} ({doc.stats.token_count})" for doc in documents if doc.stats.token_count <= short_max
        ],
    }


def conversion_holes(documents: list[Document]) -> list[str]:
    """One entry per hole a deleted template left in a sentence, by document."""
    return [doc.title for doc in documents for _ in SPACE_BEFORE_PUNCTUATION_RE.finditer(doc.full_text)]


def deleted_templates(raw_dir: Path, n: int) -> list[tuple[str, int]]:
    """The templates normalize.py deletes, most frequent first. Reading the list
    once shows whether any of them carries text that should have been kept."""
    counts: Counter[str] = Counter()
    for path in raw_dir.glob("*.json"):
        wikitext = wikitext_of(json.loads(path.read_text(encoding="utf-8")))
        for template in mwparserfromhell.parse(wikitext).filter_templates(recursive=True):
            if not template_text(template):
                counts[template.name.strip().lower()] += 1
    return counts.most_common(n)


def _print_list(label: str, items: list[str]) -> None:
    counts = Counter(items)
    print(f"  {label}: {len(items)}")
    for item, count in counts.most_common(15):
        print(f"      {count:3d}  {item}" if count > 1 else f"           {item}")


def print_report(documents: list[Document], config: Config) -> None:
    sections = [section for doc in documents for section in doc.sections]
    print(f"{len(documents)} documents, {len(sections)} sections\n")

    print("Documents per doc_type:")
    for doc_type, count in Counter(doc.doc_type for doc in documents).most_common():
        print(f"  {doc_type:10s} {count:4d}")

    columns, table = topic_table(documents)
    print("\nSections per doc_type and topic:")
    print(" " * 12 + "".join(f"{column[:11]:>12s}" for column in columns))
    for doc_type, counts in sorted(table.items()):
        print(f"  {doc_type:10s}" + "".join(f"{counts.get(column, 0):12d}" for column in columns))

    print("\nMost frequent topic combinations:")
    for combination, count in topic_combinations(documents, TOP_N):
        print(f"  {count:4d}  {combination}")

    print(f"\nTop {TOP_N} headings in '{config.topics.fallback}':")
    for title, count in general_titles(documents, config.topics.fallback, TOP_N):
        print(f"  {count:4d}  {title}")

    short = [section for section in sections if section.token_count <= config.audit.short_section_max_tokens]
    print(f"\nSections of at most {config.audit.short_section_max_tokens} tokens: {len(short)}")

    sizes = [section.token_count for section in sections]
    print("\nSections longer than each chunk size (the ones that strategy splits):")
    for strategy in config.chunking.strategies:
        print(f"  > {strategy.size:5d} tokens ({strategy.id}): {share_over(sizes, strategy.size):5.1f}%")

    print("\nSigns of a wrong article:")
    signals = wrong_article_signals(documents, config.eval.generate.mention_terms, config.audit.short_document_max_tokens)
    for label, items in signals.items():
        _print_list(label, items)

    print("\nSigns of text lost in conversion:")
    _print_list("holes left by deleted templates", conversion_holes(documents))

    print(f"\nMost frequent deleted templates (top {TOP_N}):")
    for name, count in deleted_templates(config.paths.raw, TOP_N):
        print(f"  {count:5d}  {name}")


def main() -> None:
    config = Config.load(Path("config.yaml"))
    print_report(load_documents(config.paths.docs), config)


if __name__ == "__main__":
    main()
