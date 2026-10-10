"""Turns normalized Documents into chunks, one JSONL file per strategy.

Every strategy runs the same stock LlamaIndex SentenceSplitter. Strategies
differ in exactly two things: `unit` (what one LlamaIndex Document is) and the
chunk size. The splitter never cuts across two Documents, so "respecting
sections" is not a different chunker, it is a different Document.

The header of a chunk (Song, Album, Section, ...) is the Document's metadata:
LlamaIndex prepends it to the text it embeds. Chunking therefore writes no
header text of its own."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import tiktoken
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from src.config import ChunkingConfig, ChunkingStrategy, Config
from src.ingest.normalize import load_documents, load_tokenizer
from src.schemas import Chunk, Document, Section

# Carried in the Document metadata so a transformation can read them, but kept
# out of the embedded text and the LLM prompt: they are not part of the header.
ID_KEYS = ["doc_id", "char_start"]


def header_fields_for(unit: str, doc_type: str, header_fields: dict[str, list[str]]) -> list[str]:
    """The article-unit arm has no Section field: one blind chunk can span
    several sections, so naming one in the header would be wrong."""
    fields = header_fields[doc_type]
    if unit == "article":
        return [field for field in fields if field != "Section"]
    return fields


def header_value(field: str, doc: Document, section: Section | None) -> str:
    """Resolves one header field. Song, Person, Studio... are all the name of
    the thing the article is about; only Album and Section need a lookup. A
    song's album is always set (seeds.yaml lists every song under one)."""
    if field == "Section":
        return section.path if section else ""
    if field == "Album" and doc.doc_type == "song":
        return doc.album or ""
    return doc.entity


def build_header(doc: Document, section: Section | None, fields: list[str]) -> dict[str, str]:
    return {field: header_value(field, doc, section) for field in fields}


def _llama_document(text: str, header: dict[str, str], doc_id: str, char_start: int) -> LlamaDocument:
    return LlamaDocument(
        text=text,
        metadata={**header, "doc_id": doc_id, "char_start": char_start},
        excluded_embed_metadata_keys=ID_KEYS,
        excluded_llm_metadata_keys=ID_KEYS,
    )


def make_llama_documents(
    doc: Document, strategy: ChunkingStrategy, config: ChunkingConfig
) -> list[LlamaDocument]:
    """`unit: article` is one Document holding the whole article; `unit:
    section` is one Document per section, which is what makes it impossible for
    a chunk to mix sections."""
    fields = header_fields_for(strategy.unit, doc.doc_type, config.header_fields)

    if strategy.unit == "article":
        header = build_header(doc, None, fields)
        return [_llama_document(doc.full_text, header, doc.doc_id, 0)]

    return [
        _llama_document(s.text, build_header(doc, s, fields), doc.doc_id, s.char_start)
        for s in doc.sections
    ]


def make_splitter(strategy: ChunkingStrategy, tokenizer: tiktoken.Encoding) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=strategy.size,
        chunk_overlap=strategy.overlap,
        tokenizer=tokenizer.encode,
    )


def section_at(doc: Document, offset: int) -> Section:
    """The section containing a document-level offset: the last one that
    starts at or before it. It is how a blind chunk, which can span several
    sections, gets assigned to the one where it starts."""
    found = doc.sections[0]
    for section in doc.sections:
        if section.char_start > offset:
            break
        found = section
    return found


def split_to_chunk(node: TextNode, doc: Document, section: Section, chunk_id: str) -> Chunk:
    """Adds the Document's `char_start` to the node's offsets. In a section
    arm `node.start_char_idx` is relative to the section, and the evaluation
    compares document-level positions (CLAUDE.md, trap 2)."""
    offset = node.metadata["char_start"]
    header = {key: value for key, value in node.metadata.items() if key not in ID_KEYS}
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc.doc_id,
        section_id=section.section_id,
        doc_type=doc.doc_type,
        topics=section.topics,
        title=doc.title,
        section_path=section.path,
        url=doc.url,
        revid=doc.revid,
        header=header,
        text=node.text,
        char_start=offset + node.start_char_idx,
        char_end=offset + node.end_char_idx,
    )


def chunk_document(
    doc: Document, strategy: ChunkingStrategy, config: ChunkingConfig, splitter: SentenceSplitter
) -> list[Chunk]:
    """Chunks are numbered per owner (the article, or the section), so an id
    like `wp_1__s2__c0` says where the chunk came from. The owner's number
    restarts at 0, which keeps ids stable if another section changes."""
    chunks: list[Chunk] = []
    counts: dict[str, int] = {}
    for llama_doc in make_llama_documents(doc, strategy, config):
        for node in splitter.get_nodes_from_documents([llama_doc]):
            start = node.metadata["char_start"] + node.start_char_idx
            section = section_at(doc, start)
            owner_id = doc.doc_id if strategy.unit == "article" else section.section_id
            index = counts.get(owner_id, 0)
            counts[owner_id] = index + 1
            chunks.append(split_to_chunk(node, doc, section, f"{owner_id}__c{index}"))
    return chunks


def check_chunks(doc: Document, chunks: list[Chunk], expected_fields: list[str]) -> None:
    """Build-time assertions 2 and 5. Both catch bugs that raise nothing and
    quietly invalidate every retrieval number measured afterwards."""
    for chunk in chunks:
        if doc.full_text[chunk.char_start:chunk.char_end] != chunk.text:
            raise AssertionError(f"{chunk.chunk_id}: offsets do not point at the chunk text")
        if list(chunk.header) != expected_fields:
            raise AssertionError(f"{chunk.chunk_id}: header {list(chunk.header)} != {expected_fields}")
        empty = [field for field, value in chunk.header.items() if not value.strip()]
        if empty:
            raise AssertionError(f"{chunk.chunk_id}: empty header fields {empty}")


def chunk_corpus(
    documents: list[Document], strategy: ChunkingStrategy, config: ChunkingConfig, tokenizer: tiktoken.Encoding
) -> list[Chunk]:
    splitter = make_splitter(strategy, tokenizer)
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, strategy, config, splitter)
        fields = header_fields_for(strategy.unit, doc.doc_type, config.header_fields)
        check_chunks(doc, chunks, fields)
        all_chunks.extend(chunks)
    return all_chunks


def chunks_path(config: Config, strategy_id: str) -> Path:
    return config.paths.chunks / config.project.corpus_version / f"{strategy_id}.jsonl"


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


def print_summary(strategy_id: str, chunks: list[Chunk], tokenizer: tiktoken.Encoding) -> None:
    sizes = [len(tokenizer.encode(chunk.text)) for chunk in chunks]
    print(
        f"{strategy_id:20s} {len(chunks):6d} chunks  "
        f"median {statistics.median(sizes):5.0f} tokens  max {max(sizes):5d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the normalized corpus.")
    parser.add_argument("--strategy", help="one strategy id; default: all of them")
    parser.add_argument("--limit", type=int, help="trial run on the first N documents; writes nothing")
    args = parser.parse_args()

    config = Config.load(Path("config.yaml"))
    tokenizer = load_tokenizer(config)

    documents = load_documents(config.paths.docs)
    if args.limit:
        documents = documents[: args.limit]

    for strategy in config.chunking.strategies:
        if args.strategy and strategy.id != args.strategy:
            continue
        chunks = chunk_corpus(documents, strategy, config.chunking, tokenizer)
        print_summary(strategy.id, chunks, tokenizer)
        if not args.limit:
            write_chunks(chunks, chunks_path(config, strategy.id))


if __name__ == "__main__":
    main()
