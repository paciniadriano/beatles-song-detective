"""Converts between a Chunk and the LlamaIndex TextNode that gets embedded and
stored in Qdrant. build.py writes nodes and search.py reads them back, so the
two directions live side by side: if they disagree about a key, chunks come
back from the index subtly different from the ones that went in."""

from __future__ import annotations

import uuid

from llama_index.core.schema import MetadataMode, TextNode

from src.schemas import Chunk

# Everything a Chunk carries besides its text and its header. These become
# Qdrant payload (so they can be filtered on) but are kept out of the embedded
# text and the LLM prompt: only the header fields are meant to be read.
CHUNK_KEYS = [
    "chunk_id", "doc_id", "section_id", "doc_type", "topics", "title",
    "section_path", "url", "revid", "char_start", "char_end",
]


def point_id(namespace: str, chunk_id: str) -> str:
    """Qdrant point ids must be UUIDs or unsigned integers, so a readable id
    like "wp_1__s2__c0" is rejected (CLAUDE.md, trap 1). uuid5 is deterministic:
    rebuilding a collection gives every chunk the same point id again. The
    readable `chunk_id` stays in the payload."""
    return str(uuid.uuid5(uuid.UUID(namespace), chunk_id))


def chunk_to_node(chunk: Chunk, namespace: str) -> TextNode:
    """The header fields are the node's embedded metadata; LlamaIndex prepends
    them to the text (MetadataMode.EMBED), which is how a chunk carries its
    header with no custom code."""
    payload = {key: getattr(chunk, key) for key in CHUNK_KEYS}
    return TextNode(
        id_=point_id(namespace, chunk.chunk_id),
        text=chunk.text,
        metadata={**chunk.header, **payload},
        excluded_embed_metadata_keys=CHUNK_KEYS,
        excluded_llm_metadata_keys=CHUNK_KEYS,
    )


def retrieval_text(chunk: Chunk) -> str:
    """The header lines, then the chunk text: exactly what the embedder and
    BM25 read (MetadataMode.EMBED), and what the reranker reads too, so the
    three judge the same text."""
    return TextNode(text=chunk.text, metadata=chunk.header).get_content(MetadataMode.EMBED)


def node_to_chunk(node: TextNode, score: float | None = None) -> Chunk:
    """Whatever is left in the metadata after the payload keys is the header."""
    header = dict(node.metadata)
    payload = {key: header.pop(key) for key in CHUNK_KEYS}
    return Chunk(text=node.get_content(MetadataMode.NONE), header=header, score=score, **payload)
