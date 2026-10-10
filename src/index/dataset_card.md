---
license: cc-by-sa-4.0
language:
  - en
pretty_name: Beatles Song Detective search index
tags:
  - rag
  - qdrant
  - wikipedia
  - beatles
---

# Beatles Song Detective: search index

The search index behind the Beatles Song Detective Space: a local
[Qdrant](https://qdrant.tech/) store with one collection,
`corpus_v1__section_t256_o0__nomic`, that the Space downloads at startup.

## What it contains

- **326 English Wikipedia articles** about the Beatles' songs, their
  albums, the people who made them, the studios, equipment and recording
  techniques, frozen at fixed revisions in September 2026. Each chunk keeps its
  article title, revision id and URL, so every passage can be traced to the
  exact page it was taken from.
- **Chunks of up to 256 tokens**, cut within sections (never across two), each
  with a short header naming its article and section (and, for a song, its
  album).
- **Two vectors per chunk**, for hybrid search: a dense one from
  `nomic-ai/nomic-embed-text-v1.5` and a sparse BM25 one (`Qdrant/bm25`).

It is a copy of the exact collection the evaluation was measured on, made by
`src/index/publish.py`, not a rebuild.

## How to open it

```python
from qdrant_client import QdrantClient

client = QdrantClient(path="qdrant_store")  # a folder it can write its .lock into
print(client.count("corpus_v1__section_t256_o0__nomic"))
```

## Licence and attribution

The text comes from English Wikipedia and is licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/)
(CC BY-SA 4.0). The authors are the Wikipedia contributors to each article;
the article titles, revision ids and URLs stored with every chunk identify them.
The text was cleaned (markup and some sections removed) and split into chunks;
it was not otherwise edited.

The code that built this index is MIT-licensed and lives with the app.
