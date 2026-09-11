# Phase 1 architecture

```text
data/papers/*.pdf
        |
        v
PDF discovery (pdf.py)
        |
        v
PageText(document, page_number, text)
        |
        v
normalization + boundary-aware chunking (chunking.py)
        |
        v
Chunk(text, document, page_number, offsets, IDs)
        |
        v
data/processed/chunks.jsonl
```

The page is the metadata boundary in Phase 1. A chunk never crosses pages, so a
page citation remains unambiguous. The tradeoff is that content around a page
break cannot share a chunk; a later phase can evaluate whether multi-page
sections are worth more complex provenance.

Chunking uses characters rather than model-specific tokens. That makes the
algorithm inspectable before a particular embedding model is chosen. It aims for
`chunk_size`, prefers paragraph or sentence endings, and repeats `overlap`
characters of context. Exact chunk quality is empirical: retrieval evaluation,
not convention, should eventually determine these settings.

JSON Lines keeps the intermediate artifact human-readable and stream-friendly.
Each line is an independent chunk record. It is deliberately not a vector-store
schema; Phase 2 can read these records and add embeddings without re-extracting
private papers.

