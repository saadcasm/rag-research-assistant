# Phase 1 and Phase 2A architecture

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
        |
        v
local embedding model (embeddings.py)
        |
        v
embeddings.npy + integrity manifest (index.py)
        |
        +------------------------------+
                                       |
natural-language query                 |
        |                              |
        v                              v
same local embedding model     row-aligned chunk vectors
        |                              |
        +--------------+---------------+
                       v
          NumPy cosine similarity (retrieval.py)
                       |
                       v
          top-k Chunk + score results
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

## Phase 2A component boundaries

`embeddings.py` is the only component that knows about Sentence Transformers.
It exposes separate document and query methods while guaranteeing both use the
same model. Indexing and retrieval depend on that small interface, so unit tests
can use deterministic fake vectors without downloading a model or asserting
fragile real-model scores.

`index.py` writes a dense `float32` NumPy matrix. Matrix row `i` corresponds to
chunk `i` from `chunks.jsonl`. The manifest records that row-to-chunk ID mapping,
the exact model name, vector dimension, and a SHA-256 fingerprint of the source
JSONL. Loading fails if any of those invariants no longer hold. This prevents a
subtle but serious error: showing metadata from one chunk beside another chunk's
vector.

`retrieval.py` performs a brute-force scan. It normalizes the query and every
document vector through the cosine formula, ranks the resulting one-dimensional
score array, and selects the first `k` rows. This costs roughly `O(N × D)` per
query for `N` chunks of dimension `D`. The direct calculation is appropriate for
a small research collection and makes the mechanism observable before an
approximate-nearest-neighbor index is introduced.

The embedding model is deliberately stored in the manifest. A vector coordinate
has meaning only within the model that learned that vector space, so a query from
a different model must never be compared with the persisted document matrix.
