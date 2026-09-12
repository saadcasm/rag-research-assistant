# Phase 6: persistent local vector storage with Qdrant

## Why add a vector database now?

The NumPy implementation remains ideal for learning the dense-retrieval math:
one row per chunk, manual cosine similarity, and a complete scan. It is not a
database. The application owns matrix persistence, row-to-metadata integrity,
filtering, updates, and every future scaling decision.

A vector database stores embeddings as searchable records and couples them with
metadata. Qdrant calls a named group of records a **collection** and each record
a **point**. A point has an ID, vector, and optional JSON **payload**. Phase 6
uses the official Python client's embedded local mode, so data persists on disk
without Docker, a server, cloud account, API key, or network connection.

Qdrant is an alternative dense backend, not a replacement for the Phase 2
implementation. NumPy remains the default and is still the clearest reference.
The project constrains `qdrant-client>=1.19,<2.0`; the measured implementation
and tests used version 1.19.0.

## Collection design

The default collection is `rag_research_chunks`, persisted under
`data/processed/qdrant/`. It contains one unnamed dense vector per point. Vector
dimension is read from the validated NumPy index rather than hardcoded; the
current model produces 384 dimensions. The distance metric is cosine because
the NumPy baseline ranks by cosine similarity.

Each payload contains exactly the data needed to reconstruct `Chunk`:

```text
chunk_id, document, page_number, chunk_index,
char_start, char_end, text
```

Qdrant point IDs accept integers or UUIDs, while existing chunk IDs contain
filenames and punctuation. `point_id_for_chunk` deterministically derives a
UUIDv5 from the logical `chunk_id`. Uploading the same chunk again therefore
updates the same point rather than adding a duplicate. The original chunk ID
remains in payload and is still the application-facing identity.

## Build and safety metadata

`qdrant-build` loads the existing NumPy index and copies its exact vectors into
Qdrant. Reusing those vectors is deliberate: if rankings differ, the difference
must come from storage/search behavior rather than another embedding pass.

Collection metadata records:

- schema version;
- embedding model name;
- embedding dimension;
- chunk count;
- SHA-256 fingerprint of `chunks.jsonl`.

Every open validates this metadata, the live vector configuration, cosine
distance, and exact point count. Query construction also verifies that the
cached query model name and returned vector dimension match the collection.
Payload is type-checked before a `Chunk` is reconstructed.

An existing compatible collection is upserted with the same IDs. An incompatible
collection is never silently replaced; `--recreate` is required. If ingestion
changes `chunks.jsonl`, the fingerprint fails and explains that a rebuild is
necessary.

## Search and filtering

The Qdrant retriever embeds the query with the collection's model, requests the
best points, validates their payloads, and returns ordinary `SearchResult`
objects. BM25, RRF, reranking, evaluation, context construction, and Ollama never
receive a Qdrant-specific type.

`--document filename.pdf` becomes an exact Qdrant payload filter on `document`.
It is limited to Qdrant dense retrieval for now so a hybrid BM25 branch cannot
silently reintroduce chunks from other documents. No payload index is created
for 209 points; a larger corpus should index fields used frequently in filters.

## Exact search now, HNSW later

The embedded Qdrant local engine reports that it performs exact brute-force
search. At 209 vectors this is the transparent, correct choice. It compares the
query with every candidate, like NumPy, rather than trading recall for speed.

Server-scale Qdrant can build an HNSW (Hierarchical Navigable Small World) graph.
HNSW connects each vector to nearby vectors across graph layers, navigates from
coarse to increasingly local neighborhoods, and avoids scanning everything.
That produces fast approximate nearest-neighbor search, with parameters that
trade memory, build time, query time, and recall. Qdrant can also force exact
full-scan queries when correctness checks require them. None of that complexity
benefits this corpus yet.

For cosine collections, Qdrant normalizes vectors on upload and calculates a dot
product at query time. This can create tiny floating-point score differences
from the NumPy formula while preserving rank order.

## Commands

```bash
rag-research-assistant qdrant-build
rag-research-assistant qdrant-info
rag-research-assistant qdrant-build --recreate

rag-research-assistant search "What is RAG?" \
  --retriever dense --dense-backend qdrant

rag-research-assistant evaluate \
  --retriever dense --dense-backend qdrant

rag-research-assistant compare --dense-backend qdrant

rag-research-assistant ask "How can RAG update knowledge?" \
  --retriever hybrid --dense-backend qdrant --rerank --show-context
```

Custom storage and collection names are available through `--qdrant-path` and
`--collection`. NumPy remains the default when `--dense-backend` is omitted.

## Real local verification

The collection was built from the current three-paper corpus and reopened by
separate CLI processes:

| Property | Observed value |
| --- | --- |
| Persisted path | `data/processed/qdrant/` |
| Collection | `rag_research_chunks` |
| Points after first build | 209 |
| Points after idempotent second build | 209 |
| Vector dimension | 384 |
| Distance | Cosine |
| Embedding model | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` |
| Search behavior | Exact local full scan |

The fixed Phase 4 evaluation dataset produced:

| Strategy | Hit@1 | Hit@3 | Hit@5 | Mean first rank | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy dense | 56.2% | 68.8% | 100.0% | 2.12 | 46.9% | 59.4% | 93.8% |
| Qdrant dense | 56.2% | 68.8% | 100.0% | 2.12 | 46.9% | 59.4% | 93.8% |
| Qdrant hybrid + reranker | 75.0% | 100.0% | 100.0% | 1.38 | 59.4% | 90.6% | 90.6% |

NumPy and Qdrant returned the identical ordered top five for all 20 questions.
The maximum corresponding cosine-score difference was approximately `9.3e-8`,
consistent with floating-point normalization. No evaluation label changed.

Approximate cold-command wall times were 6.0 seconds for NumPy dense, 4.7
seconds for Qdrant dense, and 10.5 seconds for Qdrant hybrid + reranker. These
include Python and model startup and are too noisy to establish a speed winner.
At this scale Qdrant overhead can easily dominate; its purpose here is the
storage abstraction and growth path.

A grounded Qdrant hybrid+reranked smoke query completed with local
`qwen3.5:4b`, `think=false`, five displayed sources, and valid references `[3]`
and `[5]`. Generation did not know which dense backend supplied the chunks.

## What this phase demonstrates

Changing storage should not improve retrieval quality when vectors, metric, and
search semantics remain the same. The identical rankings are therefore the
desired outcome. Phase 4 and Phase 5 evaluation protected the system from a
silent infrastructure regression; metadata checks protect it from a stale or
misaligned collection. A production vector database becomes valuable as update,
filtering, persistence, and scale requirements grow—not merely because it has a
vector-database label.
