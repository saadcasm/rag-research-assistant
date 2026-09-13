# Phase 7: Advanced document parsing and chunking

Phase 7 asks a deliberately practical question: which way of turning these PDFs
into evidence improves retrieval? It does not change the Phase 4 labels, BM25,
RRF, cross-encoder reranker, Qdrant, or generation system.

```text
PDF --pypdf--> PageText --ChunkingStrategy--> Chunk + provenance
                                             |
                                             +--> JSONL / BM25 / embeddings / Qdrant
```

## Strategies

`legacy` is the Phase 1 baseline and the default. It chunks each page separately
with character targets and overlap. Its old IDs are retained exactly, allowing a
reproducible benchmark.

`boundary` stays within a page but packs paragraph/sentence units before falling
back to a hard character split. It needs no model. PDF text is imperfect, so a
sentence regex is intentionally a lightweight heuristic rather than an NLP parser.

`structural` may join text across pages and tracks `start_page`, `end_page`, and
a conservative detected section title. It recognizes common research-paper labels
and numbered headings, but pypdf does not expose font/style semantics; labels are
therefore hints, not truth.

`semantic` embeds neighbouring sentences with the already-selected local Sentence
Transformers model. A cosine similarity below `0.35` suggests a topic boundary.
To avoid one-sentence fragments, it merges groups below `max(200, chunk_size / 3)`
characters and hard-splits only groups above the maximum. It is deterministic for
a fixed model/version and corpus, but slower because it embeds sentences during
ingestion. It has no network requirement after the model is cached.

`pypdf` remains the only parser dependency. It supplies reliable one-based page
metadata and keeps the project small; Phase 7 did not demonstrate a concrete need
for another parser.

## Provenance and citations

`Chunk.page_number` remains for Phase 1--6 compatibility and means `start_page`.
New chunks explicitly store `start_page`, `end_page`, `section_title`, and
`chunking_strategy`. A page-spanning source is shown as, for example,
`dense-passage-retrieval.pdf, pages 2–3`, not falsely as one page.

Evaluation labels still name one expected page. A chunk matches when that page is
inside its inclusive page range. This makes the policy explicit without changing
any ground truth. Qdrant stores the added fields in its payload and uses the JSONL
SHA-256 plus schema version, so changing strategy makes an existing collection
stale and requires a rebuild.

## Commands

```bash
rag-research-assistant ingest --chunking-strategy structural \
  --output data/processed/chunks-structural.jsonl
rag-research-assistant inspect --input data/processed/chunks-structural.jsonl \
  --document dense-passage-retrieval.pdf --page 2
rag-research-assistant compare-chunks --document dense-passage-retrieval.pdf --page 2
```

Each alternative chunk file needs its own `embed`, `qdrant-build`, and evaluation
run. Generated chunk files, vector matrices, and Qdrant collections are ignored
by Git.

## Real experiment

Run on the three local papers, unchanged 20-question Phase 4 dataset, and the
strongest established pipeline: local Qdrant dense retrieval + BM25 + RRF +
cross-encoder reranking. Times are approximate wall-clock ingestion times on the
development Mac; semantic includes local sentence embedding.

| strategy | chunks | avg chars | cross-page | sections | ingest | Hit@1 / @3 / @5 | mean rank | recall@1 / @3 / @5 |
|---|---:|---:|---:|---:|---:|---|---:|---|
| legacy | 209 | 1004 | 0 | 0 | 1.8s | 75.0 / 100 / 100% | 1.38 | 59.4 / 90.6 / 90.6% |
| boundary | 196 | 1003 | 0 | 0 | 1.7s | 68.8 / 93.8 / 100% | 1.69 | 56.2 / 84.4 / 90.6% |
| structural | 167 | 1106 | 46 | 167 | 1.7s | 62.5 / 75.0 / 81.2% | 1.38 | 53.1 / 71.9 / 78.1% |
| semantic | 301 | 546 | 51 | 301 | 10.8s | 62.5 / 81.2 / 87.5% | 1.43 | 56.2 / 71.9 / 78.1% |

Legacy is retained as the selected default. First-correct ranks below make every
answerable question visible (`—` means no matching page-span chunk in top five).

| question | legacy | boundary | structural | semantic |
|---|---:|---:|---:|---:|
| rag_definition | 1 | 1 | — | 1 |
| rag_memory_types | 1 | 1 | 1 | 2 |
| rag_components | 1 | 3 | 1 | 1 |
| rag_sequence_vs_token | 1 | 1 | 1 | 1 |
| rag_update_knowledge | 1 | 1 | 2 | 2 |
| rag_closed_book | 1 | 1 | 1 | 1 |
| rag_corpus_chunks | 3 | 1 | 2 | 2 |
| rag_more_documents | 1 | 1 | 1 | 1 |
| dpr_runtime | 1 | 3 | 1 | 4 |
| dpr_training_negatives | 2 | 3 | 4 | — |
| dpr_bm25_results | 3 | 3 | — | 1 |
| dpr_lexical_semantic | 1 | 1 | 1 | 1 |
| sgpt_semantic_search | 1 | 1 | 1 | 1 |
| sgpt_cross_bi_encoder | 1 | 1 | 1 | 1 |
| sgpt_asymmetric_search | 1 | 1 | 1 | 1 |
| sgpt_pooling_training | 2 | 4 | — | — |

Boundary improves `rag_corpus_chunks` from rank 3 to 1. Semantic improves
`dpr_bm25_results` from rank 3 to 1, but loses `dpr_training_negatives` entirely.
`dpr_runtime` was already rank 1 with legacy; its structural page 1–2 unit is
qualitatively cleaner, but its score does not improve, and semantic falls to rank
4. Structural and semantic both miss `sgpt_pooling_training`; structural also
misses `rag_definition` and `dpr_bm25_results`.

This is the desired lesson: a more coherent-looking chunk does not necessarily
make a fixed embedding model and reranker retrieve better. Smaller semantic chunks
also increase candidate count and can separate facts the benchmark expects together.
