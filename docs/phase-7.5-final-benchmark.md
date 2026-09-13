# Phase 7.5: final four-strategy retrieval benchmark

## Outcome

The 52-paper, 100-question experiment compares only the chunking strategy. All
four runs use the same local retrieval stack and labels:

- Qdrant exact cosine search with
  `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (384 dimensions)
- BM25 with the existing defaults
- 20 candidates from each base retriever
- reciprocal rank fusion with `rrf_k=60`
- `cross-encoder/ms-marco-MiniLM-L6-v2`, maximum pair length 512
- reranking 20 fused candidates to top 5
- 90 retrieval-scored questions and 10 unanswerable questions excluded from
  retrieval success metrics

No evaluation label, model, candidate depth, fusion constant, or retriever
parameter changed. No generation was run.

| Strategy | Chunks | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | Mean first-correct rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 3,793 | **45.6%** | 72.2% | **84.4%** | **45.0%** | 71.7% | **82.2%** | 1.895 |
| boundary | 3,510 | 42.2% | 63.3% | **84.4%** | 41.7% | 61.7% | **82.2%** | 2.158 |
| structural | 2,939 | 44.4% | 73.3% | 82.2% | 43.9% | 71.1% | 80.6% | 1.824 |
| semantic | 5,124 | 40.0% | **75.6%** | **84.4%** | 38.3% | **73.3%** | **82.2%** | **1.816** |

Mean first-correct rank is averaged only over questions that succeed by rank 5.
It must be read with Hit@5: a strategy can have a low mean by entirely missing
more hard questions. The metric does not make semantic the overall winner.

## Category breakdown

Each cell lists the three cutoffs as `@1 / @3 / @5`. Recall differs from Hit
mainly for multi-source and cross-page records because those questions have more
than one expected source.

### Legacy

| Category | n | Hit@1/3/5 | Recall@1/3/5 | Mean rank |
|---|---:|---:|---:|---:|
| comparison | 3 | 33.3 / 66.7 / 66.7 | 33.3 / 66.7 / 66.7 | 1.50 |
| cross-page context | 3 | 33.3 / 33.3 / 66.7 | 16.7 / 33.3 / 50.0 | 3.00 |
| difficult distractor | 19 | 36.8 / 63.2 / 84.2 | 36.8 / 63.2 / 84.2 | 2.19 |
| exact number | 23 | 52.2 / 82.6 / 82.6 | 52.2 / 82.6 / 82.6 | 1.47 |
| exact terminology | 13 | 46.2 / 84.6 / 100.0 | 46.2 / 84.6 / 100.0 | 2.00 |
| multi-source | 5 | 0.0 / 20.0 / 60.0 | 0.0 / 10.0 / 30.0 | 3.33 |
| section detail | 13 | 53.8 / 76.9 / 84.6 | 53.8 / 76.9 / 84.6 | 1.82 |
| semantic paraphrase | 11 | 63.6 / 81.8 / 90.9 | 63.6 / 81.8 / 90.9 | 1.60 |

### Boundary

| Category | n | Hit@1/3/5 | Recall@1/3/5 | Mean rank |
|---|---:|---:|---:|---:|
| comparison | 3 | 66.7 / 66.7 / 66.7 | 66.7 / 66.7 / 66.7 | 1.00 |
| cross-page context | 3 | 0.0 / 66.7 / 66.7 | 0.0 / 33.3 / 33.3 | 2.50 |
| difficult distractor | 19 | 31.6 / 68.4 / 78.9 | 31.6 / 68.4 / 78.9 | 2.13 |
| exact number | 23 | 39.1 / 60.9 / 82.6 | 39.1 / 60.9 / 82.6 | 2.21 |
| exact terminology | 13 | 53.8 / 61.5 / 84.6 | 53.8 / 61.5 / 84.6 | 2.00 |
| multi-source | 5 | 20.0 / 20.0 / 60.0 | 10.0 / 10.0 / 40.0 | 3.00 |
| section detail | 13 | 46.2 / 61.5 / 100.0 | 46.2 / 61.5 / 100.0 | 2.46 |
| semantic paraphrase | 11 | 63.6 / 81.8 / 100.0 | 63.6 / 81.8 / 100.0 | 1.82 |

### Structural

| Category | n | Hit@1/3/5 | Recall@1/3/5 | Mean rank |
|---|---:|---:|---:|---:|
| comparison | 3 | 66.7 / 66.7 / 66.7 | 66.7 / 66.7 / 66.7 | 1.00 |
| cross-page context | 3 | 0.0 / 66.7 / 100.0 | 0.0 / 33.3 / 66.7 | 2.67 |
| difficult distractor | 19 | 31.6 / 73.7 / 94.7 | 31.6 / 73.7 / 94.7 | 2.44 |
| exact number | 23 | 56.5 / 78.3 / 78.3 | 56.5 / 78.3 / 78.3 | 1.33 |
| exact terminology | 13 | 61.5 / 84.6 / 92.3 | 61.5 / 84.6 / 92.3 | 1.50 |
| multi-source | 5 | 20.0 / 40.0 / 40.0 | 10.0 / 20.0 / 30.0 | 1.50 |
| section detail | 13 | 30.8 / 61.5 / 76.9 | 30.8 / 61.5 / 76.9 | 2.40 |
| semantic paraphrase | 11 | 54.5 / 81.8 / 81.8 | 54.5 / 81.8 / 81.8 | 1.33 |

### Semantic

| Category | n | Hit@1/3/5 | Recall@1/3/5 | Mean rank |
|---|---:|---:|---:|---:|
| comparison | 3 | 33.3 / 66.7 / 66.7 | 33.3 / 66.7 / 66.7 | 1.50 |
| cross-page context | 3 | 66.7 / 66.7 / 66.7 | 33.3 / 33.3 / 33.3 | 1.00 |
| difficult distractor | 19 | 21.1 / 57.9 / 78.9 | 21.1 / 57.9 / 78.9 | 2.27 |
| exact number | 23 | 43.5 / 82.6 / 87.0 | 43.5 / 82.6 / 87.0 | 1.75 |
| exact terminology | 13 | 53.8 / 92.3 / 100.0 | 53.8 / 92.3 / 100.0 | 1.85 |
| multi-source | 5 | 20.0 / 40.0 / 40.0 | 10.0 / 20.0 / 20.0 | 1.50 |
| section detail | 13 | 38.5 / 84.6 / 92.3 | 38.5 / 84.6 / 92.3 | 1.75 |
| semantic paraphrase | 11 | 54.5 / 81.8 / 90.9 | 54.5 / 81.8 / 90.9 | 1.60 |

These small category slices are diagnostic, not independent leaderboards. For
example, cross-page contains only three questions.

## Meaningful rank changes from legacy

Boundary improved 21 questions, degraded 23, and left 46 unchanged. Five legacy
misses became hits, but five legacy hits became misses. Informative examples are
`p75_rankgpt_noveleval` (miss to rank 1), `p75_retro_database` (miss to rank 3),
`p75_deepimpact_token_value` (rank 1 to miss), and `p75_replug_results` (rank 1
to miss).

Structural improved 27, degraded 21, and left 42 unchanged. It recovered six
legacy misses but lost eight legacy hits. It moved `p75_replug_integration` from
a miss to rank 1, `p75_dpr_dual_encoder` from a miss to rank 2, and the RAG versus
REPLUG comparison from rank 4 to rank 1. Conversely, `p75_ance_efficiency` and
`p75_colbert_late_interaction` moved from rank 1 to misses.

Semantic improved 20, degraded 26, and left 44 unchanged. Six misses became hits
and six hits became misses. It moved `p75_gtr_data_efficiency` from a miss to
rank 1, `p75_cocondenser_loss` from rank 5 to rank 1, and
`p75_raptor_levels` from rank 5 to rank 1. It lost
`p75_colbert_late_interaction`, `p75_late_chunking_training`, and
`p75_rocketqav2_distillation` from ranks 1, 3, and 3 respectively to misses.

Many failures still retrieve the correct document but the wrong page. The test
therefore exposes evidence localization, not merely paper-level discovery.

## Cost and storage

Times are one practical CPU run on the development Mac, not a calibrated
performance benchmark. The embedding model was loaded once (4.35 seconds) and
reused. Query and reranker startup was also shared across evaluations.

| Strategy | Mean chunk chars | Total chunk chars | Embed | Qdrant build | Vector index | Qdrant store | Evaluate | Of which reranking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy | 1,006 | 3,816,467 | 64.4s | 2.56s | 5.92 MB | 19.84 MB | 31.1s | 28.4s |
| boundary | 1,010 | 3,544,135 | 63.2s | 2.26s | 5.54 MB | 18.50 MB | 31.7s | 29.1s |
| structural | 1,100 | 3,232,438 | 58.9s | 1.94s | 4.65 MB | 16.09 MB | 34.5s | 32.0s |
| semantic | 563 | 2,885,197 | **47.0s** | 3.37s | 8.09 MB | 24.73 MB | **28.9s** | 25.8s |

Semantic has the most chunks but the least total chunk text and much shorter
chunks, so its embedding run was fastest in this measurement. It still consumes
the most vector and Qdrant storage. Reranking dominates evaluation time; the
non-reranking portion was only 2.5–3.1 seconds per 100 questions. Timing variation
is too small and uncontrolled to claim a strategy is intrinsically faster.

## Interpretation and recommendation

Legacy remains the recommended default. It has the best Hit@1, ties the best
Hit@5 and Recall@5, avoids semantic's roughly 25% larger Qdrant store, and has the most
stable trade-off across categories. Boundary reaches the same Hit@5 but delays
correct evidence and loses substantial Hit@3. Semantic's smaller units help
exact terminology and section detail by rank 3, but do not justify its lower
Hit@1, weaker difficult-distractor results, and larger index as a general default.

Structural is the most interesting alternative. It has the smallest index,
nearly matches the overall leaders, performs strongly on exact terms and hard
distractors, and is the only strategy with 100% cross-page Hit@5. Its cross-page
Recall@5 of 66.7% also beats legacy's 50.0%, boundary's 33.3%, and semantic's
33.3%. That is evidence that page-spanning structural chunks help the behavior
they were designed for, although three questions are far too few for a broad
claim. Structural's weaker section-detail and overall Hit@5 prevent making it
the default yet.

Compared with the original three-paper benchmark, the expanded corpus is much
harder. Legacy fell from 75.0/100/100% Hit@1/3/5 to 45.6/72.2/84.4%. The old
result made legacy look uniformly dominant; the expanded benchmark reveals
category-specific wins for semantic and structural chunking and much weaker
multi-source behavior across every strategy. This is the central Phase 7.5
lesson: corpus scale and deliberate semantic overlap change the conclusions,
and chunking quality must be evaluated with the downstream retrieval stack.

No compatibility bug was found and no application, chunking, retrieval, or
evaluation code changed. Full local reports remain under the ignored
`data/evaluation/results/phase-7.5/` directory. A compact, source-controlled
machine-readable result is in
`data/evaluation/benchmarks/phase-7.5-results.json`.
