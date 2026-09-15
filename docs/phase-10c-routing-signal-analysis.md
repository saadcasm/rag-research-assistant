# Phase 10C-1: baseline routing-signal analysis

## Scope and labels

This offline analysis reuses the saved Phase 10B baseline rankings and MQ2 outcomes. It makes no Ollama calls, does not rerun retrieval, and does not change production routing.

- Scored questions: 90
- MQ2-helpful / unchanged / degraded: 5 / 85 / 0
- Class imbalance: 5 helpful vs 85 unchanged; no degraded examples

## Cheap signals

Signals cover reranker score shape, top-k document concentration, query/protected-term coverage, and hybrid-to-reranked stability. Cross-encoder and RRF scores are treated as relative ranking signals, never probabilities.

The separation column is an AUC-style rank statistic rescaled to 0–1: zero means no ordering difference and one means complete ordering in either direction. It measures in-sample ordering, not predictive confidence.

Separate dense and BM25 rankings were not persisted by Phase 10B, so dense/BM25 agreement is explicitly unavailable in this pass.

## Strongest univariate separation (exploratory)

| Signal | In-sample separation | Direction among helpful cases | Best escalation rate for 100% rescue | Precision |
|---|---:|---|---:|---:|
| `unique_documents_top_k` | 0.445 | higher_when_helpful | 94.4% | 5.9% |
| `protected_term_count` | 0.435 | lower_when_helpful | 61.1% | 9.1% |
| `dominant_document_share` | 0.402 | lower_when_helpful | 93.3% | 6.0% |
| `hybrid_mean_rank_shift` | 0.376 | lower_when_helpful | 63.3% | 8.8% |
| `protected_term_coverage_top_k` | 0.358 | higher_when_helpful | n/a | n/a |
| `hybrid_reranked_document_overlap_at_5` | 0.278 | lower_when_helpful | 87.8% | 6.3% |
| `hybrid_top_chunk_retained` | 0.235 | higher_when_helpful | 77.8% | 7.1% |
| `top_reranker_score` | 0.224 | lower_when_helpful | 68.9% | 8.1% |

## Routing reference points

Always-baseline has 0% escalation, 0% rescue recall, and estimated average latency 0.171s.

Always-MQ2 has 100% escalation, 100% rescue recall, 5.6% routing precision, and estimated average latency 6.640s.

### Best in-sample threshold trade-offs

| Target | Exploratory rule | Escalation | Precision | Missed | Unnecessary | Estimated latency |
|---|---|---:|---:|---:|---:|---:|
| Rescue all 5 | `protected_term_count at_or_below 1.5` | 61.1% | 9.1% | 0 | 50 | 4.124s |
| Rescue 4 of 5 | `hybrid_mean_rank_shift at_or_below 0.775` | 35.6% | 12.5% | 1 | 28 | 2.471s |

The four-of-five rule misses `p75_monot5_reformulation`. Its first twelve false positives are: `p75_atlas_few_shot`, `p75_cpack_resources`, `p75_coil_index`, `p75_colbert_late_interaction`, `p75_contriever_beir`, `p75_doct5query_direction`, `p75_dpr_bm25_margin`, `p75_e5_supervision`, `p75_graphrag_index`, `p75_hnsw_layers`, `p75_hyrr_training_pool`, `p75_late_chunking_order`.

Every threshold in the JSON was swept at observed-value boundaries in both directions. These are exploratory in-sample probes, not calibrated or production-ready rules.

The score gaps and score spread were among the weakest separators. This is a useful negative result: an apparently confident top cross-encoder score does not reliably identify when MQ2 will help.

## Helpful-question examples

- `p75_beir_composition`: not found → 5 — How broad is BEIR in terms of datasets and retrieval architectures evaluated?
- `p75_monot5_reformulation`: 3 → 2 — How does monoT5 cast document ranking as a sequence-to-sequence task?
- `p75_rankgpt_noveleval`: not found → 5 — Why did RankGPT introduce the NovelEval test set?
- `p75_realm_signal`: 4 → 3 — What learning signal allows REALM to pretrain its retriever without supervised retrieval labels?
- `p75_hyde_false_details`: 5 → 4 — How does HyDE limit the effect of false details in its generated hypothetical document?

The four-of-five rule retains the always-MQ2 Hit@3/5 values (0.7333/0.8667) because its missed monoT5 improvement changes rank 3 to rank 2 without crossing those cutoffs. That coincidence should not be mistaken for a generally safe miss.

## 10C-1 conclusion

No cheap signal is selective enough to justify a router yet. Capturing all five improvements still escalates 61.1% of questions. Capturing four requires 35.6% escalation and 28 unnecessary LLM calls for four useful ones. This lowers estimated average latency versus always-MQ2, but routing precision remains only 12.5%.

Phase 10C-2 is worth revisiting only after out-of-sample validation or collection of the missing dense/BM25 agreement signals. Implementing the current fitted threshold would encode benchmark leakage as production logic.

## Limitations and next step

- Only five positive cases exist, so all separation and threshold results are unstable.
- Thresholds are explored and evaluated on the same benchmark, creating selection leakage and overfitting risk.
- Cross-encoder and RRF scores are uncalibrated and must not be interpreted as probabilities.
- Query coverage uses persisted 400-character previews rather than complete chunks.
- Dense/BM25 component agreement was not available in the saved Phase 10B report.

Review whether any simple signal captures four or five helpful cases at a materially lower escalation rate. Even a visually strong rule must be checked with leave-one-positive-out stability and a new question set before Phase 10C-2 implements a router.
