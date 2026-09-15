# Phase 10E: contextual compression results

This is a post-retrieval experiment. Every condition uses the same frozen top-five legacy chunks in the same order. Evidence retention is measured with verified supporting-passage proxies; it is not answer-generation quality.

Questions: 100 (90 answerable; 10 unanswerable)

| Condition | Mean ratio | p50 | p95 | Evidence term retention | Query-term retention | Mean compression latency | Omitted sources | Provenance failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000s | 0 | 0 |
| per_chunk_75pct | 0.6603 | 0.6653 | 0.7062 | 0.9619 | 0.9579 | 0.064s | 0 | 0 |
| per_chunk_50pct | 0.4202 | 0.4220 | 0.4894 | 0.8939 | 0.9117 | 0.064s | 0 | 0 |
| per_chunk_30pct | 0.2946 | 0.2808 | 0.3837 | 0.8569 | 0.8794 | 0.064s | 0 | 0 |
| global_75pct | 0.7359 | 0.7368 | 0.7476 | 0.9768 | 0.9776 | 0.064s | 0 | 0 |
| global_50pct | 0.4879 | 0.4900 | 0.4976 | 0.9524 | 0.9271 | 0.064s | 9 | 0 |
| global_30pct | 0.2862 | 0.2878 | 0.2976 | 0.8369 | 0.8610 | 0.064s | 95 | 0 |

## Interpretation boundary

These results quantify size, latency, and evidence-retention proxies only. Select a candidate after inspecting per-question failures, then compare it with uncompressed context in the separately controlled, small generation diagnostic.

## Analysis

Global 75% is the safest configuration in this matrix. It removed 133,650 of 506,319 source characters (26.4%), retained 97.7% of the evidence terms observable in baseline-retrieved expected pages, preserved 25 of 26 exactly contained supporting passages, omitted no sources, needed no minimum-evidence safeguards, and added about 64 ms per query. Approximately 63 ms of that cost was batched cross-encoder scoring; segmentation and reconstruction were sub-millisecond.

Global 50% is a more attractive size reduction in aggregate—51.2% of source text removed with 95.2% mean conditional evidence-term retention—but the mean hides unsafe tails. Individual retention reached 41%, and nine of 500 retrieved sources were omitted. Per-chunk 50% retained every source but reduced the mean evidence proxy to 89.4% and triggered 28 safeguards. Both 30% conditions caused clear evidence loss; global 30% omitted 95 sources and lost all measurable evidence terms for three questions.

Cross-page questions were the weakest category for global 75% (57.4% mean conditional retention across the two measurable cases), followed by exact terminology (96.4%). Difficult distractor, comparison, multi-source, and section-detail questions retained essentially all measurable evidence on average. These category counts are small and must not be treated as population estimates.

Only 76 questions had non-zero expected evidence observable in the retrieved document/page context, and only 26 supporting passages were exact normalized substrings of baseline context. This is why conditional term retention is the primary proxy and why it cannot replace generation inspection. A missing passage can reflect retrieval failure, chunk boundaries, or PDF normalization—not compressor behavior.

Global 75% was advanced to the small controlled generation diagnostic. This selection is an experimental comparison decision, not a production-default change.
