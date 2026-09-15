# Phase 10E: contextual compression results

This is a post-retrieval experiment. Every condition uses the same frozen top-five legacy chunks in the same order. Evidence retention is measured with verified supporting-passage proxies; it is not answer-generation quality.

Questions: 5 (5 answerable; 0 unanswerable)

| Condition | Mean ratio | p50 | p95 | Evidence term retention | Query-term retention | Mean compression latency | Omitted sources | Provenance failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.000s | 0 | 0 |
| per_chunk_75pct | 0.6497 | 0.6612 | 0.6845 | 0.9900 | 0.9214 | 0.155s | 0 | 0 |
| per_chunk_50pct | 0.4670 | 0.4644 | 0.5421 | 0.9900 | 0.8929 | 0.155s | 0 | 0 |
| per_chunk_30pct | 0.3226 | 0.2708 | 0.4467 | 0.7726 | 0.8595 | 0.155s | 0 | 0 |
| global_75pct | 0.7366 | 0.7362 | 0.7428 | 0.9900 | 1.0000 | 0.155s | 0 | 0 |
| global_50pct | 0.4842 | 0.4854 | 0.4883 | 0.9900 | 0.8992 | 0.155s | 2 | 0 |
| global_30pct | 0.2804 | 0.2798 | 0.2970 | 0.7835 | 0.8373 | 0.155s | 5 | 0 |

## Interpretation boundary

These results quantify size, latency, and evidence-retention proxies only. Select a candidate after inspecting per-question failures, then compare it with uncompressed context in the separately controlled, small generation diagnostic.
