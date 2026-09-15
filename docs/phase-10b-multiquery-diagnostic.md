# Phase 10B: multi-query retrieval results

The frozen baseline is compared with original+2 and original+3 generated alternatives. Each query receives dense+BM25 hybrid retrieval; those rankings are fused before one unchanged cross-encoder rerank against the original question.

Dataset: `data/evaluation/questions-phase-7.5.jsonl`
Generation model: `qwen3.5:4b`

## Aggregate retrieval metrics

| Metric | Baseline | Multi-query-2 | Multi-query-3 |
|---|---:|---:|---:|
| Hit@1 | 0.6000 | 0.6000 | 0.6000 |
| Hit@3 | 0.6000 | 0.6000 | 0.6000 |
| Hit@5 | 0.8000 | 1.0000 | 1.0000 |
| Recall@1 | 0.6000 | 0.6000 | 0.6000 |
| Recall@3 | 0.6000 | 0.6000 | 0.6000 |
| Recall@5 | 0.8000 | 1.0000 | 1.0000 |
| Mean first relevant rank | 1.7500 | 2.4000 | 2.4000 |

## Behavior and cost

- Questions: 5 (5 scored; 0 unanswerable)
- MQ2 improved/degraded/unchanged: 1/0/4
- MQ3 improved/degraded/unchanged: 1/0/4
- Generation failures: 0; MQ2/MQ3 shortfall questions: 0/0
- Average valid variants: 3.00; rejected variants: 0
- Drift-warning questions: 2
- Average generation: 6.486s (p50 6.330s; p95 7.864s)
- Average baseline/MQ2/MQ3 retrieval: 0.208s / 0.255s / 0.234s
- Average MQ2/MQ3 total: 6.742s / 6.720s

## Interpretation checkpoint

No production adoption decision is encoded here. Inspect aggregate metrics, regressions, query shortfalls, and drift warnings together before drawing a conclusion. The companion JSON retains per-query rankings and fused candidates.
