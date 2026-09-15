# Phase 10F: HyDE results

HyDE generates one answer-like passage as a semantic probe. The hypothetical text is never evidence, and every final cross-encoder call uses the original question against real legacy chunks.

Questions: 5 (5 scored; 0 unanswerable)

| Metric | Baseline | HyDE-only | Baseline + HyDE |
|---|---:|---:|---:|
| Hit@1 | 0.6000 | 0.4000 | 0.6000 |
| Hit@3 | 0.6000 | 0.6000 | 0.6000 |
| Hit@5 | 0.8000 | 0.8000 | 0.8000 |
| Recall@1 | 0.6000 | 0.4000 | 0.6000 |
| Recall@3 | 0.6000 | 0.6000 | 0.6000 |
| Recall@5 | 0.8000 | 0.8000 | 0.8000 |
| Mean first relevant rank | 1.7500 | 2.0000 | 1.7500 |

## Category Hit@5

| Category | Questions | Baseline | HyDE-only | Fused |
|---|---:|---:|---:|---:|
| difficult_distractor | 1 | 1.0000 | 1.0000 | 1.0000 |
| exact_number | 3 | 0.6667 | 0.6667 | 0.6667 |
| exact_terminology | 1 | 1.0000 | 1.0000 | 1.0000 |

## Rank movement

- HyDE-only improved/unchanged/degraded: 0/4/1
- Fused improved/unchanged/degraded: 0/5/0
- HyDE-only rescued/lost top-five evidence: 0/0
- Fused rescued/lost top-five evidence: 0/0

## Probe and latency

- Average hypothetical length: 83.6 words / 648.6 characters
- Fallbacks/errors/truncations: 0/0/0
- Missing-protected-term / numeric-detail / citation-like output questions: 0/2/0
- Introduced-entity / any-warning questions: 5/2
- Mean baseline retrieval: 0.256s
- Mean generation / embedding / HyDE dense retrieval: 3.539s / 0.020s / 0.002s
- Mean HyDE-only / fused total: 3.714s / 3.968s
- Unanswerable top-result changes, HyDE-only/fused: 0/0
- Mean unanswerable fused overlap with baseline top five: 0.00/5

## Interpretation boundary

Do not infer a recommendation from this generated table alone. Inspect rescues, regressions, numeric/entity drift, unanswerable probes, and fusion safeguards before deciding whether HyDE merits retention as an optional experiment.
