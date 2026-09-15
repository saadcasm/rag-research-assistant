# Phase 10D: parent-child retrieval results

Semantic child chunks are retrieved through the frozen dense+BM25+RRF+cross-encoder stack, then expanded to physical-page or conservatively matched structural parents. Parent relevance is provenance-based: the parent document/page span must cover an expected source.

## Child retrieval

| Metric | Legacy baseline | Semantic child |
|---|---:|---:|
| Hit@1 | 0.6000 | 0.4000 |
| Hit@3 | 0.6000 | 0.8000 |
| Hit@5 | 0.8000 | 0.8000 |
| Recall@1 | 0.6000 | 0.4000 |
| Recall@3 | 0.6000 | 0.8000 |
| Recall@5 | 0.8000 | 0.8000 |
| Mean first relevant rank | 1.7500 | 1.5000 |

## Parent retrieval

| Metric | Page parent | Structural parent |
|---|---:|---:|
| Hit@1 | 0.2000 | 0.4000 |
| Hit@3 | 0.4000 | 0.6000 |
| Hit@5 | 1.0000 | 0.8000 |
| Recall@5 | 1.0000 | 0.8000 |
| Mean first relevant rank | 3.4000 | 2.0000 |
| Average characters | 4212.8400 | 2622.9600 |
| Median characters | 4142.0000 | 1200.0000 |
| p95 characters | 6032.4000 | 5658.0000 |
| Average expansion ratio | 7.3075 | 4.0290 |
| Deduplication rate | 0.2700 | 0.0600 |

## Coverage and latency

- Structural exact mappings: 3310/5124 (64.6%); page fallbacks: 1814
- Average baseline/child retrieval: 0.206s / 0.170s
- Average parent mapping: 0.000040s
- Average page/structural aggregation: 0.000209s / 0.000149s

## Interpretation checkpoint

Inspect child localization, provenance-level parent metrics, evidence-containment rates, context sizes, fallbacks, and regressions before considering integration. Parent Hit does not measure answer-generation quality, and no production adoption decision is encoded here.
