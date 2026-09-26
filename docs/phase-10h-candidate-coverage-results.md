# Phase 10H-1: Candidate-Pool Coverage

> Oracle labels are used only after retrieval to measure availability.

Source artifact: `data/evaluation/benchmarks/phase-10g-multihop-results.json`

Questions: **26** (22 answerable, 4 partially answerable)

No retrieval or model inference was rerun. Candidate order is the saved Phase 10G baseline hybrid order.

## All questions

| Depth | Oracle full | Partial | Available-hop full | Available-hop partial |
|---:|---:|---:|---:|---:|
| 5 | 0.0000 | 0.1923 | 0.1538 | 0.2692 |
| 10 | 0.0385 | 0.3654 | 0.1923 | 0.4423 |
| 20 | 0.1538 | 0.5192 | 0.3077 | 0.5962 |
| 50 | unavailable | unavailable | unavailable | unavailable |

## Cohorts

| Cohort/depth | Oracle full | Partial | Available-hop full | Available-hop partial |
|---|---:|---:|---:|---:|
| Answerable @5 | 0.0000 | 0.1364 | 0.0000 | 0.1364 |
| Answerable @10 | 0.0455 | 0.3409 | 0.0455 | 0.3409 |
| Answerable @20 | 0.1818 | 0.5227 | 0.1818 | 0.5227 |
| Answerable @50 | unavailable | unavailable | unavailable | unavailable |
| Partially answerable @5 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Partially answerable @10 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Partially answerable @20 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| Partially answerable @50 | unavailable | unavailable | unavailable | unavailable |

## Headroom reference

Current final Full Hop Coverage@10: **0.0769**

Current answerable-only final Full Hop Coverage@10: **0.0909**

```json
{
  "all_questions": {
    "5": -0.07692307692307693,
    "10": -0.038461538461538464,
    "20": 0.07692307692307693,
    "50": null
  },
  "answerable_only": {
    "5": -0.09090909090909091,
    "10": -0.045454545454545456,
    "20": 0.09090909090909091,
    "50": null
  }
}
```

Depth 50 is explicitly unavailable when the saved candidate pool contains only 20 rows.

## Decision status

This generated report does not use gold labels for ranking and does not automatically encode a Phase 10H-2 decision. See `docs/phase-10h-candidate-coverage.md` for the reviewed project decision.
