# Phase 10I-1: Component-Level Candidate-Generation Analysis

> Gold passages are applied only after dense, BM25, and RRF rankings are frozen.

Dataset: `data/evaluation/questions-phase-10g-multihop-draft.jsonl`

Questions: **26**; available evidence hops: **49**

Original questions only; depths: **5, 10, 20, 50, 100**; RRF k: **60**.

## Passage recall

| Branch | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|
| dense | 0.2245 | 0.3265 | 0.4694 | 0.6327 | 0.7959 |
| bm25 | 0.2857 | 0.3878 | 0.4694 | 0.7347 | 0.8163 |
| hybrid | 0.2041 | 0.3878 | 0.5714 | 0.7755 | 0.8571 |

## Document recall

| Branch | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|
| dense | 0.7755 | 0.8571 | 0.9796 | 1.0000 | 1.0000 |
| bm25 | 0.6939 | 0.8367 | 0.9388 | 0.9796 | 1.0000 |
| hybrid | 0.7347 | 0.8776 | 1.0000 | 1.0000 | 1.0000 |

## Correct document found, exact passage missing

| Branch | @5 | @10 | @20 | @50 | @100 |
|---|---:|---:|---:|---:|---:|
| dense | 0.5510 | 0.5306 | 0.5102 | 0.3673 | 0.2041 |
| bm25 | 0.4082 | 0.4490 | 0.4694 | 0.2449 | 0.1837 |
| hybrid | 0.5306 | 0.4898 | 0.4286 | 0.2245 | 0.1429 |

## Available-hop question coverage

Partially answerable questions are scored only against their available hop.

| Branch/depth | Full | Partial | Zero-hop | One-hop | All-hop |
|---|---:|---:|---:|---:|---:|
| dense @5 | 0.0769 | 0.2500 | 15 | 11 | 2 |
| dense @10 | 0.1538 | 0.3654 | 11 | 14 | 4 |
| dense @20 | 0.2692 | 0.5192 | 6 | 17 | 7 |
| dense @50 | 0.4615 | 0.6667 | 3 | 15 | 12 |
| dense @100 | 0.6923 | 0.8205 | 1 | 11 | 18 |
| bm25 @5 | 0.1538 | 0.3269 | 13 | 12 | 4 |
| bm25 @10 | 0.2308 | 0.4423 | 9 | 15 | 6 |
| bm25 @20 | 0.3077 | 0.5192 | 7 | 15 | 8 |
| bm25 @50 | 0.6154 | 0.7628 | 2 | 12 | 16 |
| bm25 @100 | 0.7308 | 0.8397 | 1 | 10 | 19 |
| hybrid @5 | 0.1538 | 0.2692 | 16 | 10 | 4 |
| hybrid @10 | 0.2308 | 0.4423 | 9 | 15 | 6 |
| hybrid @20 | 0.3846 | 0.6154 | 4 | 16 | 10 |
| hybrid @50 | 0.6538 | 0.8077 | 1 | 12 | 17 |
| hybrid @100 | 0.7308 | 0.8718 | 0 | 10 | 19 |

## Category-level hybrid coverage

Small categories are descriptive only.

| Category | Questions | Full@5 | Full@10 | Full@20 | Full@50 | Full@100 |
|---|---:|---:|---:|---:|---:|---:|
| benchmark_synthesis | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cross_paper_comparison | 14 | 0.0000 | 0.0000 | 0.2143 | 0.5714 | 0.7143 |
| cross_paper_synthesis | 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| cross_section_synthesis | 2 | 0.0000 | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| design_and_outcome | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| method_evolution | 1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| partially_answerable | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| problem_and_evaluation | 1 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Diagnostic counts

```json
{
  "classification_counts": {
    "DENSE_AND_BM25_FOUND": 33,
    "DENSE_ONLY_FOUND": 6,
    "BM25_ONLY_FOUND": 7,
    "FULL_MISS": 3
  },
  "flag_counts": {
    "FOUND_ONLY_DEEP": 14,
    "DOCUMENT_FOUND_PASSAGE_MISSED": 3,
    "BOTH_FOUND_BUT_RRF_DEMOTED": 7
  },
  "first_passage_rank": {
    "dense": {
      "found_hops": 39,
      "median": 14.0,
      "p75": 48.0,
      "p90": 72.0
    },
    "bm25": {
      "found_hops": 40,
      "median": 14.0,
      "p75": 32.0,
      "p90": 50.0
    },
    "hybrid": {
      "found_hops": 42,
      "median": 11.5,
      "p75": 28.0,
      "p90": 50.0
    }
  }
}
```

Retrieval-only latency totals:

```json
{
  "dense": 0.42955499899107963,
  "bm25": 0.3588940409827046,
  "rrf": 0.0028135420216131024,
  "total": 0.7912625819953973
}
```

## Interpretation status

This generated report records diagnostics but does not select or implement a retrieval intervention. Review the per-hop records before deciding Phase 10I-2.
