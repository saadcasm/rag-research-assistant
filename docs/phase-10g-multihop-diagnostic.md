# Phase 10G: Query Decomposition and Multi-Hop Retrieval

> Experimental only. The production retrieval path is unchanged.

Dataset: `data/evaluation/questions-phase-10g-multihop-draft.jsonl`

Questions: **5**

Answerable: **5**; partially answerable: **0**

Successful decompositions: **4**; fallbacks: **0**

## Hop coverage

| Condition | Full@3 | Full@5 | Full@10 | Partial@3 | Partial@5 | Partial@10 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.3000 |
| always_decompose | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.4000 |
| original_plus_decomposed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.3000 |

## Interpretation status

No production decision is encoded here. Review the per-question artifact before drawing conclusions.
