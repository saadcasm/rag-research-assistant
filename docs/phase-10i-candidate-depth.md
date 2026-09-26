# Phase 10I-2: Controlled Candidate-Depth Experiment

Status: **complete; depth 20 remains the production default because deeper oracle
headroom did not survive cross-encoder reranking**.

Phase 10I-2 varies one independent variable: the number of candidates retrieved by
each branch, fused, and scored by the cross-encoder. Production remains at depth 20.

## Conditions

For each original question, every condition runs independently:

```text
Qdrant dense top N + BM25 top N
    -> unchanged RRF(k=60), truncated to N unique candidates
    -> unchanged cross-encoder scores every fused candidate
    -> fixed final top 10
```

`N` is 20, 50, or 100. Thus the expected cross-encoder workloads are at most 20,
50, and 100 candidates; the persisted count makes any reduction from duplicate
fusion explicit. No query transformation, generation, model change, parameter
tuning, or chunking change is present.

Retrieval and reranking run without gold labels. Exact Phase 10G passage labels are
applied afterward to measure candidate-pool oracle coverage and final coverage.
Partially answerable questions are evaluated only against available hops.

## Measurements

The multi-hop report contains Full/Partial Hop Coverage and recovered-hop counts at
final ranks 3, 5, and 10; candidate-pool oracle coverage; 20-to-50 and 20-to-100
question movement; rescued/regressed hop IDs; branch, pre-rerank RRF, and final
cross-encoder ranks for every hop; and dense, BM25, RRF, reranker, and total latency.

The same three conditions run on the frozen Phase 7.5 100-question benchmark and
report Hit@1/3/5, Recall@1/3/5, MFR, latency, and per-question results. Unanswerable
questions remain outside retrieval scoring.

No memory estimate is fabricated. Candidate counts and latency provide the practical
cost indicators; qualitative memory observations can be recorded after the run.

## Manual commands

One multi-hop question across all depths:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_candidate_depth_eval --smoke
```

Complete multi-hop plus frozen-regression benchmark:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_candidate_depth_eval --benchmark
```

Expected outputs:

- `data/evaluation/benchmarks/phase-10i-candidate-depth-results.json`
- `docs/phase-10i-candidate-depth-results.md`

Do not select a production depth until oracle availability, final reranked coverage,
ordinary retrieval regression, question-level rescues/regressions, and cross-encoder
latency have all been reviewed together.

## Validated result and decision

The controlled run confirms that candidate generation has substantial deeper-pool
headroom, but the existing pointwise cross-encoder does not convert it into a better
final evidence set:

| Depth | Oracle full | Final Full@10 | Final Partial@10 | Mean total latency |
|---:|---:|---:|---:|---:|
| 20 | 30.77% | 23.08% | 51.92% | 0.170 s |
| 50 | 53.85% | 26.92% | 44.23% | 0.365 s |
| 100 | 73.08% | 19.23% | 40.38% | 0.698 s |

Depth 50 rescues only `p10g_splade_deepimpact_sparse`: its DeepImpact hop enters
through BM25 rank 21, reaches RRF rank 39, and is promoted to final rank 10. Against
that one rescue, five questions lose their only recovered hop. Depth 100 produces no
net improved question relative to depth 20, degrades six, and turns one previously
full question into partial coverage.

The frozen regression does not justify the extra cost either. Depth 50 leaves
Hit@1/3 unchanged and improves Hit@5 by only 1.11 percentage points and Recall@5 by
1.67 points. Depth 100 slightly reduces Hit/Recall@1 and @3 and returns Hit@5 to the
depth-20 value. Mean total latency rises by about `2.15x` at depth 50 and `4.11x` at
depth 100; the cross-encoder dominates this increase (`0.143 s`, `0.343 s`, and
`0.670 s` per multi-hop question).

This is outcome D from the experiment design: deeper pools improve oracle coverage,
but the cross-encoder fails to promote the required passages reliably. Adding more
independently relevant candidates creates competition for a fixed top 10 and can
evict evidence that covered a different hop. The production candidate depth remains
20. Phase 10I-2 does not modify production configuration.

The next research question should target reranking or evidence-set objectives rather
than blindly increasing depth. Any follow-up must still account for the residual
document-localization failures found in Phase 10I-1 and must preserve the frozen
regression benchmark.
