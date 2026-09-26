# Phase 10H-1: Candidate-Pool Coverage and Oracle Ceiling

Status: **complete; Phase 10H-2 stopped because the candidate pool has insufficient
full-hop headroom**.

Phase 10H asks whether multi-hop failure occurs before or after final evidence-set
selection. It first measures whether each required evidence passage already exists
in the frozen candidate pool. Gold hop labels are used only for this retrospective
measurement; they never influence retrieval or candidate ordering.

## Reused data

The Phase 10G result artifact contains, for every question:

- the final cross-encoder top 10;
- the baseline hybrid ranking before cross-encoder reranking;
- exactly 20 baseline hybrid candidates in their original order.

Phase 10H-1 therefore reuses those persisted candidates. It does not initialize or
run Qdrant, BM25, an embedding model, a cross-encoder, or Ollama. Depths 5, 10, and
20 are available. Depth 50 is explicitly reported as unavailable rather than
estimated or fabricated.

## Label-isolation boundary

```text
Phase 10G retrieval artifact + legacy chunk lookup
    -> reconstruct frozen candidate pools (no gold arguments accepted)

validated Phase 10G hop labels + frozen pool
    -> evaluator-only passage matching
    -> oracle availability metrics
```

`load_phase10g_candidate_pools` cannot inspect gold hops because they are not an
input to the function. It reads only question identity, the saved baseline branch,
candidate rank, chunk ID, and score. `evaluate_candidate_pool` receives the frozen
pool afterward and uses labels solely to determine first matching ranks and oracle
coverage.

## Metrics

At each available depth, the report records:

- Oracle Full Hop Availability;
- Partial Hop Availability;
- average recovered hops;
- zero-, one-, and all-hop counts;
- first candidate rank for every expected hop;
- answerable and partially-answerable cohorts;
- category-level breakdown;
- headroom relative to current final Full Hop Coverage@10.

Partially answerable questions also receive separate available-hop metrics. Their
intentionally absent hop remains absent, so required-hop full coverage stays false
while available-hop full coverage can be true.

Every question retains the original candidate chunk IDs and ranks, evaluator-only
hop matches, duplicate warnings, unavailable-depth warnings, and evaluation
latency. Input artifact, dataset, and chunk-file SHA-256 values make the report
reproducible.

## Manual command

From the repository root:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_evidence_selection_eval --candidate-coverage
```

Expected outputs:

- `data/evaluation/benchmarks/phase-10h-candidate-coverage-results.json`
- `docs/phase-10h-candidate-coverage-results.md`

The command is local and model-free. The checked-in result is the manually validated
run used for the decision below.

## Stop rule

Do not implement Phase 10H-2 until this report is reviewed. If candidate-pool
Full Hop Availability@20 is close to final Full Hop Coverage@10, candidate
generation is the bottleneck. If it is substantially higher, set selection has
measurable headroom and a small fixed family of label-free selectors may be
justified.

## Validated result and decision

The 26-question run found that the limiting problem is candidate generation, not
final evidence-set selection:

- the current final top 10 has full-hop coverage for 2 of 26 questions (`7.69%`),
  or 2 of 22 fully answerable questions (`9.09%`);
- the depth-20 oracle contains every required hop for only 4 of 26 questions
  (`15.38%`), or 4 of 22 fully answerable questions (`18.18%`);
- therefore, even a perfect label-aware selector over the saved top 20 could rescue
  at most two additional fully answerable questions;
- 18 of 22 answerable questions still lack at least one exact gold passage at depth
  20, including three with no recovered hop at all;
- all four partially answerable questions recover their one intentionally available
  hop by depth 5. Their unavailable second hop is correctly excluded from the
  available-hop metric.

The two theoretical selector rescues are `p10g_selfrag_crag_control` (gold hops first
appear at ranks 9 and 16) and `p10g_raptor_build_use` (ranks 8 and 14). The two other
depth-20-complete questions are already complete in the final top 10. This is real
but narrow headroom: selector work could improve at most 2/22 answerable questions
in the observed pool, while it cannot help the remaining 18.

Phase 10H therefore stops after 10H-1. Diversity-aware, MMR-style, and query-facet
selectors were deliberately not implemented. The next useful investigation should
target upstream candidate recall (and, separately, whether exact passage-level
labels expose chunk-boundary mismatch), not rearrangement of the same 20 candidates.
Depth 50 was not persisted by Phase 10G, so this conclusion is limited to the
observed depth-20 pool and does not claim that deeper retrieval has no value.

The raw metrics and per-question evidence are in the
[candidate-coverage result](phase-10h-candidate-coverage-results.md) and its
machine-readable JSON counterpart.
