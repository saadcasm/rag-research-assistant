# Phase 10J: Coverage-Aware Evidence-Set Selection

Status: **complete; deterministic coverage/diversity selection is rejected as a
production strategy**.

Phase 10J keeps retrieval and cross-encoder scoring frozen, then compares four
top-10 selection policies at candidate depths 20 and 50: baseline score order,
diversity (`0.4` redundancy penalty), query-facet coverage (`0.35` reward), and a
combined selector (`0.35` coverage, `0.2` redundancy). All selectors preserve the
baseline rank-1 result.

Cross-encoder scores are converted to within-pool rank relevance from 1 to 0 rather
than treated as probabilities. Semantic redundancy is cosine similarity over the
existing row-aligned legacy embedding index; no model is loaded or rerun for chunk
vectors. Facets are deterministic protected terms, acronyms, and mixed-case method
names from the original question. Gold hops never enter facet extraction or
selection.

At each greedy step the artifact records original rank and score, normalized
relevance, covered and newly added facets, maximum redundancy, diversity and
coverage contributions, final selection score, and selection step. Same-document
and same-page evidence remain allowed. Pairwise similarity and source/page repetition
are diagnostics, not quotas.

Run one-question smoke validation:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_coverage_reranking_eval --smoke
```

Run the 26-question multi-hop and frozen 100-question regression benchmark:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_coverage_reranking_eval --benchmark
```

Expected outputs:

- `data/evaluation/benchmarks/phase-10j-coverage-reranking-results.json`
- `docs/phase-10j-coverage-reranking-results.md`

Do not select a production strategy until multi-hop coverage, oracle gap, ordinary
retrieval regression, question movement, facet quality, and latency are reviewed.

## Validated result and decision

Neither deterministic facet coverage nor semantic diversity closes the oracle gap.
At depth 20, all selectors finish with the same Full@10 (`23.08%`) and Partial@10
(`51.92%`) as baseline. At depth 50, coverage exactly matches baseline Full@10
(`26.92%`) and Partial@10 (`44.23%`); combined selection falls to `23.08%`/`42.31%`,
and diversity falls to `15.38%`/`38.46%` despite an oracle of `53.85%`.

Coverage selection changed the selected order or membership for 10 of 26 questions
at depth 20 and 12 at depth 50, but produced zero improved and zero degraded hop
outcomes. Its deterministic signals therefore moved chunks without identifying the
missing gold evidence. Diversity changed every depth-50 set, produced no rescue,
and degraded three questions. Combined selection degraded one.

The diversity penalty did what it mathematically intended but not what retrieval
needed. At depth 50, average pairwise similarity decreased only from `0.544` to
`0.520`, while average unique documents increased from `3.35` to `3.69`. That small
novelty gain displaced useful evidence and confirms that lower redundancy is not an
automatic proxy for multi-hop completeness.

Facet extraction produced at least one facet for all 26 questions and correctly
preserved many method pairs, including `DPR`/`ColBERT`, `RAPTOR`/`GraphRAG`, and
`GTR`/`SGPT`. However, surface-form rules cannot reliably express evidence needs.
For example, the late-chunking/TextTiling question yielded only `TextTiling`, and
single method names do not separate same-paper, cross-section hops. Facet occurrence
also says that a chunk mentions a method, not that it contains the required fact.

Frozen regression reinforces the rejection. Pure diversity reduces Hit@3 and
Hit@5 at both depths. Coverage is safe and yields small ranking improvements—for
example depth-20 Hit@3 rises from `72.22%` to `73.33%`—but provides no multi-hop
gain. Selector computation itself is cheap (`~1.6 ms` at depth 20 and `~4.9 ms` at
depth 50), so effectiveness rather than selector latency is the failure.

Production remains depth 20 with normal cross-encoder ordering. Phase 10J stays as
auditable experimental code and no generation comparison is justified. The
remaining gap requires a richer evidence-need or set-quality signal; merely tuning
these deterministic weights would risk benchmark overfitting without addressing
the observed semantic limitation.
