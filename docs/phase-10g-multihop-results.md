# Phase 10G: Query Decomposition and Multi-Hop Retrieval Results

Phase 10G is experimental only. The production retrieval and generation paths
remain unchanged.

## Configuration

- Dataset: 26 validated questions (22 answerable, 4 partially answerable)
- Decomposer: `qwen3.5:4b`, temperature 0, native JSON-schema output
- Maximum sub-questions: 3
- Retrieval per branch: legacy Qdrant dense + BM25 + RRF
- Cross-branch fusion: equal RRF, `k=60`, candidate depth 20
- Final relevance model: existing cross-encoder against the original question
- Final depth: 10

## Main result

| Condition | Full@3 | Full@5 | Full@10 | Partial@3 | Partial@5 | Partial@10 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.0000 | 0.0000 | 0.0769 | 0.1731 | 0.2500 | **0.4423** |
| Always decompose | 0.0000 | 0.0000 | 0.0769 | 0.1731 | 0.2500 | 0.4038 |
| Original + decomposed | 0.0000 | 0.0000 | 0.0769 | 0.1346 | 0.2500 | 0.3846 |

Decomposition does not improve full-hop coverage at any measured depth. Only two
questions achieve full coverage at 10, and all three conditions retrieve both hops
for the same two questions: `p10g_lostmiddle_ruler_longcontext` and
`p10g_hnsw_faiss_ann`. The baseline therefore already solved every measured
full-coverage success.

The four partially answerable questions can never achieve full coverage because
one labelled hop is absent. On the 22 fully answerable questions, Full@10 is
`0.0909` for every condition. Partial@10 changes from `0.4318` for baseline to
`0.3864` for always-decompose and `0.3636` for guarded fusion.

## Question-level changes

At depth 10, always-decompose gains one hop on
`p10g_coil_colbert_interactions`, but loses one on each of:

- `p10g_selfrag_crag_control`
- `p10g_gtr_sgpt_scaling`
- `p10g_raptor_build_use`

Original-plus-decomposed records those same three losses and no gain. At depth 3,
the guarded condition also loses one available hop on the partially answerable
BEIR/MIRACL case. These are changes in exact grounded-hop evidence, not merely
document-name overlap.

The largest category is cross-paper comparison. Its Partial@10 falls from `0.3929`
to `0.3571` with always-decompose and `0.3214` with guarded fusion. The two
cross-section questions fall from `0.5000` to `0.2500` in both decomposed
conditions. The single problem/evaluation question is the only category with full
coverage, but baseline already retrieves it.

## Decomposition quality

The model requested decomposition for 25 of 26 questions: 24 received two
sub-questions and one received three. There were no JSON parse failures and no
runtime fallbacks.

Structured output reliability therefore worked, but semantic planning remained
unreliable:

- BEIR versus BRIGHT was incorrectly declared not to need decomposition.
- The DPR versus ColBERT decomposition produced only DPR sub-questions and omitted
  ColBERT entirely.
- The GTR versus SGPT decomposition produced three GTR-focused questions and
  omitted SGPT.
- The HyDE/query2doc decomposition described HyDE as query expansion and
  query2doc as document selection, blurring their actual roles.
- Some outputs combined comparison work into one sub-question instead of assigning
  exactly one evidence need per branch.

This demonstrates why syntactically valid JSON is not equivalent to a valid
retrieval plan. The deterministic validator catches duplicates and superficial
paraphrases, but cannot prove that every requested entity or evidence component is
represented.

## Evidence balance

| Condition | Mean source diversity@10 | Mean duplicate occurrences | Mean dominant-query share |
|---|---:|---:|---:|
| Baseline | 3.0769 | 0.0000 | 1.0000 |
| Always decompose | 2.9231 | 5.1538 | 0.8038 |
| Original + decomposed | 2.8846 | 19.8077 | 0.9077 |

RRF does not automatically balance evidence needs. The guarded condition has heavy
cross-branch overlap and a 90.8% mean dominant-query share in the final ten. Source
diversity is slightly lower than baseline. One sub-query can therefore dominate
even though all branches enter fusion with equal formula weights.

The final cross-encoder is intentionally scored against the original question. It
protects user intent, but a single scalar relevance ranking has no explicit reason
to reserve space for every hop. This experiment shows the difference between
ranking relevance and set-level evidence coverage.

## Latency

Mean per-question timings on the 26-question run:

| Stage/condition | Seconds |
|---|---:|
| Decomposition generation | 1.9840 |
| Baseline retrieval | 0.0565 |
| Baseline reranking | 0.1499 |
| Always-decompose branch retrieval | 0.0503 |
| Always-decompose reranking | 0.1362 |
| Guarded branch retrieval | 0.1051 |
| Guarded reranking | 0.1339 |
| Guarded total including decomposition | 2.2231 |

Fusion itself averages under `0.00004` seconds. Generation dominates the added
cost. The regression run spent about 163.7 seconds on decomposition alone across
100 questions, averaging 1.637 seconds each.

## Decision

Do not integrate query decomposition into production. The experiment adds roughly
two seconds per multi-hop request, does not improve full-hop coverage, lowers
partial coverage, and sometimes omits an entire named method.

The original-query safeguard remains valuable: on the frozen regression benchmark,
guarded fusion exactly preserves every aggregate baseline metric. However, safety
without multi-hop benefit does not justify the latency or complexity.

No generation diagnostic is run. The brief requires a sensible retrieval
configuration before comparing answer synthesis; neither decomposed condition
provides more complete evidence than baseline. Running generation here would test
model prose over equally or less complete evidence, not the research hypothesis.

The implementation is retained as an auditable negative experiment. A future
iteration would need explicit entity/hop coverage validation and coverage-aware
result selection—not merely more prompt tuning or another equal-weight RRF pass.
