# Phase 10I-1: Candidate-Generation Diagnostics

Status: **complete; the validated diagnostic identifies shallow passage localization
as the dominant failure, with a small residual of branch and fusion misses**.

Phase 10H established that most missing multi-hop evidence never reaches the saved
top-20 candidate pool. Phase 10I-1 diagnoses which candidate-generation component
loses each exact evidence passage before any intervention is selected.

Production retrieval remains unchanged.

## Retrieval boundary

The experiment reuses the established legacy-corpus resources but loads no
cross-encoder and makes no Ollama call:

```text
original multi-hop question
    |-- Qdrant dense search to depth 100
    |-- existing BM25 search to depth 100
    `-- unchanged RRF over those two frozen rankings to depth 100

frozen rankings + gold evidence labels
    `-- evaluator-only passage/document matching and diagnostics
```

`retrieve_component_rankings` accepts only a question identifier, the original
question text, two retrievers, and retrieval settings. It cannot inspect a
`MultiHopExample`, expected document, page, passage, or answer key. The separate
`evaluate_component_rankings` function receives the frozen rankings and gold hops
after retrieval. This API boundary protects against evaluation leakage.

The experimental runner constructs the same Qdrant dense and BM25 branches used by
the production hybrid retriever, with reranking explicitly disabled. It calls the
branches directly so it can retain their independent rankings, then invokes the
same `reciprocal_rank_fusion` implementation with the unchanged `rrf_k=60`.

## Depths and matching

The default run retrieves once to depth 100 and evaluates at 5, 10, 20, 50, and
100. This makes all reported depths prefixes of the same ranking and avoids repeated
query embedding or BM25 work.

A passage hit uses the exact deterministic Phase 10G/10H matcher: document and page
must match, and the stored passage must occur in the chunk after normalization of
whitespace, line hyphenation, common ligatures, punctuation, and case. Document
recall is reported independently, so a correct paper with the wrong paragraph is
not mistaken for passage recall.

For partially answerable questions, intentionally unavailable hops remain explicit
but are excluded from recall denominators and question-level full-hop coverage.

## Persisted diagnostics

For every available hop, the artifact retains:

- first passage and document ranks for dense, BM25, and RRF;
- passage/document hits at every measured depth;
- correct-document/passage-missing flags at every depth;
- dense/BM25 branch classification at the maximum depth;
- deep-only and significant RRF-demotion flags;
- the RRF score and contributing branch ranks for the first matching fused chunk;
- exact-token overlap, query-term coverage, protected method/acronym preservation,
  and simple PDF-artifact indicators;
- dense cosine score for the first exact matching dense chunk when it is retrieved;
- complete component rankings and retrieval/fusion latency.

The primary classifications are `DENSE_AND_BM25_FOUND`, `DENSE_ONLY_FOUND`,
`BM25_ONLY_FOUND`, and `FULL_MISS`. Non-exclusive flags capture
`FOUND_ONLY_DEEP`, `DOCUMENT_FOUND_PASSAGE_MISSED`, and
`BOTH_FOUND_BUT_RRF_DEMOTED`. RRF demotion is diagnostic rather than a relevance
judgment: it means a passage at branch rank 20 or better moved at least 10 ranks
deeper, or fell outside the fused depth.

Aggregates include passage and document recall, document-found/passage-missing rate,
median/p75/p90 first passage rank, available-hop multi-hop coverage, branch outcome
fractions, category summaries, and retrieval latency. Small Phase 10G categories
must be treated as descriptive rather than statistically conclusive.

## Manual commands

One-question smoke diagnostic:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_candidate_generation_eval --smoke
```

The smoke output prints the question, evaluation-only hop labels, top ten from all
three component rankings, first passage/document ranks, classifications, and
latency. It does not write the benchmark artifacts.

Full 26-question analysis:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_candidate_generation_eval --benchmark
```

Expected outputs:

- `data/evaluation/benchmarks/phase-10i-candidate-generation-analysis.json`
- `docs/phase-10i-candidate-generation-analysis.md`

## Validation decision boundary

Do not change candidate depth, models, BM25, RRF, chunking, or query construction
until the real artifact is reviewed. The results must first distinguish evidence
that is merely deep from evidence missed by both branches, fusion demotion, and
correct-document/wrong-passage localization failure. Phase 10I-2 is intentionally
not implemented at this checkpoint.

## Validated result

The manual 26-question run evaluated 49 available evidence hops. Its strongest
finding is the gap between document discovery and exact passage localization:

- hybrid document recall reaches `100%` at depth 20;
- hybrid passage recall is only `57.14%` at depth 20, then rises to `77.55%` at
  depth 50 and `85.71%` at depth 100;
- the corresponding correct-document/passage-missing rates are `42.86%`, `22.45%`,
  and `14.29%`;
- hybrid available-hop full-question coverage rises from `38.46%` at depth 20 to
  `65.38%` at 50 and `73.08%` at 100;
- for the 22 fully answerable questions, hybrid full coverage rises from `27.27%`
  at depth 20 to `59.09%` at 50 and `68.18%` at 100.

This is meaningful deeper-pool headroom. Fourteen of 49 available hops are found
only below rank 20. Only three hops (`6.12%`) are missed by both dense and BM25
through depth 100, while 33 are found by both, six by dense only, and seven by BM25
only. The branches are complementary, and neither should be removed based on this
benchmark.

RRF is not the dominant failure. Seven hops show a significant demotion, and four
branch-found passages do not enter the fused top 100. Nevertheless, hybrid passage
recall is higher than either individual branch at depths 20, 50, and 100. Changing
fusion parameters before testing the simpler depth hypothesis would therefore be
premature.

The three complete branch misses all still have their correct document retrieved:

- the RAG integration passage in `p10g_rag_replug_integration`;
- the DPR architecture passage in `p10g_dpr_colbert_architecture`;
- the monoT5 passage in `p10g_bert_monot5_reranking`.

These are evidence-localization failures, not corpus-discovery failures. They make
document-first/passage-second retrieval a plausible later comparison, but the
larger observed effect is that relevant passages already exist at ranks 21–100.

### Important comparability caveat

Phase 10I retrieves each dense and BM25 branch to 100, fuses those complete lists,
and evaluates prefixes of the resulting RRF ranking. Phase 10H reused Phase 10G's
production-style pool, where each branch was retrieved only to 20 before fusion.
Consequently, Phase 10I answerable-only hybrid@20 (`27.27%`) is not a
reproduction of Phase 10H answerable-only oracle@20 (`18.18%`). Deep candidates
that occur in both branches can move into the Phase 10I top 20. This is useful
evidence that branch input depth affects fusion, but it must not be described as a
production improvement. The all-question Phase 10I rate additionally uses
available-hop scoring for the four partially answerable questions, so it should not
be compared directly with Phase 10H's required-hop all-question rate.

### Phase 10I-2 direction

No intervention is implemented in 10I-1. The evidence supports making the first
future experiment a controlled candidate-depth comparison—preserving the existing
dense model, BM25, RRF, and cross-encoder—at branch/fusion depths 20, 50, and 100.
It should measure multi-hop coverage, the frozen 100-question regression, reranking
latency, and memory/cost before any production default changes. Hierarchical
document-to-passage retrieval should remain a separate follow-up for the residual
localization misses. RRF tuning or model replacement is not yet the first-choice
hypothesis.
