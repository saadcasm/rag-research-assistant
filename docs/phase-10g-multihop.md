# Phase 10G: Query Decomposition and Multi-Hop Retrieval

Status: **complete — validated negative experiment; not adopted in production**.

Phase 10G is isolated experimental work. It is not imported by the CLI query path,
FastAPI, `RAGApplication`, or the production LangGraph workflow. The production
stack remains legacy chunks, dense + BM25, RRF, one cross-encoder rerank, complete
chunks, and Qwen generation.

## Why this experiment needs a separate dataset

The frozen Phase 7.5 benchmark is useful for regression testing, but its labels
mostly ask whether one expected page or document is retrieved. Query decomposition
is intended for questions that require multiple complementary evidence components.
Scoring that claim with a primarily single-evidence dataset would validate the
wrong behavior.

The draft dataset at
`data/evaluation/questions-phase-10g-multihop-draft.jsonl` contains 26 questions:

- 22 answerable questions with all required evidence in the corpus;
- 4 deliberately partially answerable questions with one documented missing hop;
- two or more explicit hops per question, each with expected facts, document,
  physical pages, and an extracted supporting passage.

Every available passage was independently re-located in the current 3,793 legacy
chunks using exact matching after deterministic normalization of whitespace,
line-end hyphenation, common PDF ligatures, and punctuation. No fuzzy semantic
threshold is used. The draft currently has zero programmatic evidence-location
failures. Human review is still required because exact text presence does not prove
that a question genuinely needs multiple chunks or that its wording is unambiguous.

## Decomposition contract

`QueryDecomposer` asks `qwen3.5:4b` at temperature 0 for a small object:

```json
{
  "needs_decomposition": true,
  "subquestions": ["...", "..."]
}
```

Ollama's native JSON-schema `format` field is used through the existing generator
adapter. Parsing and validation still happen locally because syntactically valid
JSON can be semantically wrong. Empty questions, exact duplicates, copies of the
original, obvious same-content-word paraphrases, and entries beyond the maximum of
three are rejected. A valid decomposition needs at least two remaining
sub-questions. Invalid output retains the raw response and falls back to baseline.

This validation is intentionally conservative. It catches clear contract failures;
it does not pretend that an arbitrary embedding-similarity threshold can prove two
questions are complementary.

## Retrieval conditions

All branches reuse the frozen hybrid retriever before one final rerank:

```text
sub-question -> Qdrant dense + BM25 -> RRF hybrid ranking

branch rankings -> equal RRF (k=60) -> stable chunk deduplication
                -> cross-encoder(original question, real chunk) -> top 10
```

The experiment compares:

1. **baseline** — original query only;
2. **always_decompose** — sub-question branches only;
3. **original_plus_decomposed** — original and sub-question branches fused.

The guarded third condition preserves the original-query branch. Generated
sub-questions influence retrieval only; they are never evidence. Every final chunk
records all query branches that contributed it. A branch failure reproduces the
baseline rather than returning a partial experimental ranking.

## Hop metrics

For each question and K in 3, 5, and 10:

- **Full Hop Coverage@K** is 1 only when every required hop has a matching
  document/page in the first K results.
- **Partial Hop Coverage@K** is the fraction of required hops recovered.
- Average hops recovered, zero/one/all-hop counts, source diversity,
  deduplication, and dominant-branch share expose failure modes hidden by Hit@K.

Partially answerable questions include the missing hop in the denominator. Their
full coverage should therefore remain false; the experiment can show that the
available half was retrieved without pretending the absent evidence exists.

## Artifacts and reproducibility

Each question artifact stores the raw model response, parsed and rejected
sub-questions, decomposition decision and latency, every branch ranking, fused
candidates, final reranked results, contribution provenance, expected/retrieved
hops, coverage, deduplication, source diversity, and fallback details.

After approving the draft dataset, run in order:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_multihop_eval --smoke
.venv/bin/python -m rag_research_assistant.experiments.phase10_multihop_eval --diagnostic 5
.venv/bin/python -m rag_research_assistant.experiments.phase10_multihop_eval --benchmark
.venv/bin/python -m rag_research_assistant.experiments.phase10_multihop_eval --regression
```

The diagnostic, benchmark, and regression modes write the filenames specified in
the Phase 10G brief under `data/evaluation/benchmarks/` and `docs/`.

Do not run the generation diagnostic yet. It should be implemented only after the
retrieval artifacts identify a sensible condition and 8–12 representative cases.
That avoids baking a generation study around a retrieval configuration before its
hop behavior is known.

## Human validation checklist

Review every draft row for:

- whether two distinct evidence needs are truly necessary;
- whether one legacy chunk already contains the entire answer;
- whether expected facts over-interpret the quoted passage;
- whether comparisons avoid implying causal or apples-to-apples claims;
- whether the four absent-paper hops are genuinely absent from the corpus;
- whether categories provide enough comparison, cross-section, design/outcome,
  benchmark, and partial-answer coverage;
- whether the repeated use of page-one abstracts creates an evaluation bias.

The dataset passed this review with four targeted wording/evidence corrections.
The completed experiment found no full-hop coverage gain, so the generation
diagnostic was intentionally skipped. See
[the final results](phase-10g-multihop-results.md).
