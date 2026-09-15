# Phase 10D: parent-child retrieval

Phase 10D is an isolated retrieval/context experiment. It does not change the
production CLI, FastAPI, `RAGApplication`, LangGraph workflow, frozen questions,
legacy default, or generation path. No query-time LLM call is introduced.

## Why parent-child retrieval

Retrieval and answer context have different ideal sizes. Small chunks give the
embedding model and cross-encoder a focused matching unit, but may omit nearby
definitions, qualifications, or experimental details. Large chunks carry that
context but mix more topics into one retrieval vector.

Parent-child retrieval keeps those roles explicit:

```text
question
  -> retrieve and rerank small child chunks
  -> map child IDs to prebuilt parent IDs
  -> deduplicate and aggregate parents
  -> return broader contexts with child provenance
```

The experiment compares the unchanged legacy baseline with semantic-child
retrieval expanded into either physical pages or conservative structural
parents.

## Chosen children

The existing `semantic` chunks are reused. No new chunker or index is created.
On the 52-paper corpus they provide 5,124 searchable units averaging roughly
563 characters, compared with 3,793 legacy chunks averaging roughly 1,006
characters. Phase 7.5 also showed that semantic chunks can localize some
terminology and section-detail questions well, although they were not the best
general default.

The existing semantic Qdrant collection, embedding model, BM25 implementation,
hybrid RRF, candidate depth, and cross-encoder are reused. The child stack is:

```text
semantic child -> Qdrant dense + BM25 -> RRF -> cross-encoder -> ranked children
```

Children are reranked before parent mapping. The parent layer never calls an
LLM or scores parent text with another model.

## Page parents

A page parent is reconstructed from the current pypdf extraction and identified
by a stable hash of `(document, physical page)`. A single-page child maps
directly to that page. For a child spanning two pages, the page with the greatest
child-word overlap is selected deterministically and a warning is recorded.

This singular mapping keeps parent aggregation understandable. It also means a
cross-page child is not guaranteed to be wholly contained in its selected page;
the containment flag makes that limitation visible.

## Structural parents and fallback

The existing bounded structural chunks are reused as structural context units.
They average roughly 1,100 characters, may span up to three pages, and never
turn an entire PDF into one parent.

PDF heading detection is heuristic and visibly noisy. A semantic child therefore
receives a structural parent only when exactly one same-document,
same-section-title structural chunk contains the child's complete normalized
text. Multiple matches, no match, or missing section metadata are treated as
unreliable. Those children fall back to their page parent with an explicit
reason.

The completed benchmark found exact structural mappings for 3,310/5,124
children (64.6%); the remainder used page fallback. See the
[Phase 10D results](phase-10d-parent-child-results.md) for retrieval, context,
containment, and latency analysis.

## Parent aggregation

Twenty children are retrieved and cross-encoder reranked. Children mapping to
the same parent are deduplicated. A parent's score is:

```text
sum(1 / (60 + child_rank))
```

This is reciprocal-rank aggregation using the project's established constant.
It gives consensus credit when several strong children support the same parent,
without mixing uncalibrated cross-encoder scores or adding learned weights.
Ties use best child rank and deterministic first-seen order.

Every parent result records its contributing child IDs and ranks, best child
score/rank, provenance, section title, character/token size, expansion ratio,
containment result, and fallback warnings.

## Parent relevance

Child metrics use the frozen evaluator unchanged. Parent relevance is evaluated
separately: a parent is relevant when its document and page span cover an
expected source document/page. A child-specific label is not compared to a
parent ID because these are different identity domains.

The report also checks whether relevant retrieved children map into returned
parents and whether those parents contain the contributing child text. Parent
Hit/Recall therefore measure provenance coverage, not answer-generation quality.

## Cost measurements

Parent construction happens once at experiment startup. Per query, the report
separates baseline retrieval, child retrieval, mapping lookup, page aggregation,
and structural aggregation. Parent lookup and rank aggregation should be cheap;
the semantic child retriever remains the model-intensive part. No Ollama timing
is involved.

## Manual validation

Run from the repository root.

One-question smoke test:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_parent_child_eval --smoke
```

It prints the question, legacy baseline results, semantic children, both parent
rankings, sizes, expansion ratios, contributing child IDs, mapping warnings,
structural coverage, and timing.

Optional five-question diagnostic:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_parent_child_eval --diagnostic 5
```

Expected diagnostic artifacts:

- `data/evaluation/benchmarks/phase-10d-parent-child-diagnostic.json`
- `docs/phase-10d-parent-child-diagnostic.md`

Full frozen benchmark:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_parent_child_eval --benchmark
```

Expected final artifacts:

- `data/evaluation/benchmarks/phase-10d-parent-child-results.json`
- `docs/phase-10d-parent-child-results.md`

The full run performs no generation, but it loads two retrieval stacks and runs
both baseline and semantic-child retrieval for 100 questions. It is therefore a
manual validation checkpoint, not a CI test.

## Interpretation limits

Parent-level provenance metrics should be read beside child retrieval metrics,
containment, expansion size, fallback coverage, and individual regressions.
A larger context covering the labelled page is not automatically a better
answer context. No integration decision should be made before the saved results
are inspected.
