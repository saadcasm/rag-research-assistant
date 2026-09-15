# Phase 10B: multi-query retrieval

Phase 10B is an isolated experiment. It does not change the CLI, FastAPI,
`RAGApplication`, LangGraph workflow, generation path, frozen 100-question
dataset, or the Phase 7.5 retrieval components.

## Why this experiment exists

Phase 10A replaced the original retrieval query with one LLM rewrite. That
created a single point of failure: a drifting rewrite could remove a useful
identity signal and destroy an otherwise good baseline ranking. Phase 10B asks
a narrower question: can alternatives add recall while the original query
continues to protect baseline evidence?

The three conditions are:

1. baseline: original query only;
2. multi-query-2: original plus the first two valid alternatives;
3. multi-query-3: original plus the first three valid alternatives.

One generation call requests three alternatives. The two multi-query conditions
use nested prefixes of that result, so they differ only in variant count and not
in separately sampled LLM output.

## Candidate flow

```text
original query ── dense + BM25 ── hybrid RRF ─┐
variant 1 ────── dense + BM25 ── hybrid RRF ─┼─ multi-query RRF
variant 2 ────── dense + BM25 ── hybrid RRF ─┤        │
variant 3 ────── dense + BM25 ── hybrid RRF ─┘        ▼
                                              cross-encoder
                                      (scores original question)
                                                      │
                                                      ▼
                                                  final top-k
```

The first RRF combines dense and lexical rankings for one query, exactly as in
the established hybrid retriever. The second RRF combines rankings across
queries. Stable chunk IDs deduplicate candidates, while a chunk found at strong
ranks under several formulations accumulates more reciprocal-rank evidence.
Only then does the unchanged cross-encoder rerank the fused candidate pool.
The cross-encoder always receives the original question, not an LLM variant.

The candidate depth remains the established value of 20 and both RRF stages use
the established `k=60`. These are frozen experimental settings, not values
tuned against Phase 10B results.

## Query generation contract

`CorpusGroundedMultiQueryGenerator` asks `qwen3.5:4b` for a small JSON object:

```json
{"queries": ["alternative one", "alternative two", "alternative three"]}
```

The prompt includes bounded preliminary evidence from the baseline's top five
results. It asks for retrieval formulations, prohibits answers and unsupported
facts, and tells the model to preserve specific identity signals. There are no
hard-coded mappings for DPR, RAG, ColBERT, paper titles, or acronyms.

Parsing and validation remove:

- exact and whitespace/case-equivalent copies of the original;
- duplicate alternatives;
- empty or excessively long alternatives;
- output beyond the requested count.

Partial output is usable. If only one valid alternative exists, both conditions
retain that one plus the original and record their respective shortfalls. A
parse error, empty response, or generator exception produces original-only
retrieval and an explicit fallback record; it does not abort the experiment.

## Drift warnings

Phase 10A's protected-token check is reused for quoted phrases, acronyms, and
model-like tokens. Phase 10B also reports explainable warning strings for:

- protected terms missing from a variant;
- new acronym/model-like identity details absent from query and evidence;
- parenthetical expansions absent from query and evidence;
- a variant with no lexical identity overlap with the original content words.

Warnings are diagnostics, not rejection scores or relevance thresholds. They
can produce false positives and must be interpreted alongside the saved query
and evidence.

## Latency and report interpretation

Multi-query retrieval deliberately does more work. Each additional formulation
runs one dense search and one BM25 search, and the conditions perform a final
cross-encoder rerank. The report separates generation, retrieval/fusion/rerank,
and total condition latency and records averages plus p50/p95 values.

The JSON report keeps aggregate metrics, experiment configuration, generated
and rejected queries, drift warnings, final ranks, per-query hybrid rankings,
fused candidates, final results, and latency. This is intentionally more detail
than the Markdown summary so results can be audited without rerunning Ollama.

The completed benchmark and interpretation are recorded in
[the Phase 10B results](phase-10b-multiquery-results.md). The implementation
guide keeps the pre-registered design separate from those measured conclusions.

## Manual validation commands

Run from the repository root with the virtual environment activated.

One-question smoke test (prints original query, generated variants, every
per-query hybrid ranking, multi-query fusion, and final reranking):

```bash
python -m rag_research_assistant.experiments.phase10_multiquery_eval --smoke
```

Optional five-question diagnostic (persists separate diagnostic artifacts and
does not overwrite final results):

```bash
python -m rag_research_assistant.experiments.phase10_multiquery_eval --diagnostic 5
```

This creates:

- `data/evaluation/benchmarks/phase-10b-multiquery-diagnostic.json`
- `docs/phase-10b-multiquery-diagnostic.md`

Full frozen benchmark:

```bash
python -m rag_research_assistant.experiments.phase10_multiquery_eval --benchmark
```

This creates:

- `data/evaluation/benchmarks/phase-10b-multiquery-results.json`
- `docs/phase-10b-multiquery-results.md`

The full run makes one LLM generation call per question and multiple retrieval
calls, so it is intentionally a user-run validation checkpoint rather than an
automated test or CI job.

## What remains out of scope

Phase 10B does not add HyDE, decomposition, parent-child retrieval, contextual
compression, metadata routing, dynamic top-k, confidence thresholds, answer
generation, or production integration. Adoption depends on measured ranking
benefit, regressions, semantic drift, and added latency—not on architectural
novelty.
