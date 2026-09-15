# Phase 10A: corpus-grounded LLM query rewriting

Phase 10A is an isolated retrieval experiment. It does not change the CLI,
FastAPI, `RAGApplication`, answer prompt, or default retrieval path. Phase 9's
deterministic LangGraph rewriter and its hard-coded educational mappings are
also unchanged.

## Question being tested

Can `qwen3.5:4b` use a few preliminary corpus chunks to create one better
retrieval query without losing the user's intent or document identity?

```text
baseline
original question -> frozen retriever -> ranked chunks

rewrite condition
original question -> frozen retriever -> top five preliminary chunks
                  -> qwen3.5:4b query rewrite
                  -> same frozen retriever -> ranked chunks
```

The frozen retriever is the Phase 7.5 legacy stack: exact local Qdrant dense
retrieval plus BM25, RRF with `k=60`, candidate depth 20, and the existing
cross-encoder reranker. The experiment calls the shared
`default_retrieval_config()` and does not reconstruct those components.

The first baseline retrieval results are reused as preliminary evidence. This
is valid because both conditions use the same deterministic stack and avoids a
redundant third retrieval. Rewrite-condition latency still includes preliminary
retrieval + rewrite generation + final retrieval.

## Corpus-grounded prompt

The rewriter receives only:

1. the original question;
2. the top preliminary chunks, each with filename, page span, chunk ID, and a
   bounded text excerpt.

Its prompt says to produce exactly one retrieval query, never an answer. It
must preserve intent, entities, paper/method/model/dataset names, quoted terms,
and acronyms. It may expand an acronym only when evidence resolves it, should
include both acronym and expansion, must not invent facts, and should return an
already-strong query unchanged.

There is no DPR, RAG, ColBERT, or other title/acronym dictionary in Phase 10A.
Any expansion must come from preliminary text. The original user question is
stored separately and would remain the question used for eventual answer
generation; the rewrite is retrieval-only.

Responses are parsed as one plain non-empty line. A small tolerance handles
surrounding quotes, a `Rewritten query:` prefix, or one Markdown fence. Empty,
multi-line, oversized, or exceptional responses fall back to the exact original
query and record the error. One failure cannot abort the benchmark.

## Semantic-drift diagnostics

This phase deliberately avoids an NER framework. A lightweight diagnostic
extracts:

- quoted phrases;
- uppercase or letter-number acronyms such as `DPR`, `BM25`, and `E5`;
- model-like tokens containing separators, such as `MiniLM-L6-v2`.

Missing terms are warnings in the report, not blockers or score thresholds.
Cross-encoder scores remain ranking signals and are never treated as calibrated
probabilities.

The diagnostic cannot detect every semantic change. In particular, it can show
that `ANCE` survived a rewrite but cannot prove that the expansion attached to
ANCE is factually correct. Human regression review remains necessary.

## Commands

Start Ollama with `qwen3.5:4b`, then inspect one query:

```bash
python -m rag_research_assistant.experiments.phase10_rewrite_eval --smoke
```

The smoke output displays `ORIGINAL QUERY`, `PRELIMINARY RESULTS`,
`REWRITTEN QUERY`, and `FINAL RESULTS`, including source/page/chunk provenance.
The question can be replaced without changing code:

```bash
python -m rag_research_assistant.experiments.phase10_rewrite_eval \
  --smoke \
  --question "How does DPR represent passages?"
```

Run the frozen 100-question benchmark:

```bash
python -m rag_research_assistant.experiments.phase10_rewrite_eval \
  --dataset data/evaluation/questions-phase-7.5.jsonl \
  --preliminary-top-k 5 \
  --final-top-k 5 \
  --temperature 0 \
  --json-output data/evaluation/benchmarks/phase-10a-rewrite-results.json \
  --markdown-output docs/phase-10a-query-rewriting-results.md
```

The JSON records every original/rewrite query, expected source, preliminary
source preview, rank movement, latency, fallback, and identity warning. It also
stores hashes of the frozen dataset and rewrite prompt.

## Measured result

The full run completed with 100 questions: 90 retrieval-scored and 10
unanswerable. There were no model failures, invalid responses, or fallbacks.

| Metric | Baseline | Rewrite |
|---|---:|---:|
| Hit@1 | 45.56% | 43.33% |
| Hit@3 | 72.22% | 74.44% |
| Hit@5 | 84.44% | 82.22% |
| Recall@1 | 45.00% | 42.78% |
| Recall@3 | 71.67% | 73.89% |
| Recall@5 | 82.22% | 80.56% |
| Mean first relevant rank among found questions | 1.895 | 1.770 |

The model changed 36 queries. Across the 90 scored questions, 7 improved, 8
degraded, and 75 were unchanged. Mean-first-rank improved partly because fewer
questions were found within the evaluated top five; it must not be interpreted
without Hit@5.

Average baseline retrieval took 0.352 seconds. Rewriting took 5.357 seconds on
average (p50 5.260, p95 6.149), and the full rewrite condition averaged 6.093
seconds. It therefore added roughly 5.74 seconds while reducing Hit@1, Hit@5,
Recall@1, and Recall@5.

Representative improvements included DeepImpact (not found to rank 2), GTR
(not found to rank 2), HyDE (rank 5 to 2), and MultiHop-RAG (rank 3 to 1).
Representative regressions included the BERT reranker, monoT5, and two RAPTOR
questions falling out of the top five.

The DPR smoke question was returned unchanged. Preliminary top-five evidence
contained DPR method details but not the title-page title/authors, so the model
correctly avoided inventing the desired expansion; DPR page 1 did not become
easier to retrieve. This shows a recall ceiling in corpus-grounded rewriting:
the model cannot safely use evidence that preliminary retrieval did not expose.

Three protected-token warnings were surfaced. More importantly, the ANCE
rewrite expanded the acronym as “Approximate nearest neighbor Negative
Contrastive Estimation”; the source paper calls it “Approximate nearest neighbor
Negative Contrastive Learning.” It improved rank 4 to 3 while introducing a
factual error. This is semantic drift that aggregate retrieval metrics cannot
detect.

See [the generated result report](phase-10a-query-rewriting-results.md) and the
machine-readable companion at
`data/evaluation/benchmarks/phase-10a-rewrite-results.json`.

## Recommendation and trade-offs

Reject this implementation as a default production stage. It is operationally
reliable and occasionally helpful, but the measured net retrieval result is
negative, latency is much higher, DPR's known failure was not fixed, and corpus
grounding did not eliminate factual drift.

Useful lessons remain:

- preliminary evidence provides inspectable grounding;
- a typed result and fallback make LLM transformation auditable;
- preserving the original question prevents answer-intent corruption;
- selective “unchanged” behavior reduces, but does not remove, regressions;
- rank gains do not validate the factual content of a rewrite;
- missing preliminary evidence limits what a grounded rewriter can resolve.

Phase 10B would logically test multi-query retrieval with explicit fusion and
the same regression accounting. It should retain the original query alongside
generated variants so one poor rewrite cannot completely replace the baseline.
That work is not implemented here.
