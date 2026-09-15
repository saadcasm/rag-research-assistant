# Phase 10E: extractive contextual compression

Phase 10E is an isolated, post-retrieval experiment. The production stack remains legacy chunks → Qdrant dense retrieval + BM25 → RRF → cross-encoder reranking. Retrieval still returns the same top-five `SearchResult` objects in the same order.

## Design

Each retrieved chunk is conservatively split at sentence and paragraph boundaries, then tiny sentences are grouped into short passages (minimum target 80 characters, normal target 360 characters). The splitter preserves exact character offsets and protects common scholarly abbreviations, initialisms, and decimal numbers. It does not rewrite or summarize source text.

The already-loaded retrieval cross-encoder batch-scores every `(question, segment)` pair once per question. These scores are uncalibrated ranking values. The same scored segments are reused for all controlled conditions:

- no compression;
- per-chunk compression at 75%, 50%, and 30% character budgets;
- global compression at 75%, 50%, and 30% character budgets.

Per-chunk selection retains at least one segment from every non-empty source. Global selection retains at least one segment overall and may omit a source; omissions are recorded. Selection follows relevance score, but reconstruction restores retrieval-source order and within-source segment order.

Every selected segment records its source chunk, document, page span, section, source rank, segment index, exact source offsets, original text, and relevance score. Automated integrity checks ensure those offsets still address the exact source substring.

## Evaluation boundary

Compression does not alter chunk retrieval rank, so Hit@k is intentionally not presented as a compression metric. The experiment measures characters and segments removed, p50/p95 compression ratios, safeguard use, omitted sources, provenance failures, and segmentation/scoring/selection latency.

The Phase 7.5 dataset contains verified supporting passages. The runner measures normalized whole-passage containment and evidence-term coverage against those passages. Term coverage is explicitly a proxy: it is neither an answer-span label nor a generation-quality score, and a zero can reflect an upstream retrieval miss. Query-term preservation is a second, weaker proxy.

## Manual validation checkpoint

Run these commands from the repository root after the deterministic test suite passes:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_contextual_compression_eval --smoke
.venv/bin/python -m rag_research_assistant.experiments.phase10_contextual_compression_eval --diagnostic 5
.venv/bin/python -m rag_research_assistant.experiments.phase10_contextual_compression_eval --benchmark
```

The diagnostic writes:

- `data/evaluation/benchmarks/phase-10e-contextual-compression-diagnostic.json`
- `docs/phase-10e-contextual-compression-diagnostic.md`

The full benchmark writes:

- `data/evaluation/benchmarks/phase-10e-contextual-compression-results.json`
- `docs/phase-10e-contextual-compression-results.md`

After inspecting those results, choose at most one or two sensible compression candidates. Only then should a small controlled generation comparison be added and manually run. It must hold retrieval, prompt, Qwen model, and decoding settings constant and vary only the supplied context.

## Compression-only result and generation candidate

The 100-question run selected global 75% compression for the first generation study. It removed 26.4% of source text while retaining 97.7% of the measurable evidence-term signal, preserved 25 of the 26 supporting passages that were exactly contained by baseline retrieval, omitted no sources, triggered no minimum safeguards, and produced zero provenance failures. Its average compression latency was approximately 64 ms, almost entirely batched cross-encoder scoring.

Global 50% removed 51.2% of source text and retained 95.2% of the mean evidence-term signal, but individual retention fell as low as 41%, and nine retrieved sources were omitted. Both 30% configurations showed clearer evidence loss. Global 75% is therefore the safer candidate for a first causal generation comparison; this does not establish it as a production default.

The fixed diagnostic covers all eight answerable categories plus two unanswerable questions. It deliberately includes known cross-page and semantic-paraphrase retention stress cases rather than selecting only favorable examples. Run:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_contextual_compression_eval --generation-diagnostic
```

Expected outputs:

- `data/evaluation/benchmarks/phase-10e-compression-generation-diagnostic.json`
- `docs/phase-10e-compression-generation-diagnostic.md`

This performs 20 local generations: original and global-75% context for each of 10 questions. Retrieval, source order, generator, prompt template, temperature, and decoding behavior are held constant. The resulting answers still require manual comparison because citation validity, refusal detection, and evidence-term overlap do not prove factual correctness or groundedness.

## Final decision

The generation diagnostic preserved seven of eight answerable results and both unanswerable abstentions, but degraded the BEIR/BRIGHT multi-source case from a correct insufficiency response to an unsupported answer. The cited source existed but did not entail the claim. Consequently, contextual compression remains experimental and no production application, CLI, API, generation, or workflow path imports it.

Global 75% is the only configuration retained for possible future study. The 50% and 30% conditions are rejected at this stage because of their evidence-retention tails and source omissions. Before reconsidering integration, a larger counterbalanced generation study should emphasize multi-source and cross-page questions and manually verify claim-to-citation entailment. LLM-based compression should wait: adding generative rewriting before solving this groundedness failure would weaken auditability and add latency.
