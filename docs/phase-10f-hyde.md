# Phase 10F: Hypothetical Document Embeddings

Phase 10F is an isolated retrieval experiment. It does not change the production CLI, FastAPI application, `RAGApplication`, LangGraph workflow, chunking, models, indexes, or generation path.

## HyDE versus query rewriting

Query rewriting asks for another query. HyDE asks Qwen for a short answer-like passage that could plausibly occur in a relevant paper, embeds that passage with the existing `multi-qa-MiniLM-L6-cos-v1` model, and uses the vector as a semantic probe. The generated passage may be wrong and is never evidence.

The experiment compares three controlled conditions:

```text
baseline: original question -> frozen hybrid retrieval -> cross-encoder

HyDE-only: question -> hypothetical passage -> existing embedder
          -> existing legacy Qdrant index -> cross-encoder(original question, real chunk)

fused: frozen baseline depth-20 ranking + HyDE dense depth-20 ranking
     -> equal RRF (k=60) -> cross-encoder(original question, real chunk)
```

All final results are real legacy chunks. Stable chunk IDs provide RRF deduplication. The original question—not the hypothetical passage—is always the cross-encoder relevance reference.

## Probe generation and length

The prompt requests only a 2–5 sentence, 60–150 word technical passage. It explicitly preserves names, methods, acronyms, models, and datasets while prohibiting citations, JSON, labels, and meta-commentary. Generation uses `qwen3.5:4b` through the existing Ollama abstraction at temperature 0.0. Because the current generator does not expose a token limit, the normalizer collapses formatting and caps the probe at 150 whitespace-delimited words. Truncation is recorded.

## Safe fallback

Generation, normalization, embedding, dense-search, or reranking failure cannot replace the baseline with an empty or partial branch. Both experimental final conditions reproduce the frozen baseline top results and record the error, fallback flag, timings, and any hypothetical text that was successfully produced.

## Drift diagnostics

Diagnostics record protected-term preservation, introduced title-case entities, introduced numerical details, suspicious parenthetical expansions, citation-like output, query-term coverage, and lexical Jaccard overlap. These are intentionally lightweight audit signals. They do not reject or alter a probe because HyDE is expected to generate content not present in the question, and its cross-encoder scores are not calibrated confidence values.

For the ten unanswerable questions, retrieval metrics remain unscored. The artifact retains each hypothetical document, all rankings, top-result changes, and overlap with baseline so plausible but unsupported probes can be reviewed directly.

## Manual validation

Run from the repository root:

```bash
.venv/bin/python -m rag_research_assistant.experiments.phase10_hyde_eval --smoke
.venv/bin/python -m rag_research_assistant.experiments.phase10_hyde_eval --diagnostic 5
.venv/bin/python -m rag_research_assistant.experiments.phase10_hyde_eval --benchmark
```

The diagnostic writes:

- `data/evaluation/benchmarks/phase-10f-hyde-diagnostic.json`
- `docs/phase-10f-hyde-diagnostic.md`

The full run writes:

- `data/evaluation/benchmarks/phase-10f-hyde-results.json`
- `docs/phase-10f-hyde-results.md`

The full benchmark performs one local Qwen generation per question and should be run manually. Afterward, inspect aggregate top-k metrics and individual rescues, found-to-not-found regressions, exact/numeric drift, entity ambiguity, unanswerable probes, fusion safeguards, and latency. No recommendation should be made before that review.

## Final decision

The 100-question benchmark rejects HyDE as a production or operationally optional path. HyDE-only slightly improved Hit@3 but reduced Hit@5, rescuing four baseline misses while losing five baseline hits. It was strongest on difficult-distractor, section-detail, and semantic-paraphrase questions, but substantially harmed exact-number questions and weakened comparison and exact-terminology cases.

Equal baseline+HyDE fusion was safe but inert: all 90 scored questions retained exactly the same first-relevant rank as baseline. It changed some lower-ranked chunks but produced no relevance improvement. Mean fused latency was 4.42 seconds versus 0.20 seconds for baseline, with hypothetical generation accounting for about 92% of fused time.

The hypothetical documents explain the volatility. False details sometimes acted as useful semantic anchors, but they also caused losses: invented DPR margins, INSTRUCTOR task counts, E5 dataset counts, BEIR model families, and an incorrect docT5query architecture displaced real evidence. All ten unanswerable questions received plausible answer-like inventions, and HyDE-only changed the top result for eight of them. The hypothetical text remained isolated from evidence, but this behavior confirms that HyDE probes cannot be trusted as knowledge.

Retain the code only as a reproducible learning experiment. Do not integrate it into production and do not test a smaller generator yet: reducing latency does not address the lack of fused retrieval gain. A future revisit would first need a new, predeclared fusion or routing hypothesis and should not tune on these results.
