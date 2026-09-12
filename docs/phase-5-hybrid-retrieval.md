# Phase 5: hybrid retrieval and reranking

## Why this phase exists

The Phase 4 dense baseline found labelled evidence in the top five for every
scored question, but placed it first only 56.2% of the time. An LLM sees all
retrieved context, yet earlier evidence is easier to notice and a lower `top-k`
budget may exclude low-ranked evidence entirely. Hit@1 is therefore an important
ranking signal for RAG even though it is not answer accuracy.

Phase 5 keeps the same 20 questions and source labels. This isolates the effect
of retrieval strategy instead of moving the benchmark target.

## Dense and lexical retrieval

Dense retrieval is the existing bi-encoder path. It encodes a question and each
chunk separately in the same learned vector space, computes cosine similarity,
and sorts the chunks. Separate encoding makes stored document vectors reusable
and lets paraphrases match even when their exact words differ.

BM25 is lexical: it only sees tokens. `bm25.py` case-folds text, extracts Unicode
word tokens, and records term frequency (TF), document frequency (DF), document
length, and average document length. For query term `t` and document `d`:

```text
IDF(t) = log(1 + (N - DF(t) + 0.5) / (DF(t) + 0.5))

BM25(t, d) = IDF(t) *
             TF(t,d) * (k1 + 1) /
             (TF(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

The defaults are `k1=1.5` and `b=0.75`. TF rewards repeated query terms but
saturates, IDF gives rare terms more weight, and length normalization prevents a
long chunk from winning merely because it contains more words. Exact identifiers,
names, numbers, and phrases often favor lexical search. Synonyms and paraphrases
often favor dense retrieval. This tokenizer deliberately has no stemming,
lemmatization, stop-word list, or domain-specific rules.

## Reciprocal Rank Fusion

Cosine and BM25 scores cannot be added meaningfully: cosine measures vector
direction while BM25 is an unbounded formula over corpus statistics. Their raw
scales are unrelated. Reciprocal Rank Fusion (RRF) uses only rank positions:

```text
RRF(chunk) = 1 / (rrf_k + dense_rank)
           + 1 / (rrf_k + bm25_rank)
```

This project uses `rrf_k=60`, a conventional smoothing constant. It is not
`top-k`: `rrf_k` controls how quickly rank contribution decays; final top-k is
how many results the caller receives. Each base retriever returns 20 candidates,
RRF merges duplicate chunk IDs and rewards chunks found by both, then returns the
requested number. Stable input order resolves exact ties.

## Candidate retrieval and cross-encoder reranking

The optional `cross-encoder/ms-marco-MiniLM-L6-v2` reranker receives each
`(question, chunk text)` pair together and returns a pairwise relevance score.
Joint attention can judge relationships that a separately encoded bi-encoder
compresses away. It is also much slower: 20 model forward passes are batched for
each query. That is why it reranks only the hybrid candidate pool and never
embeds the full corpus.

The selected model is a compact six-layer MiniLM cross-encoder, approximately
91 MB, with a 512-token input limit. It fits the target M1 Pro/16 GB laptop, but
was trained on MS MARCO passage ranking rather than these research papers. Scores
are model logits, not calibrated probabilities. Long question/chunk pairs are
truncated, and scientific terminology can remain difficult.

Run `rag-research-assistant reranker-download` once while online. Search,
evaluation, and answering load cached files only, so later use works offline.
Tests inject fake pair scorers and never contact Hugging Face.

## One interface, four score meanings

`retrievers.py` defines one `search(query, top_k)` behavior used by the CLI,
evaluation, and RAG orchestration:

| Strategy | Primary score | Meaning |
| --- | --- | --- |
| Dense | cosine | semantic vector similarity |
| BM25 | BM25 | lexical term relevance in this corpus |
| Hybrid | RRF | agreement and position across rankings |
| Hybrid + rerank | cross-encoder | pairwise model relevance |

None of these values is a probability. Never compare their raw magnitudes across
strategies. Metadata remains on the original `Chunk`, so every strategy returns
the same filename, page, chunk ID, and text contract.

## Commands

```bash
rag-research-assistant search "question" --retriever dense
rag-research-assistant search "question" --retriever bm25
rag-research-assistant search "question" --retriever hybrid
rag-research-assistant search "question" --retriever hybrid --rerank

rag-research-assistant evaluate --retriever hybrid --rerank --verbose
rag-research-assistant ask "question" --retriever hybrid --rerank --show-context
rag-research-assistant compare
```

`compare` runs dense, BM25, hybrid, and hybrid + reranker against the exact same
loaded examples. It writes aggregate and per-question changes to the ignored
`data/evaluation/results/comparison.json` file.

## Measured results

The corpus contained 209 chunks. The benchmark had 20 questions: 16 scored for
retrieval and 4 reserved for answerability/generation checks.

| Strategy | Hit@1 | Hit@3 | Hit@5 | Mean first rank | Recall@1 | Recall@3 | Recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 56.2% | 68.8% | 100.0% | 2.12 | 46.9% | 59.4% | 93.8% |
| BM25 | 50.0% | 81.2% | 93.8% | 1.87 | 40.6% | 71.9% | 87.5% |
| Hybrid RRF | 68.8% | 93.8% | 93.8% | 1.33 | 56.2% | 84.4% | 87.5% |
| Hybrid + reranker | 75.0% | 100.0% | 100.0% | 1.38 | 59.4% | 90.6% | 90.6% |

BM25 did not beat dense at rank 1, but improved Hit@3 by 12.5 percentage points,
evidence that exact lexical signals helped several questions. Hybrid improved
Hit@1 by 12.5 points and Hit@3 by 25 points over dense, while temporarily losing
one top-five hit. Reranking raised Hit@1 by another 6.2 points, produced perfect
Hit@3 and Hit@5 on this small set, and had no retrieval failures.

Dense to hybrid changes:

- Improved: `rag_sequence_vs_token` 2→1, `rag_update_knowledge` 4→2,
  `rag_corpus_chunks` 5→2, `dpr_bm25_results` 4→3, and
  `dpr_lexical_semantic` 4→1.
- Degraded: `sgpt_pooling_training` 4→outside top 5.

Hybrid to reranked changes:

- Improved: `rag_update_knowledge` 2→1 and `sgpt_pooling_training` outside top
  5→2.
- Degraded: `rag_corpus_chunks` 2→3.

On this machine, a cached BM25-only 20-question evaluation took about 0.21
seconds. Cached hybrid + reranking took about 9.87 seconds. These wall-clock
numbers are environment-specific, but the order reflects the architecture:
BM25 is cheap CPU arithmetic, dense loads one bi-encoder, hybrid runs both, and
the cross-encoder jointly processes every candidate pair.

The full strongest-strategy run with local `qwen3.5:4b`, temperature 0.1, and
`think=false` took about 174 seconds. Retrieval remained 75%/100%/100%, all four
unanswerable questions had recognized refusals, and there were no generation
errors. `rag_update_knowledge` emitted one invalid `[6]` reference with only five
contexts. That is a generation failure, not a retrieval-label problem.

## What to understand

Hybrid retrieval is not automatically better: candidate depth and rank fusion
can remove a dense top-five result. A cross-encoder can restore or improve a
candidate only if first-stage retrieval included it in the 20-item pool. Better
retrieval also does not guarantee valid prose or citations. The fixed evaluation
set exposes all three boundaries—candidate recall, ranking quality, and local
generation behavior—without a framework obscuring them.
