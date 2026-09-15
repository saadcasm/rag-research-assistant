# Phase 10A: corpus-grounded LLM query rewriting

## Experiment

The frozen Phase 7.5 retriever is compared with the same retriever preceded by one corpus-grounded LLM rewrite. The original question remains separate and is never replaced for answer generation.

Dataset: `data/evaluation/questions-phase-7.5.jsonl`
Rewrite model: `qwen3.5:4b`
Preliminary/final top-k: 5/5
Retrieval: `hybrid+rerank` using `qdrant`

## Aggregate retrieval metrics

| Metric | Baseline | Rewritten |
|---|---:|---:|
| Hit@1 | 0.4556 | 0.4333 |
| Hit@3 | 0.7222 | 0.7444 |
| Hit@5 | 0.8444 | 0.8222 |
| Recall@1 | 0.4500 | 0.4278 |
| Recall@3 | 0.7167 | 0.7389 |
| Recall@5 | 0.8222 | 0.8056 |
| Mean first relevant rank | 1.8947 | 1.7703 |

## Rewrite behavior and latency

- Questions: 100 (90 scored, 10 unanswerable)
- Changed: 36 (36.0%); unchanged: 64
- Improved/degraded/unchanged: 7/8/75
- Failures/fallbacks: 0/0; invalid or empty: 0
- Identity warnings: 3
- Average baseline retrieval: 0.352s
- Average rewrite: 5.357s (p50 5.260s, p95 6.149s)
- Average rewrite-condition total: 6.093s

## Biggest improvements

| ID | Original → rewritten | Rank change | Identity loss |
|---|---|---:|---|
| `p75_deepimpact_expansion` | Why does DeepImpact use DocT5Query before estimating document-term impacts? → Why does DeepImpact use DocT5Query to enrich documents with new terms likely to occur in queries for which the document is relevant? | not found → 2 | document-term |
| `p75_gtr_scaling` | What did GTR keep fixed while scaling up the dual encoder model? → What did the fixed-size bottleneck layer keep fixed while scaling up the dual encoder model in GTR? | not found → 2 | none |
| `p75_hyde_false_details` | How does HyDE limit the effect of false details in its generated hypothetical document? → How does HyDE limit the effect of false details in its generated hypothetical document by using an unsupervised contrastive encoder's dense bottleneck to filter out hallucinated information? | 5 → 2 | none |
| `p75_lost_middle_position` | Where in a long input did models generally use relevant information most effectively? → Where do language models use relevant information most effectively in long input contexts? | 4 → 2 | none |
| `p75_multihop_source` | What underlying knowledge base was used to construct MultiHop-RAG? → What underlying knowledge base was used to construct MultiHop-RAG, specifically an English news article dataset? | 3 → 1 | none |
| `p75_ance_global_negatives` | Why does ANCE select negatives from the whole corpus rather than relying only on local in-batch samples? → Why does Approximate nearest neighbor Negative Contrastive Estimation (ANCE) select negatives from the whole corpus rather than relying only on local in-batch samples? | 4 → 3 | none |
| `p75_doct5query_direction` | In docT5query, what is predicted from each document and then appended for retrieval? → In DocT5Query, what is predicted from each document by the sequence-to-sequence model and then appended to the document for retrieval? | 2 → 1 | none |

## Biggest regressions

| ID | Original → rewritten | Rank change | Identity loss |
|---|---|---:|---|
| `p75_bert_reranker_gain` | What relative MRR@10 improvement did the BERT passage reranker report over the previous state of the art? → What relative MRR@10 improvement did the BERT passage reranker report over IR-NET? | 1 → not found | none |
| `p75_monot5_reformulation` | How does monoT5 cast document ranking as a sequence-to-sequence task? → How does monoT5 cast document ranking as a sequence-to-sequence task using "true" and "false" target tokens? | 3 → not found | none |
| `p75_raptor_tree` | How does RAPTOR construct its retrieval tree from source chunks? → How does RAPTOR construct its retrieval tree from source chunks using SBERT embeddings, clustering, and LLM summarization? | 3 → not found | none |
| `p75_raptor_levels` | Why can RAPTOR answer questions needing both fine detail and holistic document understanding? → Why can RAPTOR answer questions needing both fine detail and holistic document understanding by utilizing a recursive tree structure that allows it to choose nodes from different tree layers matching the question's detail level? | 5 → not found | none |
| `p75_m3_functions` | Which retrieval representation families does M3-Embedding aim to support in one model? → Which retrieval representation families does M3-Embedding aim to support in one model: dense retrieval, sparse retrieval, and multi-vector retrieval? | 1 → 2 | none |
| `p75_sgpt_scale` | How large was the SGPT model that improved over prior sentence embeddings on BEIR? → How large was the SGPT-BE-5.8B model that improved over prior sentence embeddings on BEIR? | 1 → 2 | none |
| `p75_query2doc_both_retrievers` | Is query2doc limited to lexical retrieval, or can it improve dense retrievers too? → Is query2doc limited to lexical retrieval, or can it improve dense retrievers like E5 and SimLM? | 1 → 2 | none |
| `p75_selfrag_adaptive` | What problem arises when standard RAG always retrieves a fixed number of passages? → What problem arises when standard RAG always retrieves a fixed number of passages regardless of retrieval necessity? | 2 → 3 | none |

## Semantic-drift diagnostics

Missing quoted phrases, acronyms, and model-like tokens are warnings, not relevance thresholds.

- `p75_deepimpact_expansion` lost document-term: `Why does DeepImpact use DocT5Query to enrich documents with new terms likely to occur in queries for which the document is relevant?`
- `p75_compare_hyde_query2doc` lost pseudo-text: `How do HyDE (Gao et al., 2022) and query2doc use generated pseudo-documents differently before retrieval?`
- `p75_unanswerable_multilingual_accuracy` lost English-only: `What retrieval accuracy does the multilingual mContriever model achieve on Arabic research questions?`

## Failures and fallbacks

- No rewrite failures.

## Recommendation

Reject as the default path; regressions outweigh the measured benefit.

This experiment does not use cross-encoder scores as probabilities and introduces no confidence threshold. Per-question records are in the companion JSON report.
