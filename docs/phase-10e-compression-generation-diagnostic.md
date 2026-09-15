# Phase 10E: compression generation diagnostic

This controlled diagnostic compares the exact same retrieved source ranking and grounded prompt with original context versus global 75% extractive compression. It is a small qualitative study, not a statistically powered answer-quality benchmark.

- Questions: 10 (8 answerable; 2 unanswerable)
- Source characters, original/compressed: 50168/37038 (ratio 0.7383)
- Generation time, original/compressed: 82.682s/64.946s
- Generation errors, original/compressed: 0/0
- Invalid citations, original/compressed: 0/0
- Unanswerable refusals, original/compressed: 2/2
- Provenance failures: 0

## Per-question answers

### p75_atlas_few_shot

**Question:** What result shows that Atlas can learn a knowledge-intensive task from very few examples?

Category: exact_number; answerability: answerable; actual compression ratio: 0.7362

**Original answer**

Atlas achieves over 42% accuracy on the Natural Questions task using only 64 training examples, which outperforms a model with 540 billion parameters by 3% despite having 50 times fewer parameters [1]. Additionally, in a 15-shot setting for fact checking, Atlas scores 56.2%, outperforming Gopher by 5.1 points [3].

**Compressed answer**

Atlas achieves over **42% accuracy** on the Natural Questions task using only **64 training examples**, which outperforms a 540B parameter model by 3% despite having 50 times fewer parameters [1]. Additionally, in a **15-shot setting** (with 5 examples sampled from each class), Atlas scores **56.2%** on fact-checking tasks, outperforming the Gopher model by 5.1 points [3].

### p75_coil_index

**Question:** How does COIL retain inverted-index efficiency while using contextual representations?

Category: difficult_distractor; answerability: answerable; actual compression ratio: 0.7323

**Original answer**

COIL retains inverted-index efficiency by storing pre-computed contextualized token representations in inverted lists during an offline processing phase. Specifically, documents are encoded offline into token and CLS vectors, and for each unique token $t$ in the vocabulary, its contextualized vectors from all document mentions are collected to build a token's contextualized inverted list ($I_t$) [5]. At search time, the system performs efficient exact match by using each query token to look up its corresponding inverted list and computing vector similarity with the stored document vectors as matching scores [2]. This approach combines the computational efficiency of exact match systems (via the inverted list structure) with the semantic representation power of deep language models [1].

**Compressed answer**

COIL retains inverted-index efficiency by storing pre-computed contextualized token representations in inverted lists built offline. During search, the system uses each query token to look up its corresponding inverted list and computes matching scores via vector similarity with the document vectors stored therein [2]. This approach allows COIL to focus only on documents containing overlapping terms with the query, a process that remains efficient due to the inverted list index structure [3].

### p75_doct5query_direction

**Question:** In docT5query, what is predicted from each document and then appended for retrieval?

Category: exact_terminology; answerability: answerable; actual compression ratio: 0.7317

**Original answer**

In DocT5Query, a sequence-to-sequence model predicts **queries** (specifically, possible questions that the document might answer) from each document, and these predicted queries are **appended to the documents** before indexing to enrich their representation [2]. This process is described as "document expansion," where the document representation is enriched while the query representation remains static [2]. The method uses datasets consisting of pairs of query and relevant documents to train this model [5].

**Compressed answer**

In DocT5Query, a sequence-to-sequence model predicts queries that are potentially relevant to a given document [1]. These predicted queries are then appended to the documents before indexing to enrich their representation [2]. Specifically, the model generates possible questions that the document might answer [2], and these synthetic queries are added to the original document for lexical search [3].

### p75_e5_supervision

**Question:** What supervision source is used to train E5's contrastive text embeddings?

Category: section_detail; answerability: answerable; actual compression ratio: 0.7452

**Original answer**

E5's contrastive text embeddings are trained using weak supervision signals derived from a curated large-scale text pair dataset called CCPairs [1]. Unlike methods that rely on supervised datasets for task-specific fine-tuning, E5 achieves strong performance across various tasks without requiring labeled data [3]. The model employs a contrastive learning framework to learn transferable text representations [4], utilizing these weakly-supervised signals to enhance sequence-level representations from text pairs [4].

**Compressed answer**

E5's contrastive text embeddings are trained using weak supervision signals derived from a curated large-scale text pair dataset called CCPairs [1].

### p75_trec_dl_regime

**Question:** What training-data regime does the TREC 2020 Deep Learning Track study?

Category: semantic_paraphrase; answerability: answerable; actual compression ratio: 0.7472

**Original answer**

The TREC 2020 Deep Learning Track studies the **large training data regime** [1]. This is evidenced by the track's use of hundreds of thousands of human-labeled training queries for both document and passage retrieval tasks [1], specifically providing large reusable training datasets with associated large-scale click datasets to train deep learning and traditional ranking methods under these conditions [2].

**Compressed answer**

The TREC 2020 Deep Learning Track studies the **large training data regime** [1]. This is evidenced by the provision of large reusable training datasets containing hundreds of thousands of records for both document and passage retrieval tasks [3], alongside a large click dataset (ORCAS) generated from Bing search engine logs [4]. The track's primary goal was to evaluate ranking methods, particularly those with BERT-style pretraining, specifically within this context where large data is available [1].

### p75_compare_beir_bright

**Question:** What retrieval capability does BRIGHT target that is not the main focus of BEIR's heterogeneous zero-shot evaluation?

Category: multi_source; answerability: answerable; actual compression ratio: 0.7366

**Original answer**

Based on the provided context, there is no information regarding a benchmark or system named **BRIGHT**, nor does it describe any specific retrieval capability that BRIGHT targets which differs from BEIR's focus. The available evidence only details the **BEIR** benchmark [1][2], which evaluates diverse retrieval tasks (including lexical, sparse, dense, late-interaction, and re-ranking) across 18 datasets to assess zero-shot generalization [1]. It also mentions related benchmarks like **MultiReQA**, **KILT**, **STS**, **USEB**, and **MTEB** [3][4], but does not mention BRIGHT.

Therefore, the supplied evidence is insufficient to answer the question.

**Compressed answer**

Based on the provided context, **BRIGHT** targets **instruction following generative LLMs**, whereas **BEIR** focuses on evaluating **zero-shot information retrieval models** (including lexical, sparse, dense, late-interaction, and re-ranking architectures) across diverse datasets without assuming access to test-time corpora for training [5].

The context notes that while BEIR defines zero-shot dense retrieval tasks empirically for the neural retrieval community, BRIGHT studies building effective dense retrieval systems specifically for generative LLMs in a zero-shot setting where relevance labels and test-time corpora are not assumed [5]. Additionally, other benchmarks mentioned (like MultiReQA and KILT) focus on smaller corpora or knowledge-intensive tasks, which BEIR addresses with 18 diverse datasets [3], but the specific target of BRIGHT is generative LLM instruction following [5].

### p75_beir_tradeoff

**Question:** Which BEIR model families performed best on average zero-shot, and what was their drawback?

Category: comparison; answerability: answerable; actual compression ratio: 0.7390

**Original answer**

Based on the provided evidence, **re-ranking** and **late-interaction** based models achieved the best average zero-shot performance on the BEIR benchmark. However, their primary drawback is that they incur **high computational costs**. In contrast, dense and sparse-retrieval models are more computationally efficient but often underperform these approaches in terms of generalization capabilities [1].

**Compressed answer**

Based on the provided evidence, **re-ranking** and **late-interaction** based models achieved the best average zero-shot performance on the BEIR benchmark. However, their primary drawback is that they incur **high computational costs**. In contrast, dense and sparse-retrieval models are more computationally efficient but often underperform these approaches [1].

### p75_colbert_precompute

**Question:** Why can ColBERT precompute document representations even though it retains token-level matching?

Category: cross_page_context; answerability: answerable; actual compression ratio: 0.7478

**Original answer**

ColBERT can precompute document representations because it employs a "late interaction architecture" that independently encodes the query and document using BERT, followed by a cheap yet powerful interaction step that models fine-grained similarity [1]. This approach allows ColBERT to leverage the expressiveness of deep language models while retaining the ability to pre-compute document representations offline, which considerably speeds up query processing [1]. By judiciously delaying the query–document interaction, the model exploits deep LM-based representations while shifting the cost of encoding documents offline and amortizing the cost of encoding the query once across all ranked documents [2].

The specific mechanism enabling this is the retention of token-level matching via a MaxSim operator. Every query embedding interacts with all document embeddings through this operator, which computes maximum similarity (e.g., cosine similarity) for each term, and these scalar outputs are summed across query terms [2]. This paradigm allows ColBERT to perform end-to-end retrieval directly from a large document collection using vector-similarity indexes, rather than relying solely on re-ranking the output of term-based retrieval [2].

**Compressed answer**

ColBERT can precompute document representations because it leverages a mechanism that delays the query–document interaction while retaining fine-grained, token-level matching [1]. This approach allows the model to exploit deep language model (LM) representations for documents by computing them offline, which speeds up query processing and reduces encoding costs [1]. By judiciously delaying the interaction between the query and these pre-computed document representations, ColBERT can amortize the cost of encoding the query once across all ranked documents [2]. Additionally, this paradigm enables the use of vector-similarity indexes to retrieve top-k results directly from large collections, as the heavy lifting of representing documents is completed during indexing rather than at query time [2].

### p75_unanswerable_qdrant_hnsw_m

**Question:** What exact HNSW M and ef_construct values should this project use for its 52-paper Qdrant index?

Category: unanswerable; answerability: unanswerable; actual compression ratio: 0.7434

**Original answer**

The supplied evidence is insufficient to answer the question. While the context provides specific parameter values (M=16, efConstruction=500 or 40) used in experiments on a 200M SIFT dataset [5], it does not contain information regarding a "52-paper Qdrant index" or specify which parameters should be used for that particular project.

**Compressed answer**

The supplied evidence is insufficient to answer the question. While the context provides specific parameter values (M=16, efConstruction=500 or 40) used in experiments on a 200M SIFT dataset [5], it does not contain information regarding a "52-paper Qdrant index" or specify which parameters should be used for that particular project.

### p75_unanswerable_energy

**Question:** How many kilowatt-hours are required to embed all 52 PDFs on an M-series Mac?

Category: unanswerable; answerability: unanswerable; actual compression ratio: 0.7227

**Original answer**

The supplied evidence is insufficient to answer the question.

While SOURCE [2] mentions experiments conducted on a consumer-grade Apple M1 Mac with 16GB of RAM, it does not provide data regarding the energy consumption (in kilowatt-hours) required for embedding tasks or processing PDFs. The document only reports token expenditure and time required for tree-building processes [2].

**Compressed answer**

The supplied evidence is insufficient to answer the question. While SOURCE [2] mentions experiments conducted on an Apple M1 Mac with 16GB of RAM, it does not provide data on energy consumption or kilowatt-hours required for embedding tasks. Additionally, no source specifies the number of PDFs (52) being processed or their token counts.

## Review boundary

Manually assess factual correctness, citation support, clarity, lost multi-sentence dependencies, numerical/entity preservation, and whether either condition should have abstained. Automated citation/refusal/term-overlap fields are diagnostics, not a substitute for that review.

## Manual review

Seven of the eight answerable questions preserved the essential answer under compression. Both unanswerable questions correctly abstained in both conditions, and both conditions produced zero invalid citation references. Numerical evidence for Atlas, the docT5query direction, E5's CCPairs supervision, the TREC large-data regime, the BEIR trade-off, and ColBERT's delayed interaction all survived. The compressed E5 answer was notably more direct, while the compressed ColBERT answer remained correct but omitted the original answer's explicit MaxSim explanation.

The one material degradation was `p75_compare_beir_bright`. Frozen retrieval returned BEIR, MTEB, and HyDE passages but no BRIGHT passage, so the original condition correctly reported insufficient evidence. The compressed condition instead claimed that BRIGHT targets instruction-following generative LLMs and cited the HyDE source. BRIGHT's verified evidence says that it targets reasoning-intensive retrieval. The citation number was syntactically valid, but the cited passage did not support the claim. Compression did not create new text or lose a retrieved BRIGHT source; it changed the generator's behavior around an upstream retrieval failure and encouraged an unsupported synthesis.

This case demonstrates three evaluation distinctions:

1. Exact extractive provenance proves where compressed text came from, not that the generator interprets it correctly.
2. Citation validity proves that a source number exists, not that the source entails the claim.
3. Evidence-term overlap is sensitive to answer concision and cannot by itself measure correctness. Mean answer/evidence term coverage fell from 0.618 to 0.510 even though seven answerable outputs remained correct.

Compressed context used 26.2% fewer source characters. Generation took 64.9 seconds instead of 82.7 seconds in total, a measured 21.5% reduction, and was faster on nine of ten questions. This is promising but not a clean causal latency estimate: original generation always ran first, and the ten-question sample cannot separate prompt-length effects from warm-up, caching, and normal local-model variation. Compression itself added about 62 ms per question in this diagnostic.

## Decision

Keep global-75% extractive compression as an experimental option only. Do not integrate it into the CLI, FastAPI, `RAGApplication`, prompt builder, or LangGraph workflow. The aggregate size/retention trade-off is useful, but one unsupported answer in ten controlled comparisons is too serious to justify a default—especially because it occurred on a multi-source retrieval miss where abstention was the correct behavior.

More aggressive 50% and 30% budgets are rejected for generation testing at this stage. An LLM-based compressor is also not justified yet: it would add cost and introduce a second opportunity to synthesize unsupported text before the project has a stronger entailment or abstention check. A future study should counterbalance generation order, include more multi-source and cross-page questions, and manually judge claim-to-citation support.
