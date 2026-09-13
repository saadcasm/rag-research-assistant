# Proposed Phase 7.5 corpus: scaling and evaluation expansion

Status: **metadata and source-validated proposal; not approved for download
automation yet**. Do not download from this document yet. The current three PDFs
remain in the corpus and are marked `existing`. The validation date is 2026-09-13.
For every arXiv item, the listed link is the preferred direct PDF URL and its
title/authors/year/arXiv identity were checked against the arXiv record. For the
seven non-arXiv items, the listed venue or author source was checked separately.

## Selection principles

This is a 56-paper *retrieval systems* corpus, not a general AI reading list.
It mixes foundation papers with near-neighbour alternatives: dense versus sparse,
single-vector versus late interaction, query expansion versus rewriting, and
RAG versus long-context. That deliberate overlap makes natural-language queries
ambiguous in useful ways and gives a retrieval benchmark plausible distractors.

Difficulty indicates expected distractor value in this corpus: `high` means the
paper shares terminology and likely answers with several neighbours; `medium`
means it anchors a concept but still has meaningful overlap.

## Validation status and source policy

The manifest has two explicit status rules, applied to **every row**:

- A row whose canonical source is an `arXiv` direct-PDF URL has
  `metadata_verified=true`, `pdf_available=true`, and
  `preferred_pdf_source=<that URL>`. The arXiv abstract record was checked for
  exact title, author list, first-submission year, and identifier.
- A row whose canonical source is a direct ACL Anthology or TREC PDF has
  `metadata_verified=true`, `pdf_available=true`, and that URL is the preferred
  PDF source. Venue metadata was checked against that primary record.

Five older non-arXiv rows are deliberately **not approved for automated
download**: `bm25`, `deepct`, `rrf`, `product_quantization`, and `mmr`. They have
`metadata_verified=true`, `pdf_available=uncertain`, and require a separate
license/source decision. A DOI or ACM landing page is not evidence that a legal
public PDF can be fetched. This leaves 52 immediately downloadable public-PDF
records and five held records.

## Manifest

| ID | Title | Authors | Year | Canonical source | Primary cluster | Benchmark rationale / conceptual overlap | Difficulty |
|---|---|---|---:|---|---|---|---|
| rag_lewis_2020 *(existing)* | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Patrick Lewis et al. | 2020 | [arXiv:2005.11401](https://arxiv.org/pdf/2005.11401) | RAG | Original RAG reference; overlaps REALM, FiD, Atlas, RETRO and RePlug on external memory. | high |
| realm | REALM: Retrieval-Augmented Language Model Pre-Training | Kelvin Guu et al. | 2020 | [arXiv:2002.08909](https://arxiv.org/pdf/2002.08909) | RAG | Contrasts retrieval-augmented pretraining with generation-time RAG. | high |
| fid | Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering | Gautier Izacard, Edouard Grave | 2020 | [arXiv:2007.01282](https://arxiv.org/pdf/2007.01282) | RAG | Fusion-in-Decoder is a close architectural alternative to RAG. | high |
| retro | Improving Language Models by Retrieving from Trillions of Tokens | Sebastian Borgeaud et al. | 2021 | [arXiv:2112.04426](https://arxiv.org/pdf/2112.04426) | RAG | Retrieval at language-model scale; overlaps RAG/REALM but shifts the scale question. | high |
| atlas | Few-shot Learning with Retrieval Augmented Language Models | Gautier Izacard et al. | 2022 | [arXiv:2208.03299](https://arxiv.org/pdf/2208.03299) | RAG | RAG training and few-shot use; close to FiD and Lewis RAG. | high |
| replug | REPLUG: Retrieval-Augmented Black-Box Language Models | Weijia Shi et al. | 2023 | [arXiv:2301.12652](https://arxiv.org/pdf/2301.12652) | RAG | Tests plug-in retrieval when the generator cannot be fine-tuned. | high |
| self_rag | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | Akari Asai et al. | 2023 | [arXiv:2310.11511](https://arxiv.org/pdf/2310.11511) | RAG | Adds adaptive retrieval/critique; overlaps corrective and advanced RAG. | high |
| corrective_rag | Corrective Retrieval Augmented Generation | Shi-Qi Yan et al. | 2024 | [arXiv:2401.15884](https://arxiv.org/pdf/2401.15884) | RAG | Useful distractor for “when should retrieval be corrected?” questions. | high |
| raptor | RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval | Parth Sarthi et al. | 2024 | [arXiv:2401.18059](https://arxiv.org/pdf/2401.18059) | RAG | Hierarchical retrieval overlaps chunking, long context, and multi-hop evidence. | high |
| dpr_karpukhin_2020 *(existing)* | Dense Passage Retrieval for Open-Domain Question Answering | Vladimir Karpukhin et al. | 2020 | [arXiv:2004.04906](https://arxiv.org/pdf/2004.04906) | Dense retrieval | Core bi-encoder baseline; overlaps ANCE, RocketQA, Contriever, and ColBERT. | high |
| ance | Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval | Lee Xiong et al. | 2020 | [arXiv:2007.00808](https://arxiv.org/pdf/2007.00808) | Dense retrieval | Hard negatives and ANN refresh make it a close DPR training distractor. | high |
| rocketqav2 | RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking | Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, Ji-Rong Wen | 2021 | [arXiv:2110.07367](https://arxiv.org/pdf/2110.07367) | Dense retrieval | Couples retriever and reranker; intentionally spans two pipeline stages. Corrected from the distinct RocketQA paper (arXiv:2010.08191). | high |
| contriever | Unsupervised Dense Information Retrieval with Contrastive Learning | Gautier Izacard et al. | 2021 | [arXiv:2112.09118](https://arxiv.org/pdf/2112.09118) | Dense retrieval | Contrasts supervised DPR with unsupervised contrastive retrieval. | high |
| gtr | Large Dual Encoders Are Generalizable Retrievers | Jianmo Ni et al. | 2022 | [arXiv:2112.07899](https://arxiv.org/pdf/2112.07899) | Dense retrieval | Scale/generalization comparison for dual encoders. | high |
| cocondenser | Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval | Luyu Gao, Zhuyun Dai, Jamie Callan | 2021 | [arXiv:2108.05540](https://arxiv.org/pdf/2108.05540) | Dense retrieval | Corrected author list; another pretraining route to DPR-like models. | high |
| colbert | ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | Omar Khattab, Matei Zaharia | 2020 | [arXiv:2004.12832](https://arxiv.org/pdf/2004.12832) | Dense retrieval | Multi-vector late interaction creates a useful counterpoint to DPR’s single vector. | high |
| colbertv2 | ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction | Keshav Santhanam et al. | 2021 | [arXiv:2112.01488](https://arxiv.org/pdf/2112.01488) | Dense retrieval | Compression and residual representations overlap ColBERT and vector indexing. | high |
| bm25 | Some Simple Effective Approximations to the 2-Poisson Model for Probabilistic Weighted Retrieval | Stephen Robertson, Steve Walker | 1994 | [SIGIR canonical record](https://doi.org/10.1007/978-1-4471-2099-5_17) | Sparse / lexical | Essential lexical baseline; a productive contrast to every dense/hybrid paper. PDF availability to verify. | high |
| deepct | Context-Aware Term Weighting For First Stage Passage Retrieval | Zhuyun Dai, Jamie Callan | 2020 | [SIGIR DOI](https://doi.org/10.1145/3397271.3401204) | Sparse / lexical | Learns term weights while retaining an inverted-index retrieval shape. The prior arXiv ID was unrelated; no safely verified public PDF yet. | high |
| doct5query | Document Expansion by Query Prediction | Rodrigo Nogueira et al. | 2019 | [arXiv:1904.08375](https://arxiv.org/pdf/1904.08375) | Sparse / lexical | Blurs sparse retrieval and generative expansion; overlaps query2doc. | high |
| coil | COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List | Luyu Gao, Zhuyun Dai, Jamie Callan | 2021 | [arXiv:2104.07186](https://arxiv.org/pdf/2104.07186) | Sparse / lexical | Replaces the unverified uniCOIL placeholder with a verified contextualized-inverted-list paper. | high |
| splade | SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking | Thibault Formal et al. | 2021 | [arXiv:2107.05720](https://arxiv.org/pdf/2107.05720) | Sparse / lexical | Important learned sparse expansion family; overlaps uniCOIL and DeepImpact. | high |
| deepimpact | Learning Passage Impacts for Inverted Indexes | Antonio Mallia, Omar Khattab, Nicola Tonellotto, Torsten Suel | 2021 | [arXiv:2104.12016](https://arxiv.org/pdf/2104.12016) | Sparse / lexical | DeepImpact is the method name; the official paper title is corrected. | high |
| rrf | Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods | Gordon Cormack, Charles Clarke, Stefan Buettcher | 2009 | [SIGIR canonical record](https://dl.acm.org/doi/10.1145/1571941.1572114) | Hybrid retrieval | The exact fusion idea used by this project; overlaps hybrid and ranking evaluation. PDF availability to verify. | high |
| hyrr | HYRR: Hybrid Infused Reranking for Passage Retrieval | Jing Lu et al. | 2022 | [arXiv:2212.10528](https://arxiv.org/pdf/2212.10528) | Hybrid retrieval | Directly tests hybrid candidate pools and reranker robustness. | high |
| hyde | Precise Zero-Shot Dense Retrieval without Relevance Labels | Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan | 2022 | [arXiv:2212.10496](https://arxiv.org/pdf/2212.10496) | Query transformation / multi-query | HyDE generates a hypothetical document before dense retrieval; it is not a hybrid rank-fusion method. | high |
| sgpt *(existing)* | SGPT: GPT Sentence Embeddings for Semantic Search | Niklas Muennighoff | 2022 | [arXiv:2202.08904](https://arxiv.org/pdf/2202.08904) | Embeddings | Existing decoder-based embedding reference; overlaps SBERT/E5/BGE. | high |
| sbert | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Nils Reimers, Iryna Gurevych | 2019 | [arXiv:1908.10084](https://arxiv.org/pdf/1908.10084) | Embeddings | Foundational sentence embedding paper and direct semantic-search baseline. | high |
| simcse | SimCSE: Simple Contrastive Learning of Sentence Embeddings | Tianyu Gao, Xingcheng Yao, Danqi Chen | 2021 | [arXiv:2104.08821](https://arxiv.org/pdf/2104.08821) | Embeddings | Contrastive embedding training creates terminology overlap with Contriever. | high |
| e5 | Text Embeddings by Weakly-Supervised Contrastive Pre-training | Liang Wang et al. | 2022 | [arXiv:2212.03533](https://arxiv.org/pdf/2212.03533) | Embeddings | Widely used retrieval embedding family; good practical comparison to the project model. | high |
| instructor | One Embedder, Any Task: Instruction-Finetuned Text Embeddings | Hongjin Su et al. | 2022 | [arXiv:2212.09741](https://arxiv.org/pdf/2212.09741) | Embeddings | Tests task instruction conditioning against plain embedding queries. | high |
| bge | C-Pack: Packed Resources For General Chinese Embeddings | Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, Jian-Yun Nie | 2023 | [arXiv:2309.07597](https://arxiv.org/pdf/2309.07597) | Embeddings | Correct official title; a useful BGE-adjacent embedding resource paper. | high |
| bert_reranker | Passage Re-ranking with BERT | Rodrigo Nogueira, Kyunghyun Cho | 2019 | [arXiv:1901.04085](https://arxiv.org/pdf/1901.04085) | Reranking | Canonical cross-encoder baseline; overlaps MonoT5 and current Phase 5 reranking. | high |
| monot5 | Document Ranking with a Pretrained Sequence-to-Sequence Model | Rodrigo Nogueira et al. | 2020 | [arXiv:2003.06713](https://arxiv.org/pdf/2003.06713) | Reranking | A generative reranker creates useful ambiguity with RAG generators. | high |
| rankt5 | RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses | Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai Hui, Ji Ma, Jing Lu, Jianmo Ni, Xuanhui Wang, Michael Bendersky | 2022 | [arXiv:2210.10634](https://arxiv.org/pdf/2210.10634) | Reranking | Direct sibling of MonoT5 with ranking-loss distinctions. | high |
| rankgpt | Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents | Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren | 2023 | [arXiv:2304.09542](https://arxiv.org/pdf/2304.09542) | Reranking | Corrected lead author; adds listwise LLM reranking without requiring it in the app yet. | high |
| faiss | Billion-scale similarity search with GPUs | Jeff Johnson, Matthijs Douze, Hervé Jégou | 2017 | [arXiv:1702.08734](https://arxiv.org/pdf/1702.08734) | Vector similarity search | Correct official title capitalization; grounds ANN and product-quantization vocabulary behind vector stores. | medium |
| hnsw | Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs | Yu. A. Malkov, D. A. Yashunin | 2016 | [arXiv:1603.09320](https://arxiv.org/pdf/1603.09320) | Vector similarity search | Correct official title capitalization; core graph-ANN trade-off paper. | medium |
| product_quantization | Product Quantization for Nearest Neighbor Search | Hervé Jégou, Matthijs Douze, Cordelia Schmid | 2011 | [IEEE canonical record](https://doi.org/10.1109/TPAMI.2010.57) | Vector similarity search | Foundational compression/distance-approximation paper. PDF availability to verify. | medium |
| texttiling | TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages | Marti Hearst | 1997 | [ACL Anthology PDF](https://aclanthology.org/J97-1003.pdf) | Document chunking / parsing | Classical topic segmentation counterpoint to embedding-assisted semantic chunking. | high |
| layoutlm | LayoutLM: Pre-training of Text and Layout for Document Image Understanding | Yiheng Xu et al. | 2019 | [arXiv:1912.13318](https://arxiv.org/pdf/1912.13318) | Document chunking / parsing | Forces distinction between text extraction and layout-aware document structure. | medium |
| late_chunking | Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models | Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, Han Xiao | 2024 | [arXiv:2409.04701](https://arxiv.org/pdf/2409.04701) | Document chunking / parsing | Corrected authors; direct Phase 7 follow-on: contextualize before chunking rather than after. | high |
| beir | BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models | Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych | 2021 | [arXiv:2104.08663](https://arxiv.org/pdf/2104.08663) | Retrieval evaluation | Corrected official spelling and author list; essential reminder that one corpus can mislead retrieval choices. | high |
| mteb | MTEB: Massive Text Embedding Benchmark | Niklas Muennighoff et al. | 2022 | [arXiv:2210.07316](https://arxiv.org/pdf/2210.07316) | Retrieval evaluation | Connects embedding choice to broad, task-specific evaluation. | high |
| msmarco | MS MARCO: A Human Generated MAchine Reading COmprehension Dataset | Payal Bajaj et al. | 2016 | [arXiv:1611.09268](https://arxiv.org/pdf/1611.09268) | Retrieval evaluation | Corrected lead author; source of many retriever/reranker training assumptions. | high |
| bright | BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval | Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S. Siegel, Michael Tang, Ruoxi Sun, Jinsung Yoon, Sercan O. Arik, Danqi Chen, Tao Yu | 2024 | [arXiv:2407.12883](https://arxiv.org/pdf/2407.12883) | Retrieval evaluation | Corrected authors; challenges shallow relevance and tests reasoning-heavy retrieval. | high |
| trec_dl | Overview of the TREC 2020 Deep Learning Track | Nick Craswell et al. | 2020 | [TREC proceedings](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.DL.pdf) | Retrieval evaluation | Canonical judged ranking evaluation reference. | medium |
| lost_in_middle | Lost in the Middle: How Language Models Use Long Contexts | Nelson F. Liu et al. | 2023 | [arXiv:2307.03172](https://arxiv.org/pdf/2307.03172) | Long-context vs retrieval | Direct challenge to the claim that a longer context replaces retrieval. | high |
| longbench | LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding | Yushi Bai et al. | 2023 | [arXiv:2308.14508](https://arxiv.org/pdf/2308.14508) | Long-context vs retrieval | Broad long-context benchmark with retrieval-adjacent tasks. | medium |
| ruler | RULER: What's the Real Context Size of Your Long-Context Language Models? | Hsieh et al. | 2024 | [arXiv:2404.06654](https://arxiv.org/pdf/2404.06654) | Long-context vs retrieval | Synthetic long-context stress tests create useful contrast to real retrieval. | high |
| query2doc | Query2doc: Query Expansion with Large Language Models | Liang Wang et al. | 2023 | [arXiv:2303.07678](https://arxiv.org/pdf/2303.07678) | Query rewriting / multi-query | Deliberately overlaps docT5query and HyDE but expands the query side. | high |
| generative_rf | Generative Relevance Feedback with Large Language Models | Iain Mackie et al. | 2023 | [arXiv:2304.13157](https://arxiv.org/pdf/2304.13157) | Query rewriting / multi-query | Separates feedback expansion from rewriting and hypothetical documents. | high |
| rewrite_retrieve_read | Query Rewriting for Retrieval-Augmented Large Language Models | Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan | 2023 | [arXiv:2305.14283](https://arxiv.org/pdf/2305.14283) | Query transformation / multi-query | “Rewrite-Retrieve-Read” is the framework name in this paper, not its official title. | high |
| multihop_rag | MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries | Yixuan Tang, Yi Yang | 2024 | [arXiv:2401.15391](https://arxiv.org/pdf/2401.15391) | Advanced retrieval | Corrected author list; brings compositional evidence needs that a single chunk cannot satisfy. | high |
| graphrag | From Local to Global: A Graph RAG Approach to Query-Focused Summarization | Darren Edge et al. | 2024 | [arXiv:2404.16130](https://arxiv.org/pdf/2404.16130) | Advanced retrieval | Introduces graph/global-local terminology without adding it to the application. | high |
| m3_embedding | M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu | 2024 | [arXiv:2402.03216](https://arxiv.org/pdf/2402.03216) | Advanced retrieval | Corrected author list; one paper covering dense, sparse, and multi-vector embeddings. | high |
| mmr | The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries | Jaime Carbonell, Jade Goldstein | 1998 | [SIGIR canonical record](https://dl.acm.org/doi/10.1145/290941.291025) | Advanced retrieval | Adds relevance-versus-diversity trade-offs; PDF availability to verify. | medium |

## Distribution and review notes

### Metadata correction ledger

The original proposal was intentionally treated as a hypothesis. This validation
found and corrected the following concrete errors:

- `rocketqa` was labelled RocketQAv2 while pointing at **arXiv:2010.08191**, the
  distinct 2020 RocketQA paper. It is now `rocketqav2`, **arXiv:2110.07367**,
  2021, with the Ren et al. author list.
- `deepct` pointed to **arXiv:1910.13048**, an unrelated sparse-linear-regression
  paper. DeepCT is now the 2020 SIGIR work identified by DOI
  `10.1145/3397271.3401204`; no direct public PDF has been approved.
- `unicoil` pointed to **arXiv:2106.14807**, a notes/conceptual-framework paper,
  not the claimed work. It was replaced by the verified COIL paper,
  **arXiv:2104.07186**, rather than silently retaining a bad record.
- DeepImpact’s method nickname was presented as its paper title. Its official
  title is *Learning Passage Impacts for Inverted Indexes* (arXiv:2104.12016).
- The official titles/capitalization were corrected for C-Pack, FAISS, HNSW,
  BEIR, and the 2305.14283 query-rewriting paper. “Rewrite-Retrieve-Read” is a
  framework name, while the official title is *Query Rewriting for
  Retrieval-Augmented Large Language Models*.
- Author corrections include RocketQAv2, CoCondenser, C-Pack, RankT5, RankGPT,
  Late Chunking, BEIR, MS MARCO, BRIGHT, MultiHop-RAG, and M3-Embedding.
- HyDE moved from **hybrid retrieval** to **query transformation / multi-query**:
  it generates a hypothetical document and then invokes dense retrieval, rather
  than combining independently ranked sparse and dense lists.

No duplicate paper/version is retained. The initially dangerous near-duplicate
was RocketQA versus RocketQAv2: the corpus now includes only RocketQAv2. The
validated total remains **57**, not 56.

Primary-cluster distribution: RAG **9**, dense retrieval **8**, sparse/lexical
**6**, hybrid **2**, embeddings **6**, reranking **4**, vector similarity search
**3**, document chunking/parsing **3**, retrieval evaluation **5**, long-context
versus retrieval **3**, query transformation/multi-query **4**, and advanced
retrieval **4**. Total: **57**.

Year distribution: 1990s **3**, 2000s **1**, 2010s **9**, 2020 **8**, 2021 **10**,
2022 **9**, 2023 **9**, 2024 **8**. The apparent older skew is intentional:
BM25, RRF, MMR, TextTiling, PQ, and HNSW are vocabulary anchors that make modern
papers harder to distinguish, not obsolete filler.

### Deliberate retrieval difficulties

- **Dense-training ambiguity:** DPR, ANCE, RocketQA, Contriever, GTR, CoCondenser,
  ColBERT, and ColBERTv2 all discuss negatives, encoders, passages, and retrieval,
  but make different architectural/training claims.
- **Sparse/hybrid ambiguity:** BM25, DeepCT, uniCOIL, SPLADE, DeepImpact, RRF,
  HYRR, and HyDE force lexical, learned-sparse, fusion, and expansion distinctions.
- **Embedding/reranker ambiguity:** SBERT, SimCSE, E5, INSTRUCTOR, BGE, SGPT,
  BERT reranking, MonoT5, RankT5, and RankGPT reuse representation/ranking terms
  while operating at different stages.
- **RAG versus context ambiguity:** RAG, FiD, Atlas, RETRO, RePlug, Self-RAG,
  CRAG, RAPTOR, Lost in the Middle, LongBench, and RULER invite questions about
  whether retrieval, context length, or generation architecture is responsible.

### Gaps and likely redundancy

The proposal is deliberately English-heavy, passage-centric, and academic-PDF
heavy. It lacks domain retrieval (legal, biomedical, code), multimodal retrieval,
conversational search, non-English evaluation, and real enterprise tables. Add
those only after this retrieval-core benchmark is stable; otherwise they change
too many variables at once.

Potentially redundant pairs/groups to review before downloading are:

- **ColBERT and ColBERTv2:** retain both only if evolution/compression is a useful
  question type; otherwise v2 can stand in for the family.
- **MonoT5 and RankT5:** retain both only if the evaluation set will distinguish
  sequence-to-sequence scoring from ranking losses.
- **SBERT, SimCSE, E5, INSTRUCTOR, BGE, SGPT:** high-value overlap, but six
  embedding papers can dominate corpus vocabulary. Keep all only intentionally.
- **Query2doc, generative relevance feedback, HyDE, and Rewrite-Retrieve-Read:**
  strong controlled family for query transformation; retain together for that reason.

PDF availability is uncertain for the five held records above, including `deepct`
whose former arXiv ID was shown to be unrelated. Prefer a legal author, ACL
Anthology, or institutional PDF before approving any of them. At download time,
verify the direct PDF response, title page, source version, license, and SHA-256;
do not substitute a third-party mirror merely to fill the quota.
