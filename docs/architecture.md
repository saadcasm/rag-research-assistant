# Phase 1 through Phase 10F architecture

```text
data/papers/*.pdf
        |
        v
PDF discovery (pdf.py)
        |
        v
PageText(document, page_number, text)
        |
        v
normalization + boundary-aware chunking (chunking.py)
        |
        v
Chunk(text, document, page_number, offsets, IDs)
        |
        v
data/processed/chunks.jsonl
        |
        v
local embedding model (embeddings.py)
        |
        v
embeddings.npy + integrity manifest (index.py)
        |
        +------------------------------+
                                       |
natural-language query                 |
        |                              |
        v                              v
same local embedding model     row-aligned chunk vectors
        |                              |
        +--------------+---------------+
                       v
          NumPy cosine similarity (retrieval.py) ----+
                                                     |
chunks.jsonl -> tokenization + BM25 (bm25.py) -------+
                                                     v
                                  rank fusion (hybrid.py)
                                                     |
                                                     v
                          optional cross-encoder (reranking.py)
                                                     |
                                                     v
                                  top-k Chunk + typed score
                       |
                       v
       numbered source context (context.py)
                       |
                       v
        grounded prompt (prompting.py)
                       |
                       v
       local Ollama generator (generation.py)
                       |
                       v
       cited answer + source mapping (rag.py)

curated questions.jsonl
        |
        +------> existing semantic search
                       |
                       v
 expected sources <-> ranked sources
                       |
                       v
       Hit@k + first-correct rank
                       |
                       v
       JSON report + terminal summary
```

The page is the metadata boundary in Phase 1. A chunk never crosses pages, so a
page citation remains unambiguous. The tradeoff is that content around a page
break cannot share a chunk; a later phase can evaluate whether multi-page
sections are worth more complex provenance.

Chunking uses characters rather than model-specific tokens. That makes the
algorithm inspectable before a particular embedding model is chosen. It aims for
`chunk_size`, prefers paragraph or sentence endings, and repeats `overlap`
characters of context. Exact chunk quality is empirical: retrieval evaluation,
not convention, should eventually determine these settings.

JSON Lines keeps the intermediate artifact human-readable and stream-friendly.
Each line is an independent chunk record. It is deliberately not a vector-store
schema; Phase 2 can read these records and add embeddings without re-extracting
private papers.

## Phase 2A component boundaries

`embeddings.py` is the only component that knows about Sentence Transformers.
It exposes separate document and query methods while guaranteeing both use the
same model. Indexing and retrieval depend on that small interface, so unit tests
can use deterministic fake vectors without downloading a model or asserting
fragile real-model scores.

`index.py` writes a dense `float32` NumPy matrix. Matrix row `i` corresponds to
chunk `i` from `chunks.jsonl`. The manifest records that row-to-chunk ID mapping,
the exact model name, vector dimension, and a SHA-256 fingerprint of the source
JSONL. Loading fails if any of those invariants no longer hold. This prevents a
subtle but serious error: showing metadata from one chunk beside another chunk's
vector.

`retrieval.py` performs a brute-force scan. It normalizes the query and every
document vector through the cosine formula, ranks the resulting one-dimensional
score array, and selects the first `k` rows. This costs roughly `O(N × D)` per
query for `N` chunks of dimension `D`. The direct calculation is appropriate for
a small research collection and makes the mechanism observable before an
approximate-nearest-neighbor index is introduced.

The embedding model is deliberately stored in the manifest. A vector coordinate
has meaning only within the model that learned that vector space, so a query from
a different model must never be compared with the persisted document matrix.

## Phase 3 component boundaries

`context.py` transforms ranked `SearchResult` records into evidence blocks. Each
block includes its citation number, filename, page, chunk ID, and full chunk
text. It separately returns `CitationSource` records for display after generation.
The citation map is therefore deterministic; the LLM does not invent the map.

`prompting.py` receives only a question and formatted context. It states that the
context is quoted data rather than instructions, restricts the answer to that
evidence, requires numbered citations, and defines the insufficient-evidence
behavior. Keeping this pure makes the exact LLM input easy to inspect and test.

`generation.py` defines the small `TextGenerator` protocol. `OllamaGenerator` is
one implementation that checks `GET /api/tags` for local availability and sends
a non-streaming request to `POST /api/generate`. It knows nothing about chunks,
embeddings, or retrieval. Another local runtime can implement the same protocol
without changing context construction or answer orchestration.

`rag.py` owns stage ordering:

```text
search -> optional debug callback -> context -> prompt -> generate -> GroundedAnswer
```

The debug callback runs after retrieval and before generation, which makes
`--show-context` an honest view of the evidence the generator is about to see.
If no chunks are retrieved, orchestration returns a deterministic
insufficient-context response without calling the LLM.

This separation exposes two fundamentally different failure classes. Retrieval
failure means the right evidence was not selected; prompt wording or a larger LLM
cannot recover evidence it never received. Generation failure means useful
evidence was present but the runtime failed or the model produced an unsupported,
incorrect, or poorly cited answer.

## Phase 4 component boundaries

`evaluation.py` sits beside the production RAG path rather than inside it. It
loads human labels, calls the existing `search` function, compares ranked chunks
with expected sources, optionally invokes the existing prompt/generator
components, and produces typed diagnostics. It does not reimplement embedding,
cosine similarity, prompting, or Ollama access.

The JSONL dataset is source-control input. Each line has a stable ID, question,
answerability category, expected source list, and explanatory notes. A source
requires an exact filename match. Page and chunk ID become additional exact
constraints only when present. This supports page-level labels today without
preventing more precise chunk labels later.

Retrieval metrics are calculated only for `answerable` and
`partially_answerable` examples. For each question, evaluation retrieves at least
five chunks once, then derives Hit@1, Hit@3, and Hit@5 from prefixes of that same
ranking. This avoids three embedding calls and ensures the cutoff comparison is
internally consistent. Multiple labels use "any expected source" for Hit@k and
"distinct expected sources found / expected sources" for recall@k.

`unanswerable` examples deliberately keep their top-k results in the report but
have null retrieval metrics. A dense retriever always ranks the available rows;
there is no built-in "none of these passages answers the question" state.
Refusal behavior therefore belongs to optional generation evaluation.

Generation evaluation uses the same `build_context`, `build_grounded_prompt`,
and `TextGenerator` boundary as Phase 3. Numeric bracket citations are extracted
and resolved deterministically. Out-of-range numbers are invalid. This proves
only that a cited context item exists—not that it entails the generated claim.
The refusal detector is likewise a transparent phrase heuristic, not a
factuality metric.

Reports include embedding model, dimension, indexed chunk count, retrieval
depth, optional generation settings, aggregate metrics, and full per-question
context. Retrieval scores are reproducible when the dataset, chunks, model, and
index are unchanged. The creation timestamp varies, and local LLM text may vary
even at low temperature.

Generated reports remain outside version control. Tests construct fake vectors
and generators so CI validates orchestration and mathematics without private
PDFs, network access, model downloads, or Ollama.

## Phase 5 component boundaries

`bm25.py` owns lexical tokenization and corpus statistics. `BM25Index` is built
once per command and reused for every query, so document frequencies and lengths
are not recalculated inside evaluation. The implementation is explicit Python
over roughly 209 chunks; no search server or hidden NLP pipeline is justified at
this scale.

`hybrid.py` knows only ranked `SearchResult` lists. It merges duplicate chunk IDs
and calculates Reciprocal Rank Fusion with `rrf_k=60`. It never sees cosine or
BM25 internals and therefore cannot accidentally treat unlike raw scores as
comparable.

`reranking.py` owns the optional Sentence Transformers cross-encoder adapter.
It jointly scores only the 20 candidate question/chunk pairs and returns new
`SearchResult` values whose score is explicitly labelled cross-encoder. Its
small protocol permits deterministic fake scorers in tests.

`retrievers.py` is the shared strategy boundary:

```text
Retriever.search(query, top_k) -> ordered SearchResult records
```

`DenseRetriever`, `BM25Retriever`, `HybridRetriever`, and
`RerankingRetriever` compose the specialized modules. `rag.py` and
`evaluation.py` receive this abstraction rather than selecting algorithms or
duplicating them. When none is supplied programmatically, both preserve the
Phase 2 dense default.

The production reranker path loads the Hugging Face cache with
`local_files_only=True`. The separate `reranker-download` command performs the
one authorized download. Cache files and generated comparison reports are local
artifacts excluded by `.gitignore`.

See [the Phase 5 guide](phase-5-hybrid-retrieval.md) for formulas, score
semantics, commands, performance tradeoffs, and the measured comparison.

## Phase 6 component boundaries

`qdrant_store.py` is the only component that understands Qdrant collections,
points, UUID mapping, payload filters, or local persistence. It converts Qdrant
responses back to the same `SearchResult` and `Chunk` contracts consumed by all
later phases.

```text
                        +-> embeddings.npy -> DenseRetriever ------+
chunks.jsonl -> vectors |                                       |
                        +-> local Qdrant -> QdrantDenseRetriever --+
                                                                |
                                        common dense ranking <---+
                                                |
                                                +-> BM25 -> RRF
                                                            |
                                                    cross-encoder
                                                            |
                                                   context -> Qwen
```

The `--dense-backend` decision is made once in CLI construction. Hybrid fusion,
reranking, evaluation, and `rag.py` depend on the `Retriever` protocol and have
no backend conditionals. Programmatic callers that omit a retriever retain the
original NumPy behavior.

The collection metadata is the Qdrant equivalent of the NumPy manifest: model,
dimension, chunk fingerprint, count, and schema version travel with the stored
vectors. Each point payload contains the seven fields required to reconstruct a
Chunk. Arbitrary logical chunk IDs become deterministic UUIDv5 point IDs, so
upsert is idempotent without changing application identity.

Embedded local Qdrant persists under `data/processed/qdrant/` and performs an
exact full scan for this small collection. NumPy and Qdrant therefore share the
same correctness model today. At larger scale, a server deployment could use
Qdrant's HNSW graph and payload indexes without changing downstream retrieval
contracts.

See [the Phase 6 guide](phase-6-qdrant.md) for the storage model, safety checks,
CLI, exact-versus-approximate search, filtering, and measured equivalence.
# Phase 7: chunking strategies

The ingestion boundary is now `DocumentParser (pypdf) -> ChunkingStrategy -> JSONL`.
`legacy`, `boundary`, `structural`, and `semantic` produce the same `Chunk` contract,
so BM25, dense retrieval, RRF, reranking, and Qdrant do not need strategy branches.
The contract carries an inclusive page span and optional section title; `page_number`
is retained as its start page for compatibility. See [Phase 7](phase-7-advanced-chunking.md).

## Phase 8 application boundary

`application.py` moves retriever construction out of the CLI transport. Both
CLI and FastAPI construct the same `Retriever` graph through `load_retriever`
and ultimately call `rag.answer_question`. HTTP routes contain no BM25, fusion,
reranking, prompting, or generation implementation.

```text
CLI arguments -> RetrievalConfig ---+
                                      v
                              load_retriever
                                      |
HTTP lifespan -> ApplicationSettings +-> RAGApplication
                                              |
HTTP request -> Pydantic contract ------------+
                                              v
                                    answer_question
                                              |
                                              v
                               GroundedAnswer + sources
```

FastAPI's lifespan owns one process-scoped `RAGApplication`. It loads the
embedding model, reranker, chunks/BM25 state, Qdrant client, and Ollama generator
once, reuses them across requests, and closes Qdrant on shutdown. The HTTP layer
converts existing `GroundedAnswer` and `CitationSource` records into a public
Pydantic response contract. See [the Phase 8 API guide](phase-8-api.md).

## Phase 9 parallel framework experiments

The default architecture above is unchanged. The isolated `frameworks/`
package adapts the existing `Retriever` to LangChain Documents/Runnables and
uses LangGraph for one bounded retrieve/evaluate/rewrite branch:

```text
existing Retriever -> LangChainRAGPipeline -> GroundedAnswer
                  \-> BoundedRetrievalGraph -> GroundedAnswer
```

The framework paths do not own ingestion, indexes, retrieval algorithms,
evaluation, FastAPI routes, or application defaults. See
[the Phase 9 comparison](phase-9-langchain-langgraph.md).

## Phase 10A experimental query transformation

The production graph remains unchanged. The Phase 10A runner calls the same
frozen retriever for preliminary and final retrieval, with a corpus-grounded
`QueryRewriter` between them. It delegates relevance scoring back to
`evaluation.evaluate_retrieval_results` and writes auditable JSON/Markdown.

```text
original question -> existing retriever -> preliminary evidence
                                      \-> local LLM -> one rewritten query
                                                     -> existing retriever
```

The original question and transformed retrieval query are separate fields.
Nothing in `RAGApplication`, FastAPI, CLI, or Phase 9 imports this experiment.
See [Phase 10A](phase-10a-query-rewriting.md).

## Phase 10B experimental multi-query retrieval

Phase 10B remains outside every production transport and workflow. It reuses
the established `RerankingRetriever`, but inserts a second RRF stage at its
existing boundary between hybrid candidate retrieval and cross-encoder
reranking:

```text
each query -> Qdrant dense + BM25 -> hybrid RRF -> query ranking
query rankings -> multi-query RRF -> cross-encoder(original question) -> top-k
```

The original query is always the first ranking. Generated alternatives can add
candidates but cannot replace that ranking. Both RRF stages deduplicate by the
stable chunk ID, and the final cross-encoder implementation, model, candidate
depth, and original-question input remain unchanged.

See [Phase 10B](phase-10b-multi-query.md).

The frozen benchmark supports keeping multi-query-2 as an optional experiment,
not replacing the production default: it modestly improved top-3/top-5 retrieval
without baseline-relative rank regressions, but generation dominated latency and
three alternatives did not improve on two. See the
[Phase 10B results](phase-10b-multiquery-results.md).

## Phase 10C-1 experimental routing analysis

Phase 10C-1 is offline analysis, not a runtime branch. It reads Phase 10B's
persisted baseline reranker results, original-query hybrid candidates, and MQ2
outcome labels, then extracts score-shape, document-concentration, query-term,
identity-term, and reranking-stability signals. It systematically sweeps every
observed threshold boundary in both directions and reports routing cost and
retrieval metrics.

```text
Phase 10B result JSON -> cheap baseline signals + MQ2 labels
                      -> exploratory threshold sweeps
                      -> routing metrics and leakage warning
```

Nothing imports this analysis from FastAPI, the CLI, `RAGApplication`, or the
LangGraph workflow. No production router is implemented because five positive
examples do not support a stable rule. See the
[Phase 10C-1 report](phase-10c-routing-signal-analysis.md).

## Phase 10D experimental parent-child retrieval

Phase 10D reuses the semantic chunk index as a focused child layer. After the
existing hybrid and cross-encoder stack ranks children, stable prebuilt mappings
expand them into physical pages or uniquely containing structural chunks.
Parents are deduplicated and ranked by reciprocal contributing-child ranks.

```text
semantic child -> dense + BM25 -> hybrid RRF -> cross-encoder
               -> stable parent mapping -> parent rank aggregation
               -> page or structural context + child provenance
```

Ambiguous structural relationships fall back to page parents and remain visible
as warnings. This experiment is not imported by FastAPI, the CLI,
`RAGApplication`, or LangGraph. See the
[Phase 10D engineering guide](phase-10d-parent-child.md).

The benchmark keeps legacy as the production default. Full-page expansion
weakens early precision and adds roughly 8× context. Exact structural parents
are the more promising branch: they preserve top-five retrieval and average
about 2× expansion, but only 64.6% of children map without page fallback. See
the [Phase 10D results](phase-10d-parent-child-results.md).

## Phase 10E experimental contextual compression

Phase 10E leaves retrieval untouched and transforms only the text supplied
after the frozen top-five legacy ranking:

```text
legacy retrieval -> unchanged SearchResult ranking
                 -> sentence-group segmentation
                 -> batched cross-encoder segment scoring
                 -> budget selection
                 -> original source and segment order
                 -> compressed, provenance-bearing context
```

The compressor reuses the process's existing cross-encoder rather than loading
another model. Segment scores decide which exact source spans survive; they are
not probabilities and never reorder retrieved sources. Per-chunk selection
keeps at least one segment from each non-empty chunk. Global selection keeps at
least one segment overall and explicitly records omitted source ranks.

The 100-question experiment favors the conservative global-75% condition over
more aggressive budgets, but the 10-question generation diagnostic contains
one unsupported compressed answer on an upstream multi-source retrieval miss.
That result prevents production integration. FastAPI, CLI, `RAGApplication`,
generation prompting, and the Phase 9 graph continue to use complete legacy
chunks. See the [Phase 10E guide](phase-10e-contextual-compression.md).

## Phase 10F experimental HyDE retrieval

HyDE adds one isolated semantic probe while preserving real corpus evidence:

```text
question -> Qwen hypothetical passage -> existing embedder
         -> existing legacy Qdrant collection -> real chunks

frozen baseline ranking + HyDE dense ranking
         -> equal RRF -> cross-encoder(original question, real chunk)
```

The Qdrant retriever now exposes a behavior-preserving precomputed-vector
method so the experiment can measure embedding separately. Normal production
search still embeds its query and follows the same code path as before. The
hypothetical passage never enters a prompt as evidence, and it is never used
for final cross-encoder scoring.

HyDE-only moved many rankings but traded four top-five rescues for five losses.
Equal fusion prevented every measured relevance regression, yet also produced
zero gains across 90 scored questions while increasing mean request cost from
about 0.20 to 4.42 seconds. No API, CLI, `RAGApplication`, generation, or
LangGraph path imports HyDE. See the [Phase 10F guide](phase-10f-hyde.md).
