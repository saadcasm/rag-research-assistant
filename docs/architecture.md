# Phase 1 through Phase 4 architecture

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
          NumPy cosine similarity (retrieval.py)
                       |
                       v
          top-k Chunk + score results (retrieval.py)
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
