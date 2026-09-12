# RAG Research Assistant

A portfolio project for learning retrieval-augmented generation by implementing
its fundamental components directly. The project currently covers ingestion,
local semantic retrieval, grounded local answer generation, and systematic
evaluation:

```text
PDF -> pages -> chunks -> dense + BM25 -> RRF -> optional reranker -> context
                                                              |
                                                              v
evaluation questions -> metrics + comparison             local LLM -> cited answer
```

No RAG framework, vector database, cloud LLM API, agent, or web UI is used.

## Phase 1 features

- Discovers multiple PDFs in `data/papers/` in deterministic order.
- Extracts text page by page with `pypdf`.
- Preserves the document filename and one-based page number.
- Normalizes common PDF whitespace noise.
- Creates readable, overlapping character chunks without crossing page boundaries.
- Writes chunks as inspectable JSON Lines records.
- Provides a CLI for processing papers and printing sample chunks.
- Tests chunking, metadata preservation, PDF discovery, and serialization.

## Phase 2A features

- Reads the existing Phase 1 `chunks.jsonl` records unchanged.
- Generates embeddings locally with a compact Sentence Transformers model.
- Persists a NumPy matrix and a row-to-chunk manifest under `data/processed/`.
- Detects a stale index when `chunks.jsonl` changes.
- Embeds a natural-language query with the same model as the chunks.
- Calculates cosine similarity directly with NumPy and returns the top-k rows.
- Displays score, filename, page, chunk ID, and a text preview.

## Phase 3 features

- Retrieves evidence before generation; the LLM never performs retrieval.
- Formats ranked chunks as numbered, source-labelled context.
- Builds an explicit prompt that restricts answers to supplied evidence.
- Uses a configurable model through a locally running Ollama server.
- Defaults to `qwen3.5:4b` and a conservative temperature of `0.1`.
- Maps answer citations back to filenames, page numbers, and chunk IDs.
- Optionally shows retrieved chunks before generation with `--show-context`.
- Reports missing Ollama and missing models with actionable commands.

## Phase 4 features

- Tracks a manually curated JSONL dataset of 20 questions over the three local papers.
- Separates fast retrieval evaluation from optional local generation evaluation.
- Calculates Hit@1, Hit@3, Hit@5, first-correct rank, and expected-source recall.
- Matches expected evidence by filename and optional page/chunk labels.
- Excludes unanswerable questions from retrieval Hit@k rather than treating the
  retriever's inevitable results as evidence.
- Records every ranked chunk, score, source label, and full text for diagnosis.
- Detects common refusal phrases with a small, documented heuristic.
- Extracts numeric citations and rejects references to context items that were
  never supplied.
- Writes an ignored machine-readable report and a concise terminal summary.
- Uses fakes in tests, so CI needs no PDFs, embedding download, Ollama, or Qwen.

## Phase 5 features

- Adds an explicit, dependency-free Okapi BM25 implementation over existing chunks.
- Combines dense and lexical rankings with Reciprocal Rank Fusion (RRF), not raw scores.
- Uses 20 candidates from each base retriever and `rrf_k=60` by default.
- Optionally reranks 20 hybrid candidates with a small local cross-encoder.
- Routes `search`, `ask`, and `evaluate` through one retriever interface.
- Adds `compare` for the same-dataset four-strategy benchmark and rank deltas.
- Labels cosine, BM25, RRF, and cross-encoder scores by type in terminal output.
- Preserves dense retrieval as the default and all Phase 1–4 behavior.

## Setup

Python 3.10 or newer is required. The first install includes PyTorch and Sentence
Transformers, and the first embedding command downloads the selected model. Model
inference and search then run locally; no API key is required. The `search`
command loads cached model files only and therefore works offline after `embed`
has downloaded the model once.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Phase 3 also requires Ollama. On macOS 14 or later, use the official installer:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Open the Ollama app, or start its local server from a terminal:

```bash
ollama serve
```

Download only the default Phase 3 generation model:

```bash
ollama pull qwen3.5:4b
```

## Use Phase 1

1. Place one or more text-based research PDFs in `data/papers/`.
2. Run the ingestion pipeline:

   ```bash
   rag-research-assistant ingest
   ```

3. Inspect a few chunks:

   ```bash
   rag-research-assistant inspect --limit 3
   ```

Equivalent module commands are available if the console script is not on your
path:

```bash
python -m rag_research_assistant ingest
python -m rag_research_assistant inspect --limit 3
```

Defaults can be changed explicitly:

```bash
rag-research-assistant ingest \
  --input data/papers \
  --output data/processed/chunks.jsonl \
  --chunk-size 1200 \
  --overlap 200
```

The output contains one JSON object per line:

```json
{"chunk_id":"paper.pdf:p2:c1","document":"paper.pdf","page_number":2,"chunk_index":1,"char_start":0,"char_end":1174,"text":"..."}
```

PDFs and generated chunk files are intentionally ignored by Git. This protects
private papers and keeps derived data out of version control. `.env`, keys,
virtual environments, caches, and build output are ignored as well.

## Use Phase 2A

After running Phase 1 and inspecting `chunks.jsonl`, build the local embedding
index once:

```bash
rag-research-assistant embed
```

Then search it with a natural-language question:

```bash
rag-research-assistant search \
  "What is retrieval augmented generation?" \
  --top-k 5
```

Sentence Transformers selects an available device automatically. On Apple
Silicon, you can explicitly try Metal or fall back to CPU:

```bash
rag-research-assistant embed --device mps
rag-research-assistant search "How does semantic retrieval work?" --device mps
```

The default index is:

```text
data/processed/embedding_index/
├── embeddings.npy  # float32 matrix: one row per chunk
└── manifest.json   # model, dimensions, chunk hash, row-to-chunk IDs
```

Both files are generated and ignored by Git. Re-run `embed` whenever ingestion
changes `chunks.jsonl`; search refuses to use a stale index.

## Use Phase 3

Once Ollama is running, the model is pulled, and the embedding index exists:

```bash
rag-research-assistant ask \
  "How does retrieval augmented generation reduce hallucinations?" \
  --top-k 5
```

Inspect the retrieved chunks before the LLM runs:

```bash
rag-research-assistant ask \
  "How does retrieval augmented generation reduce hallucinations?" \
  --top-k 5 \
  --show-context
```

The generation model and temperature are configurable without changing the RAG
pipeline:

```bash
rag-research-assistant ask \
  "How does RAG work?" \
  --model qwen3.5:4b \
  --temperature 0.1
```

To use another local Ollama model, pull exactly that model and pass its name to
`--model`. The embedding model does not change when the generation model changes.

## Use Phase 4

The tracked dataset is `data/evaluation/questions.jsonl`. Retrieval-only
evaluation is the default and does not contact Ollama:

```bash
rag-research-assistant evaluate
```

It evaluates 16 answerable or partially answerable questions at three cutoffs
and reports the 4 unanswerable questions separately. Use `--verbose` to see the
ranked filename/page results for every question:

```bash
rag-research-assistant evaluate --top-k 5 --verbose
```

`--top-k` is the maximum retrieval depth and must be at least 5 because Phase 4
always calculates Hit@1, Hit@3, and Hit@5. The default machine report is written
to `data/evaluation/results/latest.json`; generated reports are ignored by Git.

Run the slower end-to-end evaluation only when Ollama and the selected model are
available:

```bash
rag-research-assistant evaluate \
  --with-generation \
  --model qwen3.5:4b \
  --temperature 0.1 \
  --verbose
```

The generation report adds each answer, detected citation numbers, resolved
source mappings, invalid references, refusal detection, and per-question errors.
It does not assign a fabricated groundedness score or use an LLM as a judge.

To add a question, append one JSON object on one line:

```json
{"id":"unique_id","question":"What does the paper claim?","answerability":"answerable","expected_sources":[{"document":"paper.pdf","page_number":3}],"notes":"Why this label is justified."}
```

Allowed answerability values are `answerable`, `partially_answerable`, and
`unanswerable`. Unanswerable examples must have an empty `expected_sources`
list. `page_number` may be omitted when only the document is known; `chunk_id`
may be added when that exact chunk is intentionally part of the label.

## Use Phase 5

Choose a strategy on the existing commands; dense remains the default:

```bash
rag-research-assistant search "What is non-parametric memory?" --retriever dense
rag-research-assistant search "What is non-parametric memory?" --retriever bm25
rag-research-assistant search "What is non-parametric memory?" --retriever hybrid
```

Download the optional reranker once, then use it from the local Hugging Face cache:

```bash
rag-research-assistant reranker-download
rag-research-assistant search \
  "What is non-parametric memory?" \
  --retriever hybrid \
  --rerank
```

The same flags work with `ask` and `evaluate`. Reranking is intentionally valid
only with hybrid retrieval:

```bash
rag-research-assistant ask "How can RAG update knowledge?" \
  --retriever hybrid --rerank --show-context
rag-research-assistant evaluate --retriever hybrid --rerank --verbose
```

Run all four retrieval strategies against the unchanged dataset:

```bash
rag-research-assistant compare
```

The command writes its ignored JSON report to
`data/evaluation/results/comparison.json` and prints aggregate metrics plus the
question IDs improved or degraded from dense to hybrid and hybrid to reranked.
Generated model files stay in the normal Hugging Face cache and are never
committed.

### Measured Phase 5 retrieval results

These results use the same 20 questions, 16 retrieval-scored questions, 209
chunks, and Phase 4 source labels:

| Strategy | Hit@1 | Hit@3 | Hit@5 | Mean first-correct rank |
| --- | ---: | ---: | ---: | ---: |
| Dense | 56.2% | 68.8% | 100.0% | 2.12 |
| BM25 | 50.0% | 81.2% | 93.8% | 1.87 |
| Hybrid RRF | 68.8% | 93.8% | 93.8% | 1.33 |
| Hybrid RRF + cross-encoder | **75.0%** | **100.0%** | **100.0%** | 1.38 |

Hybrid improved five questions relative to dense but pushed
`sgpt_pooling_training` outside the top five. Reranking restored that question
at rank 2 and moved `rag_update_knowledge` from rank 2 to rank 1; it moved
`rag_corpus_chunks` from rank 2 to rank 3. This is why aggregate and
per-question comparisons both matter.

The full strongest-strategy generation check took about 174 seconds locally.
It had 4/4 recognized unanswerable refusals and no generation errors. One answer,
`rag_update_knowledge`, emitted invalid citation `[6]` with five supplied context
items. Retrieval improved, but generation remains a separate failure surface.

See [the Phase 5 learning guide](docs/phase-5-hybrid-retrieval.md) for the BM25
formula, RRF, candidate retrieval, bi-encoder versus cross-encoder behavior,
score interpretation, performance, and complete measured comparison.

## How the pipeline works

`pdf.py` produces a `PageText` object for every physical PDF page. `chunking.py`
normalizes its text and splits it near paragraph or sentence boundaries. Each
result becomes a `Chunk` with the original filename and page number copied onto
it. `pipeline.py` writes those records to `data/processed/chunks.jsonl`.

Chunks never cross a page boundary. This is a deliberate Phase 1 decision: the
retrieval result can always point to one exact page. The overlap repeats some
context between adjacent chunks so a sentence near a boundary is less likely to
lose its meaning. Character offsets refer to the normalized page text, making
chunk construction traceable.

See [the architecture note](docs/architecture.md) for the component boundaries
and tradeoffs.

## How grounded generation works

Retrieval and generation remain separate. `retrieval.py` selects chunks using
the Phase 2A embedding model and cosine similarity. `context.py` labels those
chunks as `[1]`, `[2]`, and so on. `prompting.py` combines that evidence with the
question and rules against unsupported claims. Only then does `generation.py`
send one prompt to Ollama's local `/api/generate` endpoint. `rag.py` coordinates
the stages but does not implement any of them.

The local LLM receives the grounding rules, the complete formatted text of the
retrieved chunks, their citation identifiers and metadata, and the question. It
does not receive the embedding matrix, similarity algorithm, entire PDF files,
or any unretrieved chunks.

Ollama is the local model runtime: it manages downloaded model weights, loads
them into memory, applies the model's prompt template, and performs inference.
Qwen is the generation model that predicts the answer text. Sentence
Transformers is a different model with a different job: it converts the question
and chunks into vectors for retrieval.

See [the Phase 3 learning guide](docs/phase-3-grounded-generation.md) for context
windows, grounding, temperature, and failure-mode details.

## How evaluation works

`evaluation.py` runs the selected common retriever for each curated question.
For answerable and partially answerable questions, a result matches an expected
source when its filename and every supplied optional label (page and chunk ID)
match. Hit@k is true when at least one expected source appears among the first
`k` results. First-correct rank is the earliest such position. When several
sources are expected, recall@k reports the fraction of distinct expected sources
seen by that cutoff.

Unanswerable questions are not assigned Hit@k. Cosine similarity always provides
an ordering, even when every passage is irrelevant, so merely receiving five
chunks does not prove that an answer exists. In generation mode, these examples
instead test whether the model explicitly reports insufficient evidence.

Citation validity only means a cited number maps to one of the supplied context
blocks. It does not mean that block supports the associated claim. The report
therefore preserves full retrieved text and generated answers for manual review.

See [the Phase 4 evaluation guide](docs/phase-4-evaluation.md) for metric
definitions, interpretation, limitations, and the measured baseline.

## How semantic retrieval works

The default model is
[`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1).
It is a small, 6-layer, English MiniLM model that produces 384-dimensional
vectors and was trained specifically to match questions with answer passages.
That makes it a better starting point for this asymmetric search task than the
general-purpose `all-MiniLM-L6-v2`, while remaining suitable for a laptop.

An embedding is a fixed-length numeric representation of text. Texts whose
meanings the model considers related tend to point in similar directions in the
embedding space. Query and chunk embeddings must come from the same model:
coordinates from different learned spaces have no shared meaning.

`retrieval.py` computes cosine similarity explicitly:

```text
cosine(query, chunk) = dot(query, chunk) / (norm(query) * norm(chunk))
```

Cosine similarity compares vector direction rather than magnitude. The scores
are sorted from highest to lowest, and top-k means returning only the best `k`
chunks. Semantic search can connect related wording such as “car” and “vehicle”
without requiring a literal shared term. BM25 instead rewards shared tokens,
which helps exact names, identifiers, and technical phrases. Phase 5 combines
their ranks and measures both strengths rather than assuming one always wins.

See [the Phase 2A learning guide](docs/phase-2a-retrieval.md) for a more detailed
explanation of embeddings, dimensions, cosine similarity, top-k, and model
limitations.

## Limitations to understand

- `pypdf` extracts an existing PDF text layer; it does not perform OCR. Scanned
  image-only papers need a separate OCR stage later.
- PDF reading order can be imperfect for multi-column layouts, tables, formulas,
  headers, and footnotes. Always inspect representative chunks.
- Character chunk size is transparent but not the same as embedding-model token
  count. The selected model may truncate long chunks, so token lengths and
  retrieval quality should be measured before tuning this policy.
- Repeated headers, footers, references, and hyphenation are not semantically
  cleaned in Phase 1. Those are useful future ingestion improvements.
- The embedding model is English-focused and not specialized for scientific
  notation or every research domain.
- Cosine scores are relative ranking signals, not probabilities or proof that a
  passage answers a question.
- BM25 uses exact tokens without stemming or lemmatization, so morphological
  variants and synonyms can remain disconnected.
- Raw cosine, BM25, RRF, and cross-encoder scores have different meanings and
  scales. None is a probability, and their magnitudes must not be compared.
- The selected reranker was trained on MS MARCO and truncates long pairs at 512
  tokens; research-paper terminology and evidence outside that window can be
  judged imperfectly.
- NumPy search scans every row. This is deliberately understandable and adequate
  for a learning corpus, but it is not an approximate index for large datasets.
- Retrieval can select related but non-answering chunks. A grounded prompt cannot
  repair missing evidence.
- A local LLM can ignore instructions, misuse citations, or hallucinate despite
  grounding. Citations must still be checked against the displayed sources.
- Increasing top-k adds evidence but also consumes context-window space and can
  introduce distracting passages.
- Hit@k measures retrieval of manually labelled pages, not complete system
  accuracy or answer quality.
- Page-level labels can mark a retrieved chunk as a hit even when another chunk
  on the same page contains the strongest wording.
- Refusal detection is phrase matching and can miss a valid refusal or classify
  unrelated cautious wording as a refusal.
- Citation-reference validation checks numbering only, not whether a citation
  semantically supports the claim.
- The 20-question set is small, English-only, and tied to three papers. Its main
  value is repeatable comparison, not a universal quality estimate.

## Common Phase 3 failures

- **Ollama missing:** install it with the command in Setup.
- **Server unavailable:** open Ollama or run `ollama serve`.
- **Model missing:** run `ollama pull qwen3.5:4b`, or pull the exact name passed
  to `--model`.
- **Embedding model cache missing:** run `rag-research-assistant embed` while
  online once; query-time embedding loading is intentionally cache-only.
- **Reranker cache missing:** run `rag-research-assistant reranker-download`
  while online once; retrieval-time reranker loading is cache-only.
- **Stale index:** rerun `rag-research-assistant embed` after ingestion changes.
- **Weak answer with valid generation:** inspect `--show-context`. If the needed
  evidence is absent, this is primarily a retrieval failure.
- **Timeout, empty response, or Ollama HTTP error:** this is a generation/runtime
  failure after retrieval has succeeded.

## Tests

```bash
pytest
```

## Repository layout

```text
rag-research-assistant/
├── data/
│   ├── evaluation/      # tracked questions; ignored result reports
│   ├── papers/          # private local PDFs (ignored)
│   └── processed/       # generated chunks and embeddings (ignored)
├── docs/                # architecture notes
├── src/rag_research_assistant/
│   ├── bm25.py          # transparent lexical ranking and corpus statistics
│   ├── chunking.py      # normalization and chunk construction
│   ├── cli.py           # local pipeline and evaluation commands
│   ├── comparison.py    # aggregate metrics and per-question rank deltas
│   ├── context.py       # source-labelled context and citation mapping
│   ├── embeddings.py    # local Sentence Transformers adapter
│   ├── evaluation.py    # dataset, metrics, heuristics, and reports
│   ├── generation.py    # generic generator interface and Ollama adapter
│   ├── hybrid.py        # Reciprocal Rank Fusion
│   ├── index.py         # NumPy persistence and integrity checks
│   ├── models.py        # explicit pipeline data contracts
│   ├── pdf.py           # PDF discovery and page extraction
│   ├── pipeline.py      # ingestion orchestration and JSONL persistence
│   ├── prompting.py     # grounded prompt construction
│   ├── rag.py           # answer orchestration
│   ├── reranking.py     # optional local cross-encoder adapter
│   ├── retrieval.py     # manual cosine similarity and top-k ranking
│   └── retrievers.py    # shared, composable retrieval strategies
├── tests/
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Roadmap

Phase 5 establishes a measured hybrid-and-reranking pipeline but deliberately
omits RAG frameworks, vector databases, query rewriting, web search, LLM judges,
servers, and UI. A later phase can improve one component at a time against the
same fixed questions instead of relying on impressions.
