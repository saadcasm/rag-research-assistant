# RAG Research Assistant

A portfolio project for learning retrieval-augmented generation by implementing
its fundamental components directly. The project currently covers ingestion,
local semantic retrieval, and grounded local answer generation:

```text
PDF -> pages -> chunks -> embeddings -> retrieval -> context -> local LLM -> cited answer
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
without requiring a literal shared term. It can still miss exact identifiers or
specialized meanings, so keyword and hybrid retrieval remain useful later.

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
- NumPy search scans every row. This is deliberately understandable and adequate
  for a learning corpus, but it is not an approximate index for large datasets.
- Retrieval can select related but non-answering chunks. A grounded prompt cannot
  repair missing evidence.
- A local LLM can ignore instructions, misuse citations, or hallucinate despite
  grounding. Citations must still be checked against the displayed sources.
- Increasing top-k adds evidence but also consumes context-window space and can
  introduce distracting passages.

## Common Phase 3 failures

- **Ollama missing:** install it with the command in Setup.
- **Server unavailable:** open Ollama or run `ollama serve`.
- **Model missing:** run `ollama pull qwen3.5:4b`, or pull the exact name passed
  to `--model`.
- **Embedding model cache missing:** run `rag-research-assistant embed` while
  online once; query-time embedding loading is intentionally cache-only.
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
│   ├── papers/          # private local PDFs (ignored)
│   └── processed/       # generated chunks (ignored)
├── docs/                # architecture notes
├── src/rag_research_assistant/
│   ├── chunking.py      # normalization and chunk construction
│   ├── cli.py           # ingest, inspect, embed, search, and ask commands
│   ├── context.py       # source-labelled context and citation mapping
│   ├── embeddings.py    # local Sentence Transformers adapter
│   ├── generation.py    # generic generator interface and Ollama adapter
│   ├── index.py         # NumPy persistence and integrity checks
│   ├── models.py        # explicit pipeline data contracts
│   ├── pdf.py           # PDF discovery and page extraction
│   ├── pipeline.py      # ingestion orchestration and JSONL persistence
│   ├── prompting.py     # grounded prompt construction
│   ├── rag.py           # answer orchestration
│   └── retrieval.py     # manual cosine similarity and top-k ranking
├── tests/
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Roadmap

Phase 3 completes a first local RAG loop but deliberately omits memory, chat
history, web search, hybrid search, reranking, vector databases, and frameworks.
The next useful step is evaluation: measure retrieval relevance, citation
correctness, faithfulness, and answer quality before adding infrastructure.
