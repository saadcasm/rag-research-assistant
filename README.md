# RAG Research Assistant

A portfolio project for learning retrieval-augmented generation by implementing
its fundamental components directly. The project currently covers ingestion and
local semantic retrieval:

```text
PDF -> extracted pages -> chunks + metadata -> embeddings -> cosine search
```

No RAG framework, vector database, LLM, agent, or web UI is used yet.

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
│   ├── cli.py           # ingest, inspect, embed, and search commands
│   ├── embeddings.py    # local Sentence Transformers adapter
│   ├── index.py         # NumPy persistence and integrity checks
│   ├── models.py        # explicit pipeline data contracts
│   ├── pdf.py           # PDF discovery and page extraction
│   ├── pipeline.py      # ingestion orchestration and JSONL persistence
│   └── retrieval.py     # manual cosine similarity and top-k ranking
├── tests/
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Roadmap

Phase 2A stops at retrieval. A later phase can evaluate retrieval quality and
only then introduce an LLM answering layer. Vector databases and RAG frameworks
remain intentionally out of scope until the underlying mechanics are understood.
