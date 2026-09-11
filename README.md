# RAG Research Assistant

A portfolio project for learning retrieval-augmented generation by implementing
its fundamental components directly. Phase 1 covers only the ingestion path:

```text
PDF -> page-level extracted text -> overlapping chunks + source metadata
```

No RAG framework, embeddings, vector database, LLM, agent, or web UI is used yet.

## Phase 1 features

- Discovers multiple PDFs in `data/papers/` in deterministic order.
- Extracts text page by page with `pypdf`.
- Preserves the document filename and one-based page number.
- Normalizes common PDF whitespace noise.
- Creates readable, overlapping character chunks without crossing page boundaries.
- Writes chunks as inspectable JSON Lines records.
- Provides a CLI for processing papers and printing sample chunks.
- Tests chunking, metadata preservation, PDF discovery, and serialization.

## Setup

Python 3.9 or newer is required.

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

## Limitations to understand

- `pypdf` extracts an existing PDF text layer; it does not perform OCR. Scanned
  image-only papers need a separate OCR stage later.
- PDF reading order can be imperfect for multi-column layouts, tables, formulas,
  headers, and footnotes. Always inspect representative chunks.
- Character chunk size is transparent but not the same as embedding-model token
  count. Phase 2 should choose an embedding model, measure token lengths, and
  evaluate retrieval quality before tuning this policy.
- Repeated headers, footers, references, and hyphenation are not semantically
  cleaned in Phase 1. Those are useful future ingestion improvements.

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
│   ├── cli.py           # ingest and inspect commands
│   ├── models.py        # explicit pipeline data contracts
│   ├── pdf.py           # PDF discovery and page extraction
│   └── pipeline.py      # orchestration and JSONL persistence
├── tests/
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Roadmap

Phase 2 will build on the JSONL chunks to introduce embeddings and retrieval.
It should begin only after sample chunks have been inspected and the effects of
chunk size, overlap, page boundaries, extraction quality, and metadata have been
understood.
