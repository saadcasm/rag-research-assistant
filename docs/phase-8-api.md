# Phase 8: FastAPI application layer

Phase 8 exposes the existing local RAG pipeline over HTTP without moving
retrieval or generation logic into route functions.

```text
CLI ---------+
             v
    application.load_retriever
             |
             v
      rag.answer_question
             ^
             |
FastAPI -> RAGApplication
```

`application.py` is the shared application boundary. It constructs the existing
`Retriever` graph and owns process-lifetime resources. `rag.py` remains the one
place that retrieves evidence, builds grounded context and a prompt, calls the
generator, and returns `GroundedAnswer`. `api.py` only validates HTTP input,
calls that service, serializes the result, and maps failures to HTTP responses.

## Install and run

```bash
python -m pip install -e ".[dev]"
uvicorn rag_research_assistant.api:app --reload
```

The default API configuration targets the Phase 7.5 legacy artifacts:

```text
chunks:      data/processed/corpus-52/chunks-legacy.jsonl
Qdrant path: data/processed/corpus-52/qdrant-legacy
collection:  phase75_legacy
generator:   qwen3.5:4b at http://127.0.0.1:11434
```

Start Ollama and ensure `qwen3.5:4b` is installed before starting Uvicorn.
Interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`; ReDoc is
at `http://127.0.0.1:8000/redoc`. `--reload` is useful during development but
reloads the models whenever source files change, so omit it for a steady process.

| Environment variable | Default |
|---|---|
| `RAG_CHUNKS_PATH` | `data/processed/corpus-52/chunks-legacy.jsonl` |
| `RAG_QDRANT_PATH` | `data/processed/corpus-52/qdrant-legacy` |
| `RAG_QDRANT_COLLECTION` | `phase75_legacy` |
| `RAG_OLLAMA_MODEL` | `qwen3.5:4b` |
| `RAG_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `RAG_OLLAMA_TIMEOUT` | `180` |
| `RAG_DEVICE` | automatic |

These are process settings, not request options. A request cannot switch the
embedding model and accidentally compare incompatible query and stored vectors.

## Endpoints

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "live": true,
  "ready": true,
  "dependencies": {"qdrant": "ready", "ollama": "ready"}
}
```

**Liveness** means the HTTP process can respond. **Readiness** means its
process-scoped RAG service completed startup with usable Qdrant and Ollama
dependencies. `/health` returns HTTP 200 as a cheap liveness endpoint; clients
must inspect `ready`. Dependency values are a startup snapshot rather than a new
Qdrant query and Ollama request on every health check. A later dependency failure
is returned by `/ask` and written to server logs.

### `POST /ask`

`POST` is appropriate because a question is structured input that starts
expensive retrieval and generation work. It does not mutate the corpus. `GET`
fits the cheap, cache-friendly health read.

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How does DPR represent passages?","top_k":5}'
```

`question` is required, whitespace-stripped, and limited to 4,000 characters.
`top_k` defaults to 5 and must be a JSON integer from 1 to 20. Extra fields are
rejected so misspelled options do not silently do nothing.

```json
{
  "answer": "DPR represents passages as dense vectors [1].",
  "sources": [
    {
      "citation_number": 1,
      "score": 5.93,
      "document": "dpr_karpukhin_2020.pdf",
      "start_page": 2,
      "end_page": 2,
      "chunk_id": "dpr_karpukhin_2020.pdf:p2:c6"
    }
  ],
  "metadata": {
    "top_k": 5,
    "retrieval_strategy": "hybrid+rerank",
    "retrieval_score_type": "cross-encoder",
    "dense_backend": "qdrant",
    "generation_model": "qwen3.5:4b",
    "embedding_model": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "indexed_chunks": 3793
  }
}
```

## Pydantic and HTTP errors

Pydantic models form the boundary between untrusted JSON and typed Python.
FastAPI uses them for input validation, output serialization, and the OpenAPI
schema behind `/docs`. Endpoint code therefore receives a known shape, while
humans and generated clients can discover the same contract.

| Status | Meaning here |
|---|---|
| `200` | Health response or completed grounded answer |
| `422` | Invalid schema: blank question, wrong type, extra field, or invalid `top_k` |
| `502` | Ollama was reached but generation failed or returned an invalid response |
| `503` | The RAG service, Ollama, or Qdrant is unavailable or incompatible |
| `500` | Unexpected internal failure |

Responses contain stable codes and safe messages. Detailed exceptions and local
filesystem paths stay in server logs rather than leaking to API clients.

## Lifecycle and concurrency

FastAPI's lifespan hook constructs these once per process:

- the legacy chunks and immutable BM25 statistics;
- the cached SentenceTransformer query model;
- the validated Qdrant collection and open client;
- the cached cross-encoder reranker;
- the Ollama generator and startup model check.

Requests contain only validated question data and reuse the loaded service.
Shutdown closes Qdrant. Multiple Uvicorn workers each load their own complete
resource set, so memory use grows with worker count.

`/ask` is synchronous, so FastAPI runs it in a worker thread instead of blocking
the event loop. The service uses one process-local lock around retrieval and
generation. This conservative Phase 8 choice protects local model objects,
embedded Qdrant, and Ollama from uncertain concurrent access, but limits one
process to one answer at a time. Later work can benchmark queues, narrower locks,
separate inference workers, or server-mode Qdrant before increasing concurrency.

## Tests and real smoke test

API tests inject a fake service and need no PDFs, Qdrant data, network, model
downloads, or Ollama. They cover health, valid answers, source serialization,
blank and mistyped input, extra fields, bounded `top_k`, startup failure,
generation failure, and safe error content.

The real integration smoke test used Uvicorn, the 3,793-chunk Phase 7.5 legacy
Qdrant collection, both cached local models, and Ollama with `qwen3.5:4b`.
`GET /health` was ready, OpenAPI listed both routes, and `POST /ask` answered
"How does DPR represent passages?" in about 12.2 seconds. It returned five DPR
sources and cited page 2 for passage vectors and page 3 for dot-product scoring.

## Deliberate limits

Phase 8 adds no authentication, rate limiting, streaming, request history,
runtime ingestion, file upload, background jobs, or production deployment.
`POST /ingest` is deferred because PDFs, chunks, embeddings, Qdrant, and BM25
would need an atomic, potentially long-running synchronization workflow.
Retrieval settings and generation prompts are unchanged.
