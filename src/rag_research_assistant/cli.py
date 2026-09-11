"""Command-line interface for ingestion, retrieval, and grounded generation."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from .generation import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEMPERATURE,
    GenerationError,
    OllamaGenerator,
)
from .index import InvalidIndexError, build_index, load_index
from .models import SearchResult
from .pdf import discover_pdfs
from .pipeline import build_chunks, read_jsonl, write_jsonl
from .rag import answer_question
from .retrieval import search


DEFAULT_INPUT = Path("data/papers")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX = Path("data/processed/embedding_index")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local RAG ingestion, retrieval, and grounded generation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Extract and chunk local PDFs")
    ingest.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ingest.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ingest.add_argument("--chunk-size", type=int, default=1_200)
    ingest.add_argument("--overlap", type=int, default=200)

    inspect = subparsers.add_parser("inspect", help="Print sample chunks")
    inspect.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    inspect.add_argument("--limit", type=int, default=3)

    embed = subparsers.add_parser("embed", help="Build a persistent local embedding index")
    embed.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    embed.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    embed.add_argument("--model", default=DEFAULT_MODEL)
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--device", help="Optional Sentence Transformers device, e.g. cpu or mps")

    search_parser = subparsers.add_parser("search", help="Search indexed chunks semantically")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    search_parser.add_argument(
        "--device", help="Optional Sentence Transformers device, e.g. cpu or mps"
    )
    search_parser.add_argument("--preview-chars", type=int, default=400)

    ask = subparsers.add_parser("ask", help="Answer from retrieved evidence with local Ollama")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--chunks", type=Path, default=DEFAULT_OUTPUT)
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask.add_argument("--device", help="Optional embedding device, e.g. cpu or mps")
    ask.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    ask.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ask.add_argument("--timeout", type=float, default=180.0)
    ask.add_argument("--show-context", action="store_true")
    ask.add_argument("--preview-chars", type=int, default=400)
    return parser


def _print_retrieval(results: List[SearchResult], preview_chars: int) -> None:
    print("\nRetrieved context")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview_text = " ".join(chunk.text.split())
        preview = preview_text[:preview_chars]
        if len(preview) < len(preview_text):
            preview += "..."
        print(f"\n{rank}. score={result.score:.4f}")
        print(f"   source={chunk.document} page={chunk.page_number} chunk={chunk.chunk_id}")
        print(f"   {preview}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "ingest":
        pdfs = discover_pdfs(args.input)
        if not pdfs:
            print(f"No PDF files found in {args.input}")
            return 0
        chunks = build_chunks(args.input, chunk_size=args.chunk_size, overlap=args.overlap)
        count = write_jsonl(chunks, args.output)
        print(f"Processed {len(pdfs)} PDF(s) into {count} chunk(s): {args.output}")
        return 0

    if args.command == "inspect":
        chunks = read_jsonl(args.input)
        for chunk in chunks[: max(0, args.limit)]:
            print(f"\n[{chunk.chunk_id}] {chunk.document}, page {chunk.page_number}")
            print(f"characters {chunk.char_start}:{chunk.char_end}")
            print(chunk.text)
        print(f"\nShowing {min(max(0, args.limit), len(chunks))} of {len(chunks)} chunk(s).")
        return 0

    if args.command == "embed":
        embedder = SentenceTransformerEmbedder(
            model_name=args.model, batch_size=args.batch_size, device=args.device
        )
        index = build_index(args.chunks, args.index, embedder)
        print(
            f"Embedded {len(index.chunks)} chunk(s) as a "
            f"{index.embeddings.shape} matrix with {index.model_name}: {args.index}"
        )
        return 0

    if args.preview_chars <= 0:
        print("Error: preview_chars must be positive", file=sys.stderr)
        return 2

    if args.command == "ask":
        if not 0.0 <= args.temperature <= 2.0:
            print("Error: temperature must be between 0 and 2", file=sys.stderr)
            return 2
        try:
            generator = OllamaGenerator(
                model_name=args.model,
                timeout=args.timeout,
            )
            generator.ensure_model_available()
        except (GenerationError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        try:
            index = load_index(args.chunks, args.index)
            embedder = SentenceTransformerEmbedder(
                model_name=index.model_name,
                device=args.device,
                local_files_only=True,
            )
        except (FileNotFoundError, InvalidIndexError, OSError, ValueError) as exc:
            print(f"Error: could not load the embedding index: {exc}", file=sys.stderr)
            return 2
        debug_callback = None
        if args.show_context:
            debug_callback = lambda results: _print_retrieval(
                results, args.preview_chars
            )
        try:
            grounded_answer = answer_question(
                args.question,
                index,
                embedder,
                generator,
                top_k=args.top_k,
                temperature=args.temperature,
                on_retrieved=debug_callback,
            )
        except (GenerationError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

        print("\nAnswer\n")
        print(grounded_answer.text)
        print("\nSources")
        if not grounded_answer.sources:
            print("No sources retrieved.")
        for source in grounded_answer.sources:
            print(
                f"[{source.citation_number}] {source.document}, page {source.page_number}, "
                f"chunk {source.chunk_id} (score={source.score:.4f})"
            )
        return 0

    index = load_index(args.chunks, args.index)
    embedder = SentenceTransformerEmbedder(
        model_name=index.model_name,
        device=args.device,
        local_files_only=True,
    )
    results = search(args.query, index, embedder, top_k=args.top_k)
    _print_retrieval(results, args.preview_chars)
    print(f"\nReturned {len(results)} result(s) for: {args.query}")
    return 0
