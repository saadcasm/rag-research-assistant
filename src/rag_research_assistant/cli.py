"""Command-line interface for ingestion, embedding, and semantic search."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .embeddings import DEFAULT_MODEL, SentenceTransformerEmbedder
from .index import build_index, load_index
from .pdf import discover_pdfs
from .pipeline import build_chunks, read_jsonl, write_jsonl
from .retrieval import search


DEFAULT_INPUT = Path("data/papers")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX = Path("data/processed/embedding_index")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG ingestion and semantic retrieval")
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
    return parser


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
        raise ValueError("preview_chars must be positive")
    index = load_index(args.chunks, args.index)
    embedder = SentenceTransformerEmbedder(
        model_name=index.model_name,
        device=args.device,
        local_files_only=True,
    )
    results = search(args.query, index, embedder, top_k=args.top_k)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview_text = " ".join(chunk.text.split())
        preview = preview_text[: args.preview_chars]
        if len(preview) < len(preview_text):
            preview += "..."
        print(f"\n{rank}. score={result.score:.4f}")
        print(f"   source={chunk.document} page={chunk.page_number} chunk={chunk.chunk_id}")
        print(f"   {preview}")
    print(f"\nReturned {len(results)} result(s) for: {args.query}")
    return 0
