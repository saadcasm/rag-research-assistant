"""Command-line interface for ingestion and chunk inspection."""

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .pdf import discover_pdfs
from .pipeline import build_chunks, read_jsonl, write_jsonl


DEFAULT_INPUT = Path("data/papers")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1 of the RAG Research Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Extract and chunk local PDFs")
    ingest.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ingest.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ingest.add_argument("--chunk-size", type=int, default=1_200)
    ingest.add_argument("--overlap", type=int, default=200)

    inspect = subparsers.add_parser("inspect", help="Print sample chunks")
    inspect.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    inspect.add_argument("--limit", type=int, default=3)
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

    chunks = read_jsonl(args.input)
    for chunk in chunks[: max(0, args.limit)]:
        print(f"\n[{chunk.chunk_id}] {chunk.document}, page {chunk.page_number}")
        print(f"characters {chunk.char_start}:{chunk.char_end}")
        print(chunk.text)
    print(f"\nShowing {min(max(0, args.limit), len(chunks))} of {len(chunks)} chunk(s).")
    return 0

