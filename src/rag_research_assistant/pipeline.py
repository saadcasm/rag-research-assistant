"""Phase 1 orchestration: PDFs to metadata-rich JSONL chunks."""

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .chunking import CHUNKING_STRATEGIES, chunk_pages
from .models import Chunk, PageText
from .pdf import extract_directory


def build_chunks(
    input_dir: Path,
    chunk_size: int = 1_200,
    overlap: int = 200,
    strategy: str = "legacy",
    sentence_embedder: Optional[Callable[[List[str]], object]] = None,
) -> List[Chunk]:
    """Extract and chunk every PDF in a directory."""

    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    by_document: dict[str, List[PageText]] = {}
    for page in extract_directory(input_dir):
        by_document.setdefault(page.document, []).append(page)
    chunks: List[Chunk] = []
    for pages in by_document.values():
        chunks.extend(
            chunk_pages(
                pages, chunk_size=chunk_size, overlap=overlap,
                strategy=strategy, sentence_embedder=sentence_embedder,
            )
        )
    return chunks


def write_jsonl(chunks: Iterable[Chunk], output_path: Path) -> int:
    """Write one chunk per line and return the number written."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(input_path: Path) -> List[Chunk]:
    """Load chunks previously written by this pipeline."""

    with input_path.open(encoding="utf-8") as input_file:
        return [Chunk(**json.loads(line)) for line in input_file if line.strip()]
