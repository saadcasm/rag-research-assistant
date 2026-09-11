"""Phase 1 orchestration: PDFs to metadata-rich JSONL chunks."""

import json
from pathlib import Path
from typing import Iterable, List

from .chunking import chunk_page
from .models import Chunk
from .pdf import extract_directory


def build_chunks(input_dir: Path, chunk_size: int = 1_200, overlap: int = 200) -> List[Chunk]:
    """Extract and chunk every PDF in a directory."""

    chunks: List[Chunk] = []
    for page in extract_directory(input_dir):
        chunks.extend(chunk_page(page, chunk_size=chunk_size, overlap=overlap))
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

