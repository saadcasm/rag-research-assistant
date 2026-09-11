"""Text normalization and boundary-aware character chunking."""

import re
from typing import List, Tuple

from .models import Chunk, PageText


def normalize_text(text: str) -> str:
    """Normalize extraction noise while retaining paragraph boundaries."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _preferred_end(text: str, start: int, maximum_end: int, minimum_end: int) -> int:
    """Find a readable boundary without making a very small chunk."""

    for delimiter in ("\n\n", ". ", "? ", "! ", "\n", " "):
        position = text.rfind(delimiter, minimum_end, maximum_end)
        if position != -1:
            return position + len(delimiter)
    return maximum_end


def split_text(text: str, chunk_size: int = 1_200, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """Split normalized text into `(start, end, text)` tuples.

    Character units keep Phase 1 transparent. Boundary selection prefers
    paragraphs and sentences, and overlap repeats context between neighbors.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(normalized):
        maximum_end = min(start + chunk_size, len(normalized))
        if maximum_end == len(normalized):
            end = maximum_end
        else:
            minimum_end = start + max(1, chunk_size // 2)
            end = _preferred_end(normalized, start, maximum_end, minimum_end)

        trimmed_end = end
        while trimmed_end > start and normalized[trimmed_end - 1].isspace():
            trimmed_end -= 1
        chunks.append((start, trimmed_end, normalized[start:trimmed_end]))

        if end >= len(normalized):
            break

        raw_next_start = max(0, trimmed_end - overlap)
        aligned_start = raw_next_start
        while aligned_start < trimmed_end and not normalized[aligned_start].isspace():
            aligned_start += 1
        next_start = aligned_start if aligned_start < trimmed_end else raw_next_start
        while next_start < trimmed_end and normalized[next_start].isspace():
            next_start += 1
        start = next_start if next_start > start else trimmed_end

    return chunks


def chunk_page(page: PageText, chunk_size: int = 1_200, overlap: int = 200) -> List[Chunk]:
    """Chunk a page and copy its source metadata onto every result."""

    chunks = []
    for chunk_index, (start, end, text) in enumerate(
        split_text(page.text, chunk_size=chunk_size, overlap=overlap), start=1
    ):
        chunks.append(
            Chunk(
                chunk_id=f"{page.document}:p{page.page_number}:c{chunk_index}",
                document=page.document,
                page_number=page.page_number,
                chunk_index=chunk_index,
                char_start=start,
                char_end=end,
                text=text,
            )
        )
    return chunks
