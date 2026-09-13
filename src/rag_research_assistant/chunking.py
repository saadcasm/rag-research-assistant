"""Deterministic chunking strategies, from the Phase 1 baseline to semantic groups."""

import hashlib
import re
from dataclasses import dataclass
from statistics import median
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .models import Chunk, PageText

CHUNKING_STRATEGIES = ("legacy", "boundary", "structural", "semantic")
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+|$)")
_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?\s+)|(?:[IVXLC]+\.\s+))?"
    r"(?:abstract|introduction|related work|background|methods?|methodology|"
    r"experiments?|results?|discussion|conclusion|references|[A-Z][A-Za-z -]{2,80})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TextUnit:
    text: str
    start_page: int
    end_page: int
    char_start: int
    char_end: int
    section_title: Optional[str] = None


def normalize_text(text: str) -> str:
    """Normalize extraction noise while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _preferred_end(text: str, start: int, maximum_end: int, minimum_end: int) -> int:
    for delimiter in ("\n\n", ". ", "? ", "! ", "\n", " "):
        position = text.rfind(delimiter, minimum_end, maximum_end)
        if position != -1:
            return position + len(delimiter)
    return maximum_end


def split_text(text: str, chunk_size: int = 1_200, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """The unchanged Phase 1 page-scoped character baseline."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    normalized = normalize_text(text)
    if not normalized:
        return []
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < len(normalized):
        maximum_end = min(start + chunk_size, len(normalized))
        end = maximum_end if maximum_end == len(normalized) else _preferred_end(normalized, start, maximum_end, start + max(1, chunk_size // 2))
        trimmed_end = end
        while trimmed_end > start and normalized[trimmed_end - 1].isspace():
            trimmed_end -= 1
        chunks.append((start, trimmed_end, normalized[start:trimmed_end]))
        if end >= len(normalized):
            break
        next_start = max(0, trimmed_end - overlap)
        while next_start < trimmed_end and not normalized[next_start].isspace():
            next_start += 1
        while next_start < trimmed_end and normalized[next_start].isspace():
            next_start += 1
        start = next_start if next_start > start else trimmed_end
    return chunks


def chunk_page(page: PageText, chunk_size: int = 1_200, overlap: int = 200) -> List[Chunk]:
    """Phase 1 compatibility entry point, including its stable IDs."""
    return [Chunk(f"{page.document}:p{page.page_number}:c{index}", page.document, page.page_number, index, start, end, text)
            for index, (start, end, text) in enumerate(split_text(page.text, chunk_size, overlap), 1)]


def detect_heading(line: str) -> Optional[str]:
    """Conservative heuristic; PDF extraction does not expose true heading styles."""
    candidate = " ".join(line.split())
    if not candidate or len(candidate) > 100 or candidate.endswith((".", ",", ";")):
        return None
    return candidate if _HEADING_RE.fullmatch(candidate) else None


def _split_sentences(unit: TextUnit) -> List[TextUnit]:
    parts = [part.strip() for part in _SENTENCE_RE.split(unit.text) if part.strip()]
    result, cursor = [], unit.char_start
    for part in parts:
        result.append(TextUnit(part, unit.start_page, unit.end_page, cursor, cursor + len(part), unit.section_title))
        cursor += len(part) + 1
    return result


def _boundary_units(pages: Sequence[PageText], structural: bool) -> List[TextUnit]:
    result: List[TextUnit] = []
    section: Optional[str] = None
    offset = 0
    for page in pages:
        text = normalize_text(page.text)
        for paragraph in (part.strip() for part in text.split("\n\n") if part.strip()):
            # pypdf commonly keeps a heading as a single newline-delimited line
            # immediately before body text, so inspect both paragraph and lines.
            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
            body_lines: List[str] = []
            for line in lines:
                heading = detect_heading(line) if structural else None
                if heading:
                    section = heading
                else:
                    body_lines.append(line)
            body = " ".join(body_lines)
            if body:
                unit = TextUnit(body, page.page_number, page.page_number, offset, offset + len(body), section)
                result.extend(_split_sentences(unit))
            offset += len(paragraph) + 2
        offset += 2
    return result


def _hard_split(unit: TextUnit, maximum: int) -> List[TextUnit]:
    if len(unit.text) <= maximum:
        return [unit]
    return [TextUnit(unit.text[start:start + maximum], unit.start_page, unit.end_page, unit.char_start + start,
                     unit.char_start + min(start + maximum, len(unit.text)), unit.section_title)
            for start in range(0, len(unit.text), maximum)]


def _combine(units: Sequence[TextUnit]) -> TextUnit:
    return TextUnit(" ".join(unit.text for unit in units), units[0].start_page, units[-1].end_page,
                    units[0].char_start, units[-1].char_end, units[-1].section_title or units[0].section_title)


def _pack_units(units: Iterable[TextUnit], chunk_size: int, overlap: int, allow_cross_page: bool) -> List[TextUnit]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("invalid chunk_size/overlap")
    output: List[TextUnit] = []
    current: List[TextUnit] = []
    for original in units:
        for unit in _hard_split(original, chunk_size):
            if current and not allow_cross_page and unit.start_page != current[-1].end_page:
                output.append(_combine(current)); current = []
            current_length = sum(len(item.text) for item in current) + max(0, len(current) - 1)
            if current and current_length + 1 + len(unit.text) > chunk_size:
                output.append(_combine(current))
                retained: List[TextUnit] = []; retained_length = 0
                for item in reversed(current):
                    if retained_length + len(item.text) > overlap: break
                    retained.insert(0, item); retained_length += len(item.text) + 1
                current = retained
                retained_size = sum(len(item.text) for item in current) + max(0, len(current) - 1)
                if retained_size + 1 + len(unit.text) > chunk_size:
                    current = []
            current.append(unit)
    if current: output.append(_combine(current))
    return output


def _semantic_units(pages: Sequence[PageText], chunk_size: int, overlap: int, embedder: Callable[[List[str]], object], threshold: float = 0.35) -> List[TextUnit]:
    sentences = _boundary_units(pages, structural=True)
    if not sentences: return []
    vectors = np.asarray(embedder([item.text for item in sentences]), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(sentences):
        raise ValueError("semantic sentence embedder returned an invalid matrix")
    norms = np.linalg.norm(vectors, axis=1)
    denominator = norms[:-1] * norms[1:]
    similarities = np.divide((vectors[:-1] * vectors[1:]).sum(axis=1), denominator, out=np.zeros(len(vectors) - 1), where=denominator != 0)
    groups: List[TextUnit] = []; start = 0
    for index, similarity in enumerate(similarities, 1):
        if similarity < threshold or sentences[index].section_title != sentences[index - 1].section_title:
            groups.append(_combine(sentences[start:index])); start = index
    groups.append(_combine(sentences[start:]))
    # A single low-similarity sentence is not enough evidence for a useful
    # retrieval unit. Merge small neighbouring topic groups until they reach a
    # transparent minimum, then only hard-split oversized groups.
    minimum = max(200, chunk_size // 3)
    bounded: List[TextUnit] = []
    pending: List[TextUnit] = []
    for group in groups:
        pending.append(group)
        if sum(len(item.text) for item in pending) >= minimum:
            bounded.append(_combine(pending))
            pending = []
    if pending:
        if bounded:
            bounded[-1] = _combine([bounded[-1], *pending])
        else:
            bounded.append(_combine(pending))
    return [piece for group in bounded for piece in _hard_split(group, chunk_size)]


def _chunks_from_units(document: str, units: Sequence[TextUnit], strategy: str) -> List[Chunk]:
    chunks = []
    for index, unit in enumerate(units, 1):
        fingerprint = hashlib.sha1(unit.text.encode("utf-8")).hexdigest()[:10]
        chunks.append(Chunk(f"{document}:{strategy}:c{index}:{fingerprint}", document, unit.start_page, index,
                            unit.char_start, unit.char_end, unit.text, unit.start_page, unit.end_page,
                            unit.section_title, strategy))
    return chunks


def chunk_pages(pages: Sequence[PageText], *, chunk_size: int = 1200, overlap: int = 200,
                strategy: str = "legacy", sentence_embedder: Optional[Callable[[List[str]], object]] = None) -> List[Chunk]:
    """Chunk one document through a named strategy; legacy stays reproducible."""
    if strategy not in CHUNKING_STRATEGIES: raise ValueError(f"unknown chunking strategy: {strategy}")
    ordered = sorted(pages, key=lambda page: page.page_number)
    if not ordered: return []
    if strategy == "legacy": return [chunk for page in ordered for chunk in chunk_page(page, chunk_size, overlap)]
    if strategy == "boundary": units = _pack_units(_boundary_units(ordered, False), chunk_size, overlap, False)
    elif strategy == "structural": units = _pack_units(_boundary_units(ordered, True), chunk_size, overlap, True)
    else:
        if sentence_embedder is None: raise ValueError("semantic chunking requires a sentence embedder")
        units = _semantic_units(ordered, chunk_size, overlap, sentence_embedder)
    return _chunks_from_units(ordered[0].document, units, strategy)


def chunk_statistics(chunks: Sequence[Chunk]) -> dict[str, object]:
    """Small, serializable facts for fair strategy comparisons."""
    sizes = [len(chunk.text) for chunk in chunks]
    sentence_counts = [len([part for part in _SENTENCE_RE.split(chunk.text) if part.strip()]) for chunk in chunks]
    return {"total_chunks": len(chunks), "average_characters": sum(sizes) / len(sizes) if sizes else 0,
            "median_characters": median(sizes) if sizes else 0, "min_characters": min(sizes) if sizes else 0,
            "max_characters": max(sizes) if sizes else 0,
            "cross_page_chunks": sum(chunk.start_page != chunk.end_page for chunk in chunks),
            "chunks_with_section": sum(chunk.section_title is not None for chunk in chunks),
            "average_sentences_per_chunk": sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0}
