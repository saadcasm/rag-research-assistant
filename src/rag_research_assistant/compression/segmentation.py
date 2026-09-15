"""Dependency-light sentence grouping with stable source offsets."""

import re
from typing import List, Sequence, Tuple

from ..models import SearchResult
from .models import TextSegment


_ABBREVIATIONS = {
    "al.", "approx.", "eq.", "eqs.", "et.", "et al.", "fig.", "figs.", "i.e.",
    "e.g.", "no.", "pp.", "sec.", "secs.", "table.", "vs.",
}
_INITIALISM = re.compile(r"(?:\b[A-Za-z]\.){2,}$")


def _trimmed_span(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _protected_period(text: str, index: int) -> bool:
    if 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit():
        return True
    prefix = text[max(0, index - 12) : index + 1].lower()
    if any(prefix.endswith(value) for value in _ABBREVIATIONS):
        return True
    token = text[max(0, text.rfind(" ", 0, index) + 1) : index + 1]
    return bool(_INITIALISM.search(token))


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Find conservative sentence/paragraph boundaries without changing text."""

    if not text.strip():
        return []
    boundaries: List[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\n":
            line_end = index
            while line_end < len(text) and text[line_end] == "\n":
                line_end += 1
            if line_end - index >= 2:
                boundaries.append(line_end)
            index = line_end
            continue
        if char in ".!?" and not (char == "." and _protected_period(text, index)):
            end = index + 1
            while end < len(text) and text[end] in "\"')]}":
                end += 1
            if end == len(text) or text[end].isspace():
                boundaries.append(end)
        index += 1
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    spans: List[Tuple[int, int]] = []
    start = 0
    for end in boundaries:
        trimmed = _trimmed_span(text, start, end)
        if trimmed[0] < trimmed[1]:
            spans.append(trimmed)
        start = end
    return spans


def _group_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    min_characters: int,
    target_characters: int,
) -> List[Tuple[int, int]]:
    """Combine tiny sentences into readable short passages."""

    if not spans:
        return []
    groups: List[Tuple[int, int]] = []
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        current_length = current_end - current_start
        combined_length = end - current_start
        if current_length < min_characters or combined_length <= target_characters:
            current_end = end
        else:
            groups.append((current_start, current_end))
            current_start, current_end = start, end
    if groups and current_end - current_start < min_characters:
        previous_start, _ = groups.pop()
        groups.append((previous_start, current_end))
    else:
        groups.append((current_start, current_end))
    return groups


class Segmenter:
    """Split retrieved chunks into sentence groups with exact offsets."""

    def __init__(self, *, min_characters: int = 80, target_characters: int = 360) -> None:
        if min_characters <= 0:
            raise ValueError("min_characters must be positive")
        if target_characters < min_characters:
            raise ValueError("target_characters must be at least min_characters")
        self.min_characters = min_characters
        self.target_characters = target_characters

    def segment(self, result: SearchResult, source_rank: int) -> List[TextSegment]:
        if source_rank <= 0:
            raise ValueError("source_rank must be positive")
        chunk = result.chunk
        spans = _group_spans(
            chunk.text,
            _sentence_spans(chunk.text),
            min_characters=self.min_characters,
            target_characters=self.target_characters,
        )
        return [
            TextSegment(
                text=chunk.text[start:end],
                source_chunk_id=chunk.chunk_id,
                document=chunk.document,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_title=chunk.section_title,
                source_rank=source_rank,
                segment_index=segment_index,
                start_char=start,
                end_char=end,
            )
            for segment_index, (start, end) in enumerate(spans)
        ]
