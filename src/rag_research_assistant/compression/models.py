"""Auditable data structures for post-retrieval extractive compression."""

from dataclasses import dataclass
from typing import Optional, Tuple

from ..models import SearchResult


@dataclass(frozen=True)
class TextSegment:
    """An exact, offset-addressable span from one retrieved chunk."""

    text: str
    source_chunk_id: str
    document: str
    start_page: int
    end_page: int
    section_title: Optional[str]
    source_rank: int
    segment_index: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class CompressedSegment(TextSegment):
    """A retained source span paired with an uncalibrated relevance score."""

    relevance_score: float


@dataclass(frozen=True)
class CompressedContext:
    """The selected spans for one source while preserving retrieval rank."""

    source_result: SearchResult
    source_rank: int
    original_text_length: int
    compressed_text: str
    selected_segments: Tuple[CompressedSegment, ...]
    total_segment_count: int
    minimum_safeguard_used: bool = False

    @property
    def compressed_text_length(self) -> int:
        return len(self.compressed_text)

    @property
    def compression_ratio(self) -> float:
        if self.original_text_length == 0:
            return 0.0
        return self.compressed_text_length / self.original_text_length

    @property
    def discarded_segment_count(self) -> int:
        return self.total_segment_count - len(self.selected_segments)


@dataclass(frozen=True)
class CompressionResult:
    """One strategy/budget output over an unchanged retrieval result list."""

    strategy: str
    budget_ratio: float
    contexts: Tuple[CompressedContext, ...]
    original_context_characters: int
    compressed_context_characters: int
    segments_before: int
    segments_retained: int
    segmentation_seconds: float
    scoring_seconds: float
    selection_seconds: float
    empty_output_prevention_count: int
    omitted_source_ranks: Tuple[int, ...]

    @property
    def compression_ratio(self) -> float:
        if self.original_context_characters == 0:
            return 0.0
        return self.compressed_context_characters / self.original_context_characters

    @property
    def characters_removed(self) -> int:
        return self.original_context_characters - self.compressed_context_characters

    @property
    def segments_discarded(self) -> int:
        return self.segments_before - self.segments_retained

    @property
    def compression_seconds(self) -> float:
        return self.segmentation_seconds + self.scoring_seconds + self.selection_seconds
