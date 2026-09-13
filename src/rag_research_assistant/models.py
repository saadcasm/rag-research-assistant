"""Data passed between the extraction and chunking stages."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PageText:
    """Text extracted from one physical PDF page."""

    document: str
    page_number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrieval-sized text unit with source metadata."""

    chunk_id: str
    document: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    # `page_number` is retained for Phase 1--6 compatibility and is always the
    # first physical page represented by this chunk. New code should use the
    # explicit page span when presenting provenance.
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    section_title: Optional[str] = None
    chunking_strategy: str = "legacy"

    def __post_init__(self) -> None:
        start_page = self.page_number if self.start_page is None else self.start_page
        end_page = start_page if self.end_page is None else self.end_page
        if start_page <= 0 or end_page < start_page:
            raise ValueError("chunk page span must be positive and ordered")
        object.__setattr__(self, "start_page", start_page)
        object.__setattr__(self, "end_page", end_page)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    """One chunk returned by semantic search, paired with its score."""

    score: float
    chunk: Chunk


@dataclass(frozen=True)
class CorpusIndexMetadata:
    """Backend-neutral facts recorded with an evaluation report."""

    embedding_model: str
    embedding_dimension: int
    chunk_count: int


@dataclass(frozen=True)
class CitationSource:
    """Display metadata connecting a prompt citation to a retrieved chunk."""

    citation_number: int
    score: float
    document: str
    page_number: int
    chunk_id: str
    end_page: Optional[int] = None


@dataclass(frozen=True)
class GroundedAnswer:
    """A generated answer and the evidence available to the generator."""

    text: str
    sources: List[CitationSource]
    retrieved: List[SearchResult]
