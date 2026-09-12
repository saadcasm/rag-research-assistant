"""Data passed between the extraction and chunking stages."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


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


@dataclass(frozen=True)
class GroundedAnswer:
    """A generated answer and the evidence available to the generator."""

    text: str
    sources: List[CitationSource]
    retrieved: List[SearchResult]
