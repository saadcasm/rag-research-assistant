"""Data passed between the extraction and chunking stages."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


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
