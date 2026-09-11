"""Turn ranked chunks into numbered, source-labelled evidence."""

from typing import List, Sequence, Tuple

from .models import CitationSource, SearchResult


def build_context(
    results: Sequence[SearchResult],
) -> Tuple[str, List[CitationSource]]:
    """Format retrieved chunks and return their citation mapping."""

    entries: List[str] = []
    sources: List[CitationSource] = []
    for citation_number, result in enumerate(results, start=1):
        chunk = result.chunk
        entries.append(
            "\n".join(
                [
                    f"SOURCE [{citation_number}]",
                    f"filename: {chunk.document}",
                    f"page: {chunk.page_number}",
                    f"chunk_id: {chunk.chunk_id}",
                    "text:",
                    chunk.text,
                ]
            )
        )
        sources.append(
            CitationSource(
                citation_number=citation_number,
                score=result.score,
                document=chunk.document,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
            )
        )

    return "\n\n---\n\n".join(entries), sources
