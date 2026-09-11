from rag_research_assistant.context import build_context
from rag_research_assistant.models import Chunk, SearchResult


def test_build_context_labels_chunks_and_returns_source_mapping() -> None:
    first = SearchResult(
        score=0.91,
        chunk=Chunk("paper-a:p2:c1", "paper-a.pdf", 2, 1, 0, 12, "First evidence"),
    )
    second = SearchResult(
        score=0.72,
        chunk=Chunk("paper-b:p8:c3", "paper-b.pdf", 8, 3, 10, 25, "Second evidence"),
    )

    context, sources = build_context([first, second])

    assert "SOURCE [1]" in context
    assert "filename: paper-a.pdf" in context
    assert "page: 2" in context
    assert "chunk_id: paper-a:p2:c1" in context
    assert "First evidence" in context
    assert "SOURCE [2]" in context
    assert sources[0].citation_number == 1
    assert sources[0].score == 0.91
    assert sources[1].chunk_id == "paper-b:p8:c3"


def test_build_context_handles_no_results() -> None:
    assert build_context([]) == ("", [])
