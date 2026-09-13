"""Phase 7 strategy tests use deterministic local text and fake embeddings."""

import numpy as np

from rag_research_assistant.chunking import chunk_pages, chunk_statistics, detect_heading
from rag_research_assistant.context import build_context
from rag_research_assistant.evaluation import ExpectedSource, expected_source_matches
from rag_research_assistant.models import PageText, SearchResult


def _pages() -> list[PageText]:
    return [
        PageText("paper.pdf", 1, "Abstract\n\nCats purr softly. Cats sleep often.\n\n1 Introduction\n\nRetrieval finds evidence."),
        PageText("paper.pdf", 2, "Retrieval combines lexical and semantic signals. Results improve."),
    ]


def test_legacy_is_page_scoped_and_keeps_phase_one_identity() -> None:
    chunks = chunk_pages(_pages(), chunk_size=45, overlap=5, strategy="legacy")
    assert all(chunk.start_page == chunk.end_page for chunk in chunks)
    assert chunks[0].chunk_id == "paper.pdf:p1:c1"


def test_boundary_prefers_whole_sentences_and_does_not_cross_pages() -> None:
    chunks = chunk_pages(_pages(), chunk_size=55, overlap=10, strategy="boundary")
    assert all(chunk.start_page == chunk.end_page for chunk in chunks)
    assert all(chunk.text.endswith((".", "often", "evidence")) for chunk in chunks)
    assert all(chunk.chunking_strategy == "boundary" for chunk in chunks)


def test_structural_crosses_page_and_records_heading() -> None:
    chunks = chunk_pages(_pages(), chunk_size=120, overlap=0, strategy="structural")
    assert any(chunk.start_page == 1 and chunk.end_page == 2 for chunk in chunks)
    assert any(chunk.section_title == "1 Introduction" for chunk in chunks)


def test_heading_detection_is_conservative() -> None:
    assert detect_heading("3.1 Overview") == "3.1 Overview"
    assert detect_heading("This ordinary prose sentence is not a heading.") is None


def test_semantic_breaks_are_deterministic_with_fake_embeddings() -> None:
    def fake_embed(texts: list[str]) -> np.ndarray:
        return np.asarray([[1, 0] if "Cats" in text else [0, 1] for text in texts], dtype=np.float32)

    first = chunk_pages(_pages(), chunk_size=120, overlap=0, strategy="semantic", sentence_embedder=fake_embed)
    second = chunk_pages(_pages(), chunk_size=120, overlap=0, strategy="semantic", sentence_embedder=fake_embed)
    assert first == second
    assert all(chunk.chunking_strategy == "semantic" for chunk in first)


def test_page_span_is_honest_for_evaluation_and_citation() -> None:
    chunk = chunk_pages(_pages(), chunk_size=120, overlap=0, strategy="structural")[0]
    result = SearchResult(0.9, chunk)
    assert expected_source_matches(ExpectedSource("paper.pdf", 2), result)
    context, sources = build_context([result])
    assert "pages 1–2" in context
    assert sources[0].end_page == 2


def test_chunk_statistics_reports_sentence_average() -> None:
    chunks = chunk_pages(_pages(), chunk_size=120, overlap=0, strategy="boundary")
    assert chunk_statistics(chunks)["average_sentences_per_chunk"] > 0
