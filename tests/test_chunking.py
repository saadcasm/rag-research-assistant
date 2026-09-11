import pytest

from rag_research_assistant.chunking import chunk_page, normalize_text, split_text
from rag_research_assistant.models import PageText


def test_normalize_text_preserves_paragraphs() -> None:
    assert normalize_text(" First   line\r\n\r\n\r\nSecond\tline ") == "First line\n\nSecond line"


def test_split_text_creates_overlapping_bounded_chunks() -> None:
    text = " ".join(f"word-{number}" for number in range(100))
    chunks = split_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(0 < len(chunk_text) <= 120 for _, _, chunk_text in chunks)
    assert all(chunks[index][0] < chunks[index - 1][1] for index in range(1, len(chunks)))


def test_chunk_page_preserves_source_metadata() -> None:
    page = PageText(document="paper.pdf", page_number=7, text="A sentence. " * 30)

    chunks = chunk_page(page, chunk_size=100, overlap=15)

    assert chunks
    assert all(chunk.document == "paper.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert chunks[0].chunk_id == "paper.pdf:p7:c1"


def test_split_text_does_not_skip_a_long_unbroken_string() -> None:
    text = "x" * 250

    chunks = split_text(text, chunk_size=100, overlap=20)

    assert chunks[-1][1] == len(text)
    assert all(text[start:end] == chunk_text for start, end, chunk_text in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_split_text_rejects_invalid_configuration(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("text", chunk_size=chunk_size, overlap=overlap)
