from pathlib import Path

from rag_research_assistant.models import Chunk
from rag_research_assistant.pipeline import read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    chunk = Chunk(
        chunk_id="paper.pdf:p1:c1",
        document="paper.pdf",
        page_number=1,
        chunk_index=1,
        char_start=0,
        char_end=12,
        text="Useful text.",
    )
    output_path = tmp_path / "nested" / "chunks.jsonl"

    assert write_jsonl([chunk], output_path) == 1
    assert read_jsonl(output_path) == [chunk]

