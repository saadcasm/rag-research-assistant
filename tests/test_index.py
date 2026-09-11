from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from rag_research_assistant.index import InvalidIndexError, build_index, load_index
from rag_research_assistant.models import Chunk
from rag_research_assistant.pipeline import write_jsonl


class FakeEmbedder:
    model_name = "test/model"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        values = [[len(text), index + 1] for index, text in enumerate(texts)]
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([len(text), 1], dtype=np.float32)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, "paper.pdf", 1, 1, 0, len(text), text)


def test_index_round_trip_preserves_row_to_chunk_mapping(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [_chunk("first", "short"), _chunk("second", "a longer chunk")]
    write_jsonl(chunks, chunks_path)

    built = build_index(chunks_path, tmp_path / "index", FakeEmbedder())
    loaded = load_index(chunks_path, tmp_path / "index")

    np.testing.assert_array_equal(loaded.embeddings, built.embeddings)
    assert loaded.chunks == chunks
    assert loaded.model_name == "test/model"
    assert loaded.dimension == 2


def test_index_rejects_changed_chunks_file(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl([_chunk("first", "original")], chunks_path)
    build_index(chunks_path, tmp_path / "index", FakeEmbedder())
    write_jsonl([_chunk("first", "changed")], chunks_path)

    with pytest.raises(InvalidIndexError, match="changed after embedding"):
        load_index(chunks_path, tmp_path / "index")
