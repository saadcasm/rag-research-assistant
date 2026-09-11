from typing import Sequence

import numpy as np
import pytest

from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.models import Chunk
from rag_research_assistant.retrieval import cosine_similarities, search


class QueryEmbedder:
    model_name = "test/model"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id, "paper.pdf", 1, 1, 0, 4, "text")


def test_cosine_similarities_measure_direction_not_magnitude() -> None:
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    documents = np.asarray([[10.0, 0.0], [1.0, 1.0], [-2.0, 0.0]], dtype=np.float32)

    scores = cosine_similarities(query, documents)

    np.testing.assert_allclose(scores, [1.0, 2**-0.5, -1.0], rtol=1e-6)


def test_search_ranks_chunks_and_caps_top_k() -> None:
    index = EmbeddingIndex(
        embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        chunks=[_chunk("unrelated"), _chunk("best"), _chunk("second")],
        model_name="test/model",
    )

    results = search("query", index, QueryEmbedder(), top_k=10)

    assert [result.chunk.chunk_id for result in results] == ["best", "second", "unrelated"]


def test_search_rejects_a_different_query_model() -> None:
    index = EmbeddingIndex(
        embeddings=np.ones((1, 2), dtype=np.float32),
        chunks=[_chunk("one")],
        model_name="another/model",
    )

    with pytest.raises(ValueError, match="does not match"):
        search("query", index, QueryEmbedder())


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_non_positive_top_k(top_k: int) -> None:
    index = EmbeddingIndex(
        embeddings=np.ones((1, 2), dtype=np.float32),
        chunks=[_chunk("one")],
        model_name="test/model",
    )

    with pytest.raises(ValueError, match="positive"):
        search("query", index, QueryEmbedder(), top_k=top_k)
