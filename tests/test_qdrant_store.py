from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.models import Chunk
from rag_research_assistant.pipeline import write_jsonl
from rag_research_assistant.qdrant_store import (
    DEFAULT_COLLECTION_NAME,
    InvalidQdrantIndexError,
    QdrantDenseRetriever,
    build_qdrant_index,
    chunk_from_payload,
    inspect_qdrant_index,
    point_id_for_chunk,
)
from rag_research_assistant.reranking import CrossEncoderReranker
from rag_research_assistant.retrievers import HybridRetriever, RerankingRetriever


class FakeEmbedder:
    model_name = "test/model"

    def __init__(self, query=(1.0, 0.0)) -> None:
        self.query = query

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.query, dtype=np.float32)


def _chunk(chunk_id: str, document: str, page: int, text: str) -> Chunk:
    return Chunk(chunk_id, document, page, 1, 0, len(text), text)


def _corpus(tmp_path: Path):
    chunks = [
        _chunk("first", "one.pdf", 1, "first evidence"),
        _chunk("second", "two.pdf", 2, "second evidence"),
        _chunk("third", "one.pdf", 3, "third evidence"),
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(chunks, chunks_path)
    index = EmbeddingIndex(
        np.asarray([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]], dtype=np.float32),
        chunks,
        "test/model",
    )
    return chunks_path, index


def test_build_persists_collection_dimension_payload_and_metadata(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"

    built = build_qdrant_index(chunks_path, storage, index)
    inspected = inspect_qdrant_index(chunks_path, storage)

    assert built.point_count == inspected.point_count == 3
    assert inspected.dimension == 2
    assert inspected.distance == "Cosine"
    assert inspected.model_name == "test/model"
    assert inspected.storage_path == storage
    client = QdrantClient(path=str(storage))
    try:
        points, _ = client.scroll(DEFAULT_COLLECTION_NAME, limit=10, with_payload=True)
        assert points[0].payload is not None
        assert set(points[0].payload) == {
            "chunk_id",
            "document",
            "page_number",
            "chunk_index",
            "char_start",
            "char_end",
            "text",
            "start_page",
            "end_page",
            "section_title",
            "chunking_strategy",
        }
        assert chunk_from_payload(points[0].payload) in index.chunks
    finally:
        client.close()


def test_rebuild_is_duplicate_safe_and_recreate_is_explicit(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"

    build_qdrant_index(chunks_path, storage, index)
    rebuilt = build_qdrant_index(chunks_path, storage, index)

    assert rebuilt.point_count == 3
    assert build_qdrant_index(
        chunks_path, storage, index, recreate=True
    ).point_count == 3
    assert point_id_for_chunk("first") == point_id_for_chunk("first")
    assert point_id_for_chunk("first") != point_id_for_chunk("second")


def test_incompatible_collection_requires_explicit_recreation(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    build_qdrant_index(chunks_path, storage, index)
    changed_model = EmbeddingIndex(index.embeddings, index.chunks, "other/model")

    with pytest.raises(InvalidQdrantIndexError, match="--recreate"):
        build_qdrant_index(chunks_path, storage, changed_model)

    rebuilt = build_qdrant_index(
        chunks_path, storage, changed_model, recreate=True
    )
    assert rebuilt.model_name == "other/model"


def test_query_returns_cosine_order_and_reconstructs_chunk_metadata(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    build_qdrant_index(chunks_path, storage, index)
    retriever = QdrantDenseRetriever(chunks_path, storage, FakeEmbedder())
    try:
        results = retriever.search("question", top_k=2)
    finally:
        retriever.close()

    assert [result.chunk.chunk_id for result in results] == ["first", "second"]
    assert results[0].score > results[1].score
    assert results[1].chunk.document == "two.pdf"
    assert results[1].chunk.page_number == 2


def test_document_payload_filter_restricts_dense_results(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    build_qdrant_index(chunks_path, storage, index)
    retriever = QdrantDenseRetriever(
        chunks_path, storage, FakeEmbedder(), document="one.pdf"
    )
    try:
        results = retriever.search("question", top_k=5)
    finally:
        retriever.close()

    assert [result.chunk.document for result in results] == ["one.pdf", "one.pdf"]


def test_query_dimension_must_match_collection(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    build_qdrant_index(chunks_path, storage, index)
    retriever = QdrantDenseRetriever(
        chunks_path, storage, FakeEmbedder(query=(1.0, 0.0, 0.0))
    )
    try:
        with pytest.raises(InvalidQdrantIndexError, match="query vector dimension"):
            retriever.search("question")
    finally:
        retriever.close()


def test_stale_chunks_and_model_mismatch_fail_clearly(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    build_qdrant_index(chunks_path, storage, index)

    with pytest.raises(InvalidQdrantIndexError, match="query model does not match"):
        QdrantDenseRetriever(
            chunks_path,
            storage,
            type("OtherEmbedder", (), {"model_name": "other/model"})(),
        )

    write_jsonl([*index.chunks, _chunk("new", "new.pdf", 1, "new")], chunks_path)
    with pytest.raises(InvalidQdrantIndexError, match="changed after Qdrant"):
        inspect_qdrant_index(chunks_path, storage)


def test_invalid_collection_metadata_and_payload_are_rejected(tmp_path: Path) -> None:
    chunks_path, _ = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    client = QdrantClient(path=str(storage))
    client.create_collection(
        DEFAULT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.close()

    with pytest.raises(InvalidQdrantIndexError, match="metadata"):
        inspect_qdrant_index(chunks_path, storage)
    with pytest.raises(InvalidQdrantIndexError, match="missing fields"):
        chunk_from_payload({"chunk_id": "only-one-field"})


class FakeLexicalRetriever:
    name = "bm25"
    score_name = "bm25"
    backend_name = None

    def __init__(self, results) -> None:
        self.results = results

    def search(self, query: str, top_k: int = 5):
        return self.results[:top_k]


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return np.asarray(list(range(len(pairs))), dtype=np.float32)


def test_qdrant_candidates_flow_through_hybrid_and_reranking(tmp_path: Path) -> None:
    chunks_path, index = _corpus(tmp_path)
    storage = tmp_path / "qdrant"
    build_qdrant_index(chunks_path, storage, index)
    dense = QdrantDenseRetriever(chunks_path, storage, FakeEmbedder())
    try:
        dense_results = dense.search("question", top_k=3)
        lexical = FakeLexicalRetriever(list(reversed(dense_results)))
        hybrid = HybridRetriever(dense, lexical, candidate_depth=3)
        reranked = RerankingRetriever(
            hybrid,
            CrossEncoderReranker("fake/model", scorer=FakeCrossEncoder()),
            candidate_depth=3,
        )
        results = reranked.search("question", top_k=2)
    finally:
        dense.close()

    assert len(results) == 2
    assert reranked.backend_name == "qdrant"
    assert all(result.chunk in index.chunks for result in results)
