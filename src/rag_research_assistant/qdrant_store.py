"""Persistent local Qdrant storage and dense retrieval for Phase 6."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from .embeddings import Embedder
from .index import EmbeddingIndex, file_sha256
from .models import Chunk, SearchResult
from .pipeline import read_jsonl


DEFAULT_QDRANT_PATH = Path("data/processed/qdrant")
DEFAULT_COLLECTION_NAME = "rag_research_chunks"
QDRANT_SCHEMA_VERSION = 2
_PAYLOAD_FIELDS = (
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
)


class InvalidQdrantIndexError(RuntimeError):
    """Raised when a local collection is absent, stale, or incompatible."""


@dataclass(frozen=True)
class QdrantIndexInfo:
    collection_name: str
    storage_path: Path
    point_count: int
    dimension: int
    distance: str
    model_name: str
    chunks_sha256: str
    schema_version: int
    search_mode: str = "exact"


def point_id_for_chunk(chunk_id: str) -> str:
    """Map an arbitrary stable chunk ID to a Qdrant-compatible UUID string."""

    return str(uuid5(NAMESPACE_URL, f"rag-research-assistant:{chunk_id}"))


def chunk_payload(chunk: Chunk) -> Dict[str, Any]:
    """Store exactly the fields needed to reconstruct the Chunk contract."""

    return {field: getattr(chunk, field) for field in _PAYLOAD_FIELDS}


def chunk_from_payload(payload: Optional[Mapping[str, Any]]) -> Chunk:
    """Validate Qdrant payload before reconstructing application metadata."""

    if payload is None:
        raise InvalidQdrantIndexError("Qdrant result is missing its chunk payload")
    missing = [field for field in _PAYLOAD_FIELDS if field not in payload]
    if missing:
        raise InvalidQdrantIndexError(
            f"Qdrant chunk payload is missing fields: {', '.join(missing)}"
        )
    for field in ("chunk_id", "document", "text"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise InvalidQdrantIndexError(
                f"Qdrant chunk payload has invalid {field!r}"
            )
    for field in ("page_number", "chunk_index", "char_start", "char_end", "start_page", "end_page"):
        if isinstance(payload[field], bool) or not isinstance(payload[field], int):
            raise InvalidQdrantIndexError(
                f"Qdrant chunk payload has invalid {field!r}"
            )
    if payload["page_number"] <= 0 or payload["chunk_index"] <= 0 or payload["start_page"] <= 0 or payload["end_page"] < payload["start_page"]:
        raise InvalidQdrantIndexError("Qdrant page and chunk indexes must be positive")
    if payload["char_start"] < 0 or payload["char_end"] < payload["char_start"]:
        raise InvalidQdrantIndexError("Qdrant chunk character offsets are invalid")
    if payload["section_title"] is not None and not isinstance(payload["section_title"], str):
        raise InvalidQdrantIndexError("Qdrant chunk payload has invalid section_title")
    if not isinstance(payload["chunking_strategy"], str) or not payload["chunking_strategy"]:
        raise InvalidQdrantIndexError("Qdrant chunk payload has invalid chunking_strategy")
    return Chunk(**{field: payload[field] for field in _PAYLOAD_FIELDS})


def _collection_metadata(index: EmbeddingIndex, chunks_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": QDRANT_SCHEMA_VERSION,
        "embedding_model": index.model_name,
        "embedding_dimension": index.dimension,
        "chunk_count": len(index.chunks),
        "chunks_sha256": file_sha256(chunks_path),
    }


def _validate_collection(
    client: QdrantClient,
    collection_name: str,
    chunks_path: Path,
) -> QdrantIndexInfo:
    if not client.collection_exists(collection_name):
        raise InvalidQdrantIndexError(
            f"Qdrant collection {collection_name!r} does not exist; run qdrant-build"
        )
    info = client.get_collection(collection_name)
    metadata = info.config.metadata
    if not isinstance(metadata, dict):
        raise InvalidQdrantIndexError(
            "Qdrant collection metadata is missing or corrupt; rebuild with --recreate"
        )
    required = {
        "schema_version": int,
        "embedding_model": str,
        "embedding_dimension": int,
        "chunk_count": int,
        "chunks_sha256": str,
    }
    for key, expected_type in required.items():
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, expected_type):
            raise InvalidQdrantIndexError(
                f"Qdrant collection metadata has invalid {key!r}; rebuild with --recreate"
            )
    if metadata["schema_version"] != QDRANT_SCHEMA_VERSION:
        raise InvalidQdrantIndexError(
            "unsupported Qdrant collection schema; rebuild with --recreate"
        )
    if not metadata["embedding_model"] or metadata["embedding_dimension"] <= 0:
        raise InvalidQdrantIndexError(
            "Qdrant embedding metadata is invalid; rebuild with --recreate"
        )
    if metadata["chunks_sha256"] != file_sha256(chunks_path):
        raise InvalidQdrantIndexError(
            "chunks.jsonl changed after Qdrant indexing; rebuild with --recreate"
        )
    if metadata["chunk_count"] != len(read_jsonl(chunks_path)):
        raise InvalidQdrantIndexError(
            "Qdrant chunk count does not match chunks.jsonl; rebuild with --recreate"
        )
    vectors = info.config.params.vectors
    if not isinstance(vectors, models.VectorParams):
        raise InvalidQdrantIndexError("Qdrant collection must use one unnamed dense vector")
    if vectors.size != metadata["embedding_dimension"]:
        raise InvalidQdrantIndexError(
            "Qdrant vector dimension does not match collection metadata"
        )
    if vectors.distance != models.Distance.COSINE:
        raise InvalidQdrantIndexError("Qdrant collection must use cosine distance")
    point_count = client.count(collection_name, exact=True).count
    if point_count != metadata["chunk_count"]:
        raise InvalidQdrantIndexError(
            "Qdrant point count does not match collection metadata; rebuild with --recreate"
        )
    return QdrantIndexInfo(
        collection_name=collection_name,
        storage_path=Path("."),
        point_count=point_count,
        dimension=vectors.size,
        distance=vectors.distance.value,
        model_name=metadata["embedding_model"],
        chunks_sha256=metadata["chunks_sha256"],
        schema_version=metadata["schema_version"],
    )


def build_qdrant_index(
    chunks_path: Path,
    storage_path: Path,
    index: EmbeddingIndex,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
    batch_size: int = 64,
) -> QdrantIndexInfo:
    """Persist an existing row-aligned embedding index as Qdrant points."""

    if not collection_name.strip():
        raise ValueError("collection_name cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if index.embeddings.ndim != 2 or index.embeddings.shape[0] != len(index.chunks):
        raise ValueError("embedding matrix and chunk mapping are inconsistent")
    source_chunks = read_jsonl(chunks_path)
    if source_chunks != index.chunks:
        raise ValueError("embedding index chunks do not match chunks.jsonl")
    if not np.isfinite(index.embeddings).all():
        raise ValueError("embedding matrix contains non-finite values")

    storage_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(storage_path))
    expected_metadata = _collection_metadata(index, chunks_path)
    try:
        exists = client.collection_exists(collection_name)
        if exists and recreate:
            client.delete_collection(collection_name)
            exists = False
        if exists:
            current = client.get_collection(collection_name)
            if current.config.metadata != expected_metadata:
                raise InvalidQdrantIndexError(
                    "existing Qdrant collection is incompatible; rebuild with --recreate"
                )
            vectors = current.config.params.vectors
            if not isinstance(vectors, models.VectorParams) or (
                vectors.size != index.dimension
                or vectors.distance != models.Distance.COSINE
            ):
                raise InvalidQdrantIndexError(
                    "existing Qdrant vector configuration is incompatible; "
                    "rebuild with --recreate"
                )
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=index.dimension, distance=models.Distance.COSINE
                ),
                metadata=expected_metadata,
            )

        points = [
            models.PointStruct(
                id=point_id_for_chunk(chunk.chunk_id),
                vector=index.embeddings[row].tolist(),
                payload=chunk_payload(chunk),
            )
            for row, chunk in enumerate(index.chunks)
        ]
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )
        info = _validate_collection(client, collection_name, chunks_path)
        return replace(info, storage_path=storage_path)
    finally:
        client.close()


def inspect_qdrant_index(
    chunks_path: Path,
    storage_path: Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> QdrantIndexInfo:
    """Open, validate, and describe a persisted collection."""

    if not storage_path.exists():
        raise InvalidQdrantIndexError(
            f"Qdrant storage not found at {storage_path}; run qdrant-build"
        )
    client = QdrantClient(path=str(storage_path))
    try:
        info = _validate_collection(client, collection_name, chunks_path)
        return replace(info, storage_path=storage_path)
    finally:
        client.close()


class QdrantDenseRetriever:
    """Return SearchResult records from exact local Qdrant cosine search."""

    name = "dense"
    score_name = "cosine"
    backend_name = "qdrant"

    def __init__(
        self,
        chunks_path: Path,
        storage_path: Path,
        embedder: Embedder,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        document: Optional[str] = None,
    ) -> None:
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.embedder = embedder
        self.document = document
        if not storage_path.exists():
            raise InvalidQdrantIndexError(
                f"Qdrant storage not found at {storage_path}; run qdrant-build"
            )
        self._client = QdrantClient(path=str(storage_path))
        try:
            self.info = _validate_collection(
                self._client, collection_name, chunks_path
            )
            if embedder.model_name != self.info.model_name:
                raise InvalidQdrantIndexError(
                    "query model does not match the Qdrant collection: "
                    f"{embedder.model_name!r} != {self.info.model_name!r}"
                )
        except Exception:
            self._client.close()
            raise

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        if vector.ndim != 1 or len(vector) != self.info.dimension:
            raise InvalidQdrantIndexError(
                "query vector dimension does not match the Qdrant collection"
            )
        if not np.isfinite(vector).all():
            raise ValueError("query embedding contains non-finite values")
        query_filter = None
        if self.document:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document", match=models.MatchValue(value=self.document)
                    )
                ]
            )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            SearchResult(float(point.score), chunk_from_payload(point.payload))
            for point in response.points
        ]

    def close(self) -> None:
        """Release the process-scoped local Qdrant client."""

        self._client.close()
