"""Persist and validate the mapping from embedding rows to chunks."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from .embeddings import Embedder
from .models import Chunk
from .pipeline import read_jsonl


EMBEDDINGS_FILENAME = "embeddings.npy"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


class InvalidIndexError(RuntimeError):
    """Raised when an embedding index is missing, stale, or inconsistent."""


@dataclass(frozen=True)
class EmbeddingIndex:
    """An in-memory embedding matrix and its row-aligned chunks."""

    embeddings: NDArray[np.float32]
    chunks: List[Chunk]
    model_name: str

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_matrix(matrix: NDArray[np.float32], chunk_count: int) -> None:
    if matrix.ndim != 2:
        raise InvalidIndexError(f"embeddings must be a 2D matrix, got shape {matrix.shape}")
    if matrix.shape[0] != chunk_count:
        raise InvalidIndexError(
            f"embedding rows ({matrix.shape[0]}) do not match chunks ({chunk_count})"
        )
    if matrix.shape[1] == 0:
        raise InvalidIndexError("embeddings must have at least one dimension")
    if not np.isfinite(matrix).all():
        raise InvalidIndexError("embeddings contain non-finite values")


def build_index(chunks_path: Path, index_dir: Path, embedder: Embedder) -> EmbeddingIndex:
    """Embed Phase 1 chunks and save the matrix plus an integrity manifest."""

    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise ValueError(f"no chunks found in {chunks_path}")

    matrix = np.asarray(
        embedder.embed_documents([chunk.text for chunk in chunks]), dtype=np.float32
    )
    _validate_matrix(matrix, len(chunks))

    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = index_dir / EMBEDDINGS_FILENAME
    with embeddings_path.open("wb") as output:
        np.save(output, matrix, allow_pickle=False)

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_name": embedder.model_name,
        "chunk_count": len(chunks),
        "embedding_dimension": int(matrix.shape[1]),
        "chunks_sha256": _file_sha256(chunks_path),
        "row_to_chunk_id": [chunk.chunk_id for chunk in chunks],
    }
    with (index_dir / MANIFEST_FILENAME).open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")

    return EmbeddingIndex(embeddings=matrix, chunks=chunks, model_name=embedder.model_name)


def load_index(chunks_path: Path, index_dir: Path) -> EmbeddingIndex:
    """Load an index only if it still matches the exact Phase 1 chunk file."""

    embeddings_path = index_dir / EMBEDDINGS_FILENAME
    manifest_path = index_dir / MANIFEST_FILENAME
    if not embeddings_path.exists() or not manifest_path.exists():
        raise InvalidIndexError(f"embedding index not found in {index_dir}; run the embed command")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidIndexError(f"could not read index manifest: {exc}") from exc

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InvalidIndexError("unsupported embedding index schema; rebuild the index")
    if manifest.get("chunks_sha256") != _file_sha256(chunks_path):
        raise InvalidIndexError("chunks.jsonl changed after embedding; rebuild the index")

    chunks = read_jsonl(chunks_path)
    if manifest.get("chunk_count") != len(chunks):
        raise InvalidIndexError("manifest chunk count does not match chunks.jsonl")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if manifest.get("row_to_chunk_id") != chunk_ids:
        raise InvalidIndexError("embedding rows no longer match chunk IDs; rebuild the index")

    try:
        matrix = np.asarray(np.load(embeddings_path, allow_pickle=False), dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise InvalidIndexError(f"could not read embedding matrix: {exc}") from exc
    _validate_matrix(matrix, len(chunks))

    if manifest.get("embedding_dimension") != matrix.shape[1]:
        raise InvalidIndexError("manifest dimension does not match the embedding matrix")
    model_name = manifest.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise InvalidIndexError("manifest does not contain a valid model name")

    return EmbeddingIndex(embeddings=matrix, chunks=chunks, model_name=model_name)
