"""Manual cosine-similarity ranking for the local embedding index."""

from typing import List

import numpy as np
from numpy.typing import NDArray

from .embeddings import Embedder
from .index import EmbeddingIndex
from .models import SearchResult


def cosine_similarities(
    query: NDArray[np.float32], documents: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Compute cosine similarity from one query to every document row."""

    query = np.asarray(query, dtype=np.float32)
    documents = np.asarray(documents, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError(f"query embedding must be 1D, got shape {query.shape}")
    if documents.ndim != 2:
        raise ValueError(f"document embeddings must be 2D, got shape {documents.shape}")
    if documents.shape[1] != query.shape[0]:
        raise ValueError("query and document embedding dimensions must match")
    if not np.isfinite(query).all() or not np.isfinite(documents).all():
        raise ValueError("embeddings must contain only finite values")

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)
    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("cosine similarity is undefined for zero-length vectors")

    return np.asarray(
        (documents @ query) / (document_norms * query_norm), dtype=np.float32
    )


def search(
    query: str,
    index: EmbeddingIndex,
    embedder: Embedder,
    top_k: int = 5,
) -> List[SearchResult]:
    """Embed a query, rank all chunks, and return the best `top_k`."""

    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if embedder.model_name != index.model_name:
        raise ValueError(
            "query model does not match the document index: "
            f"{embedder.model_name!r} != {index.model_name!r}"
        )

    scores = cosine_similarities(embedder.embed_query(query), index.embeddings)
    result_count = min(top_k, len(index.chunks))
    ranked_rows = np.argsort(-scores, kind="stable")[:result_count]
    return [
        SearchResult(score=float(scores[row]), chunk=index.chunks[int(row)])
        for row in ranked_rows
    ]
