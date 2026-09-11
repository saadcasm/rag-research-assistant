"""Local text embedding behind a small, testable interface."""

from typing import Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


class Embedder(Protocol):
    """The behavior needed by indexing and retrieval."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class SentenceTransformerEmbedder:
    """Create local embeddings with one Sentence Transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name, device=device, local_files_only=local_files_only
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed chunk texts into a `(chunk_count, dimensions)` matrix."""

        if not texts:
            raise ValueError("at least one document text is required")
        encoded = self._model.encode_document(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > self.batch_size,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                f"embedding model returned an invalid document matrix: {matrix.shape}"
            )
        return matrix

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query with the same model used for documents."""

        if not text.strip():
            raise ValueError("query cannot be empty")
        encoded = self._model.encode_query(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != 1:
            raise ValueError(f"embedding model returned an invalid query matrix: {matrix.shape}")
        return matrix[0]
