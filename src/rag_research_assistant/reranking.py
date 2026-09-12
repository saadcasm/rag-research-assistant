"""Optional pairwise cross-encoder reranking over a small candidate set."""

from typing import List, Protocol, Sequence

import numpy as np

from .models import SearchResult


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class PairScorer(Protocol):
    """Minimal behavior supplied by Sentence Transformers CrossEncoder."""

    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs): ...


class Reranker(Protocol):
    """Behavior required by the retrieval pipeline's reranking stage."""

    model_name: str

    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int
    ) -> List[SearchResult]: ...


class CrossEncoderReranker:
    """Score each (query, chunk) pair jointly with a local cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        batch_size: int = 16,
        device: str | None = None,
        local_files_only: bool = False,
        scorer: PairScorer | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        if scorer is None:
            from sentence_transformers import CrossEncoder

            scorer = CrossEncoder(
                model_name,
                device=device,
                local_files_only=local_files_only,
                max_length=512,
            )
        self._scorer = scorer

    def rerank(
        self, query: str, candidates: Sequence[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Replace candidate scores with pairwise relevance scores and rerank."""

        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        raw_scores = self._scorer.predict(
            [(query, result.chunk.text) for result in candidates],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError("cross-encoder returned invalid candidate scores")
        ranked_rows = sorted(
            range(len(candidates)), key=lambda row: (-float(scores[row]), row)
        )
        return [
            SearchResult(float(scores[row]), candidates[row].chunk)
            for row in ranked_rows[: min(top_k, len(ranked_rows))]
        ]
