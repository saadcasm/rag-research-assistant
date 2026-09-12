"""Composable retrieval strategies shared by search, ask, and evaluation."""

from typing import List, Protocol

from .bm25 import BM25Index
from .embeddings import Embedder
from .hybrid import reciprocal_rank_fusion
from .index import EmbeddingIndex
from .models import SearchResult
from .reranking import Reranker
from .retrieval import search as dense_search


class Retriever(Protocol):
    """Common ranking behavior independent of its scoring mechanism."""

    name: str
    score_name: str

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]: ...


class DenseRetriever:
    name = "dense"
    score_name = "cosine"

    def __init__(self, index: EmbeddingIndex, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        return dense_search(query, self.index, self.embedder, top_k=top_k)


class BM25Retriever:
    name = "bm25"
    score_name = "bm25"

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        return self.index.search(query, top_k=top_k)


class HybridRetriever:
    name = "hybrid"
    score_name = "rrf"

    def __init__(
        self,
        dense: Retriever,
        lexical: Retriever,
        *,
        candidate_depth: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        self.dense = dense
        self.lexical = lexical
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        depth = max(top_k, self.candidate_depth)
        dense_results = self.dense.search(query, top_k=depth)
        lexical_results = self.lexical.search(query, top_k=depth)
        return reciprocal_rank_fusion(
            [dense_results, lexical_results], top_k=top_k, rrf_k=self.rrf_k
        )


class RerankingRetriever:
    name = "hybrid+rerank"
    score_name = "cross-encoder"

    def __init__(
        self, base: Retriever, reranker: Reranker, *, candidate_depth: int = 20
    ) -> None:
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        self.base = base
        self.reranker = reranker
        self.reranker_model = reranker.model_name
        self.candidate_depth = candidate_depth

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        candidates = self.base.search(query, top_k=max(top_k, self.candidate_depth))
        return self.reranker.rerank(query, candidates, top_k)
