"""Process-scoped application services shared by CLI and HTTP transports."""

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from .bm25 import BM25Index
from .embeddings import Embedder, SentenceTransformerEmbedder
from .generation import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TEMPERATURE,
    OllamaGenerator,
    TextGenerator,
)
from .index import EmbeddingIndex, load_index
from .models import CorpusIndexMetadata, GroundedAnswer
from .pipeline import read_jsonl
from .qdrant_store import (
    QdrantDenseRetriever,
    inspect_qdrant_index,
)
from .rag import answer_question
from .reranking import DEFAULT_RERANKER_MODEL, CrossEncoderReranker
from .retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
)


DEFAULT_API_CHUNKS = Path("data/processed/corpus-52/chunks-legacy.jsonl")
DEFAULT_API_QDRANT_PATH = Path("data/processed/corpus-52/qdrant-legacy")
DEFAULT_API_COLLECTION = "phase75_legacy"


@dataclass(frozen=True)
class RetrievalConfig:
    """Explicit inputs for constructing one retrieval stack."""

    chunks_path: Path
    index_path: Path
    dense_backend: str = "numpy"
    qdrant_path: Path = Path("data/processed/qdrant")
    collection_name: str = "rag_research_chunks"
    retriever_name: str = "dense"
    rerank: bool = False
    candidate_depth: int = 20
    reranker_model: str = DEFAULT_RERANKER_MODEL
    device: Optional[str] = None
    document: Optional[str] = None


@dataclass
class LoadedRetriever:
    """Retriever plus metadata and resources that need an explicit close."""

    index: Optional[EmbeddingIndex]
    embedder: Optional[Embedder]
    retriever: Retriever
    metadata: CorpusIndexMetadata
    dense_retriever: Optional[Retriever] = None

    def close(self) -> None:
        close = getattr(self.dense_retriever, "close", None)
        if callable(close):
            close()


def load_retriever(config: RetrievalConfig) -> LoadedRetriever:
    """Build the same configurable retrieval graph used by every transport."""

    if config.retriever_name not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"unknown retriever: {config.retriever_name}")
    if config.dense_backend not in {"numpy", "qdrant"}:
        raise ValueError(f"unknown dense backend: {config.dense_backend}")
    if config.rerank and config.retriever_name != "hybrid":
        raise ValueError("--rerank requires --retriever hybrid")
    if config.candidate_depth <= 0:
        raise ValueError("candidate_depth must be positive")
    if config.document and (
        config.dense_backend != "qdrant" or config.retriever_name != "dense"
    ):
        raise ValueError(
            "--document currently requires --retriever dense --dense-backend qdrant"
        )

    chunks = read_jsonl(config.chunks_path)
    bm25 = BM25Retriever(BM25Index(chunks))
    index: Optional[EmbeddingIndex] = None
    embedder: Optional[Embedder] = None
    dense: Optional[Retriever] = None

    if config.dense_backend == "numpy":
        index = load_index(config.chunks_path, config.index_path)
        metadata = CorpusIndexMetadata(
            index.model_name, index.dimension, len(index.chunks)
        )
    else:
        qdrant_info = inspect_qdrant_index(
            config.chunks_path,
            config.qdrant_path,
            collection_name=config.collection_name,
        )
        metadata = CorpusIndexMetadata(
            qdrant_info.model_name,
            qdrant_info.dimension,
            qdrant_info.point_count,
        )

    if config.retriever_name != "bm25":
        embedder = SentenceTransformerEmbedder(
            model_name=metadata.embedding_model,
            device=config.device,
            local_files_only=True,
        )
        if config.dense_backend == "numpy":
            assert index is not None
            dense = DenseRetriever(index, embedder)
        else:
            dense = QdrantDenseRetriever(
                config.chunks_path,
                config.qdrant_path,
                embedder,
                collection_name=config.collection_name,
                document=config.document,
            )

    try:
        selected: Retriever
        if config.retriever_name == "dense":
            assert dense is not None
            selected = dense
        elif config.retriever_name == "bm25":
            selected = bm25
        else:
            assert dense is not None
            selected = HybridRetriever(
                dense,
                bm25,
                candidate_depth=config.candidate_depth,
                rrf_k=60,
            )

        if config.rerank:
            reranker = CrossEncoderReranker(
                model_name=config.reranker_model,
                device=config.device,
                local_files_only=True,
            )
            selected = RerankingRetriever(
                selected,
                reranker,
                candidate_depth=config.candidate_depth,
            )
    except Exception:
        close = getattr(dense, "close", None)
        if callable(close):
            close()
        raise

    return LoadedRetriever(index, embedder, selected, metadata, dense)


@dataclass(frozen=True)
class ApplicationSettings:
    """Small environment-backed configuration surface for the API process."""

    chunks_path: Path = DEFAULT_API_CHUNKS
    qdrant_path: Path = DEFAULT_API_QDRANT_PATH
    collection_name: str = DEFAULT_API_COLLECTION
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_timeout: float = 180.0
    device: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "ApplicationSettings":
        """Read deployment-specific values without introducing a config framework."""

        timeout_text = os.getenv("RAG_OLLAMA_TIMEOUT", "180")
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise ValueError("RAG_OLLAMA_TIMEOUT must be a number") from exc
        if timeout <= 0:
            raise ValueError("RAG_OLLAMA_TIMEOUT must be positive")
        device = os.getenv("RAG_DEVICE") or None
        return cls(
            chunks_path=Path(os.getenv("RAG_CHUNKS_PATH", str(DEFAULT_API_CHUNKS))),
            qdrant_path=Path(
                os.getenv("RAG_QDRANT_PATH", str(DEFAULT_API_QDRANT_PATH))
            ),
            collection_name=os.getenv(
                "RAG_QDRANT_COLLECTION", DEFAULT_API_COLLECTION
            ),
            ollama_model=os.getenv("RAG_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            ollama_url=os.getenv("RAG_OLLAMA_URL", DEFAULT_OLLAMA_URL),
            ollama_timeout=timeout,
            device=device,
        )


class RAGApplication:
    """Long-lived local RAG resources with a small request-oriented interface."""

    def __init__(
        self,
        loaded: LoadedRetriever,
        generator: TextGenerator,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.loaded = loaded
        self.generator = generator
        self.temperature = temperature
        self._request_lock = Lock()

    @property
    def retrieval_strategy(self) -> str:
        return self.loaded.retriever.name

    @property
    def retrieval_score_type(self) -> str:
        return self.loaded.retriever.score_name

    @property
    def dense_backend(self) -> Optional[str]:
        return getattr(self.loaded.retriever, "backend_name", None)

    @property
    def metadata(self) -> CorpusIndexMetadata:
        return self.loaded.metadata

    def ask(self, question: str, top_k: int) -> GroundedAnswer:
        """Serialize local model use while reusing all process-scoped resources."""

        with self._request_lock:
            return answer_question(
                question,
                self.loaded.index,
                self.loaded.embedder,
                self.generator,
                top_k=top_k,
                temperature=self.temperature,
                retriever=self.loaded.retriever,
            )

    def close(self) -> None:
        self.loaded.close()


def default_retrieval_config(settings: ApplicationSettings) -> RetrievalConfig:
    """Describe the established Phase 7.5 stack for any application transport."""

    return RetrievalConfig(
        chunks_path=settings.chunks_path,
        index_path=Path("data/processed/embedding_index"),
        dense_backend="qdrant",
        qdrant_path=settings.qdrant_path,
        collection_name=settings.collection_name,
        retriever_name="hybrid",
        rerank=True,
        candidate_depth=20,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=settings.device,
    )


def build_application(settings: ApplicationSettings) -> RAGApplication:
    """Initialize the strongest established stack once for an API process."""

    loaded = load_retriever(default_retrieval_config(settings))
    try:
        generator = OllamaGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_url,
            timeout=settings.ollama_timeout,
        )
        generator.ensure_model_available()
        return RAGApplication(loaded, generator)
    except Exception:
        loaded.close()
        raise
