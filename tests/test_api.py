from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_research_assistant.api import create_app
from rag_research_assistant.generation import GenerationError, OllamaUnavailableError
from rag_research_assistant.models import (
    CitationSource,
    CorpusIndexMetadata,
    GroundedAnswer,
)
from rag_research_assistant.qdrant_store import InvalidQdrantIndexError


class FakeService:
    retrieval_strategy = "hybrid+rerank"
    retrieval_score_type = "cross-encoder"
    dense_backend = "qdrant"
    metadata = CorpusIndexMetadata("test/embedding", 384, 52)
    generator = SimpleNamespace(model_name="test/generator")

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    def ask(self, question: str, top_k: int) -> GroundedAnswer:
        self.calls.append((question, top_k))
        if self.failure is not None:
            raise self.failure
        return GroundedAnswer(
            text="DPR uses passage vectors [1].",
            sources=[
                CitationSource(
                    citation_number=1,
                    score=8.5,
                    document="dpr.pdf",
                    page_number=3,
                    end_page=4,
                    chunk_id="dpr:structural:7",
                )
            ],
            retrieved=[],
        )

    def close(self) -> None:
        self.closed = True


def test_health_reports_liveness_and_ready_dependencies() -> None:
    service = FakeService()

    with TestClient(create_app(service=service)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "live": True,
        "ready": True,
        "dependencies": {"qdrant": "ready", "ollama": "ready"},
    }
    assert service.closed is True


def test_ask_returns_structured_answer_sources_and_metadata() -> None:
    service = FakeService()

    with TestClient(create_app(service=service)) as client:
        response = client.post(
            "/ask", json={"question": "  How does DPR represent passages?  ", "top_k": 3}
        )

    assert response.status_code == 200
    assert service.calls == [("How does DPR represent passages?", 3)]
    assert response.json() == {
        "answer": "DPR uses passage vectors [1].",
        "sources": [
            {
                "citation_number": 1,
                "score": 8.5,
                "document": "dpr.pdf",
                "start_page": 3,
                "end_page": 4,
                "chunk_id": "dpr:structural:7",
            }
        ],
        "metadata": {
            "top_k": 3,
            "retrieval_strategy": "hybrid+rerank",
            "retrieval_score_type": "cross-encoder",
            "dense_backend": "qdrant",
            "generation_model": "test/generator",
            "embedding_model": "test/embedding",
            "indexed_chunks": 52,
        },
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": 123},
        {"question": "valid", "top_k": 0},
        {"question": "valid", "top_k": 21},
        {"question": "valid", "top_k": "5"},
        {"question": "valid", "unknown": True},
    ],
)
def test_ask_rejects_invalid_schema(payload: dict) -> None:
    with TestClient(create_app(service=FakeService())) as client:
        response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_generation_failure_is_translated_without_leaking_details() -> None:
    service = FakeService(GenerationError("private failure at /secret/path"))

    with TestClient(create_app(service=service)) as client:
        response = client.post("/ask", json={"question": "What is DPR?"})

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "generation_failed"
    assert "/secret/path" not in response.text


def test_unavailable_ollama_is_a_service_dependency_error() -> None:
    service = FakeService(OllamaUnavailableError("not running"))

    with TestClient(create_app(service=service)) as client:
        response = client.post("/ask", json={"question": "What is DPR?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "generation_unavailable"


def test_failed_startup_keeps_health_live_but_blocks_ask() -> None:
    def unavailable_service(_settings):
        raise InvalidQdrantIndexError("missing index at /secret/path")

    with TestClient(create_app(service_factory=unavailable_service)) as client:
        health = client.get("/health")
        ask = client.post("/ask", json={"question": "What is DPR?"})

    assert health.status_code == 200
    assert health.json() == {
        "status": "degraded",
        "live": True,
        "ready": False,
        "dependencies": {"qdrant": "unavailable", "ollama": "unknown"},
    }
    assert ask.status_code == 503
    assert ask.json()["detail"]["code"] == "service_not_ready"
    assert "/secret/path" not in ask.text
