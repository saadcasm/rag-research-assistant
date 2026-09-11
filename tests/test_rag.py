from typing import Sequence

import numpy as np
import pytest

from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.models import Chunk
from rag_research_assistant.rag import INSUFFICIENT_CONTEXT_ANSWER, answer_question


class FakeEmbedder:
    model_name = "test/embedding-model"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class RecordingGenerator:
    model_name = "test/generator"

    def __init__(self, events=None) -> None:
        self.calls = []
        self.events = events

    def generate(self, prompt: str, temperature: float) -> str:
        if self.events is not None:
            self.events.append("generated")
        self.calls.append((prompt, temperature))
        return "The evidence supports this answer [1]."


def _chunk(chunk_id: str, document: str, page: int, text: str) -> Chunk:
    return Chunk(chunk_id, document, page, 1, 0, len(text), text)


def test_answer_question_orchestrates_retrieval_prompt_and_generation() -> None:
    index = EmbeddingIndex(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        chunks=[
            _chunk("best", "best.pdf", 3, "Strong evidence"),
            _chunk("other", "other.pdf", 7, "Other evidence"),
        ],
        model_name="test/embedding-model",
    )
    events = []
    generator = RecordingGenerator(events)

    answer = answer_question(
        "What does the evidence say?",
        index,
        FakeEmbedder(),
        generator,
        top_k=1,
        temperature=0.1,
        on_retrieved=lambda results: events.append("retrieved"),
    )

    assert events == ["retrieved", "generated"]
    assert answer.text == "The evidence supports this answer [1]."
    assert answer.sources[0].document == "best.pdf"
    assert answer.sources[0].page_number == 3
    assert answer.sources[0].chunk_id == "best"
    prompt, temperature = generator.calls[0]
    assert "Strong evidence" in prompt
    assert "What does the evidence say?" in prompt
    assert temperature == 0.1


def test_no_retrieved_context_returns_insufficient_answer_without_generation() -> None:
    index = EmbeddingIndex(
        embeddings=np.empty((0, 2), dtype=np.float32),
        chunks=[],
        model_name="test/embedding-model",
    )
    generator = RecordingGenerator()

    answer = answer_question("Question", index, FakeEmbedder(), generator)

    assert answer.text == INSUFFICIENT_CONTEXT_ANSWER
    assert answer.sources == []
    assert generator.calls == []


def test_answer_question_rejects_invalid_temperature_before_generation() -> None:
    index = EmbeddingIndex(
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        chunks=[_chunk("one", "paper.pdf", 1, "Evidence")],
        model_name="test/embedding-model",
    )
    generator = RecordingGenerator()

    with pytest.raises(ValueError, match="temperature"):
        answer_question("Question", index, FakeEmbedder(), generator, temperature=2.1)

    assert generator.calls == []
