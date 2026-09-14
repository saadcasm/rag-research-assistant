from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_research_assistant.frameworks import langchain_pipeline
from rag_research_assistant.frameworks.langchain_pipeline import (
    LangChainRAGPipeline,
    build_ollama_chat_model,
    document_to_search_result,
    search_result_to_document,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.rag import INSUFFICIENT_CONTEXT_ANSWER


def result(
    *, document: str = "dpr.pdf", start_page: int = 2, end_page: int = 3
) -> SearchResult:
    return SearchResult(
        score=7.25,
        chunk=Chunk(
            chunk_id="dpr:p2:c1",
            document=document,
            page_number=start_page,
            chunk_index=1,
            char_start=10,
            char_end=64,
            text="DPR represents passages with a dense passage encoder.",
            start_page=start_page,
            end_page=end_page,
            section_title="Method",
            chunking_strategy="legacy",
        ),
    )


class FakeRetriever:
    name = "fake"
    score_name = "fake-score"

    def __init__(self, results=None, failure: Exception | None = None) -> None:
        self.results = [result()] if results is None else results
        self.failure = failure
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        if self.failure:
            raise self.failure
        return self.results[:top_k]


def recording_model(prompts: list[str]):
    def invoke(prompt):
        prompts.append(prompt.to_string())
        return AIMessage(content="Passages use a dense encoder [1].")

    return RunnableLambda(invoke)


def test_langchain_lcel_pipeline_composes_retrieval_prompt_and_model() -> None:
    prompts: list[str] = []
    retriever = FakeRetriever()
    pipeline = LangChainRAGPipeline(retriever, recording_model(prompts), top_k=3)

    answer = pipeline.invoke("How does DPR represent passages?")

    assert retriever.calls == [("How does DPR represent passages?", 3)]
    assert answer.text == "Passages use a dense encoder [1]."
    assert "DPR represents passages with a dense passage encoder." in prompts[0]
    assert "How does DPR represent passages?" in prompts[0]


def test_langchain_document_adapter_round_trips_provenance() -> None:
    original = result()

    document = search_result_to_document(original)
    restored = document_to_search_result(document)

    assert document.id == "dpr:p2:c1"
    assert document.metadata["score"] == 7.25
    assert restored == original


def test_langchain_result_preserves_citation_page_span() -> None:
    pipeline = LangChainRAGPipeline(FakeRetriever(), recording_model([]))

    answer = pipeline.invoke("How does DPR represent passages?")

    assert len(answer.sources) == 1
    assert answer.sources[0].document == "dpr.pdf"
    assert answer.sources[0].page_number == 2
    assert answer.sources[0].end_page == 3
    assert answer.sources[0].chunk_id == "dpr:p2:c1"


def test_langchain_empty_retrieval_skips_model() -> None:
    prompts: list[str] = []
    pipeline = LangChainRAGPipeline(
        FakeRetriever(results=[]), recording_model(prompts)
    )

    answer = pipeline.invoke("No supporting evidence?")

    assert answer.text == INSUFFICIENT_CONTEXT_ANSWER
    assert answer.sources == []
    assert prompts == []


def test_ollama_wrapper_uses_explicit_local_defaults(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_chat_ollama(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(langchain_pipeline, "ChatOllama", fake_chat_ollama)

    model = build_ollama_chat_model("qwen:test", "http://localhost:9999", 0.2)

    assert model is sentinel
    assert captured == {
        "model": "qwen:test",
        "base_url": "http://localhost:9999",
        "temperature": 0.2,
        "reasoning": False,
        "validate_model_on_init": True,
    }
