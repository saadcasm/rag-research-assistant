import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_research_assistant.frameworks.langchain_pipeline import LangChainRAGPipeline
from rag_research_assistant.frameworks.langgraph_workflow import (
    BoundedRetrievalGraph,
    DeterministicQueryRewriter,
)
from rag_research_assistant.models import Chunk, SearchResult


def result(query: str, *, page: int = 2) -> SearchResult:
    return SearchResult(
        score=5.0,
        chunk=Chunk(
            chunk_id=f"dpr:p{page}:c0",
            document="dpr.pdf",
            page_number=page,
            chunk_index=0,
            char_start=0,
            char_end=32,
            text=f"Evidence retrieved for {query}",
        ),
    )


class RecordingRetriever:
    name = "fake"
    score_name = "fake-score"

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append(query)
        if self.failure:
            raise self.failure
        page = 1 if "title page" in query else 2
        return [result(query, page=page)]


class SequenceEvaluator:
    def __init__(self, values: list[bool]) -> None:
        self.values = iter(values)
        self.calls = 0

    def is_sufficient(self, question, documents) -> bool:
        self.calls += 1
        return next(self.values)


class RecordingRewriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def rewrite(self, question: str, previous_query: str, attempt: int) -> str:
        self.calls.append((question, previous_query, attempt))
        return f"{previous_query} title page"


def pipeline(retriever: RecordingRetriever) -> LangChainRAGPipeline:
    model = RunnableLambda(lambda prompt: AIMessage(content="Grounded answer [1]."))
    return LangChainRAGPipeline(retriever, model)


def test_graph_sufficient_branch_goes_directly_to_answer() -> None:
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter()
    workflow = BoundedRetrievalGraph(
        pipeline(retriever),
        evaluator=SequenceEvaluator([True]),
        rewriter=rewriter,
    )

    state = workflow.invoke("How does DPR work?")

    assert retriever.queries == ["How does DPR work?"]
    assert rewriter.calls == []
    assert state["rewrite_count"] == 0
    assert state["query_history"] == ["How does DPR work?"]
    assert state["answer"].text == "Grounded answer [1]."


def test_graph_rewrite_branch_updates_state_and_retrieves_again() -> None:
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter()
    evaluator = SequenceEvaluator([False, True])
    workflow = BoundedRetrievalGraph(
        pipeline(retriever), evaluator=evaluator, rewriter=rewriter
    )

    state = workflow.invoke("Who wrote the DPR paper?")

    assert retriever.queries == [
        "Who wrote the DPR paper?",
        "Who wrote the DPR paper? title page",
    ]
    assert evaluator.calls == 2
    assert state["rewrite_count"] == 1
    assert state["query_history"] == retriever.queries
    assert state["documents"][0].metadata["start_page"] == 1


def test_graph_enforces_maximum_rewrites_before_answering() -> None:
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter()
    workflow = BoundedRetrievalGraph(
        pipeline(retriever),
        evaluator=SequenceEvaluator([False, False, False]),
        rewriter=rewriter,
        max_rewrites=2,
    )

    state = workflow.invoke("Find difficult evidence")

    assert len(retriever.queries) == 3
    assert len(rewriter.calls) == 2
    assert state["rewrite_count"] == 2
    assert len(state["query_history"]) == 3
    assert state["answer"] is not None


def test_graph_propagates_node_failure() -> None:
    workflow = BoundedRetrievalGraph(
        pipeline(RecordingRetriever(RuntimeError("retrieval failed")))
    )

    with pytest.raises(RuntimeError, match="retrieval failed"):
        workflow.invoke("Will this fail?")


def test_default_rewriter_expands_known_method_for_authorship() -> None:
    rewritten = DeterministicQueryRewriter().rewrite(
        "Who wrote the DPR paper?", "Who wrote the DPR paper?", 1
    )

    assert rewritten == (
        "Who are the authors of Dense Passage Retrieval for Open-Domain "
        "Question Answering?"
    )
