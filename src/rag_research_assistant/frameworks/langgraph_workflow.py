"""A bounded retrieve/evaluate/rewrite workflow built with LangGraph."""

import re
from typing import List, Literal, Protocol, Sequence, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..models import GroundedAnswer
from .langchain_pipeline import LangChainRAGPipeline


class RetrievalState(TypedDict):
    """Explicit state carried between graph nodes."""

    question: str
    query: str
    documents: List[Document]
    evidence_sufficient: bool
    rewrite_count: int
    max_rewrites: int
    query_history: List[str]
    answer: GroundedAnswer | None


class EvidenceEvaluator(Protocol):
    def is_sufficient(self, question: str, documents: Sequence[Document]) -> bool: ...


class QueryRewriter(Protocol):
    def rewrite(self, question: str, previous_query: str, attempt: int) -> str: ...


class TermCoverageEvaluator:
    """Conservative, deterministic evidence check with no extra model call."""

    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "does",
        "for",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "which",
        "who",
    }

    def __init__(self, minimum_coverage: float = 0.6) -> None:
        if not 0.0 <= minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be between 0 and 1")
        self.minimum_coverage = minimum_coverage

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9]+", text.lower())
            if len(term) > 1 and term not in cls._STOPWORDS
        }

    def is_sufficient(self, question: str, documents: Sequence[Document]) -> bool:
        if not documents:
            return False
        if re.search(r"\b(who wrote|who authored|authors? of)\b", question.lower()):
            return any(
                int(document.metadata.get("start_page", 0)) == 1
                for document in documents
            )
        question_terms = self._terms(question)
        if not question_terms:
            return True
        evidence_terms = self._terms(" ".join(doc.page_content for doc in documents))
        return len(question_terms & evidence_terms) / len(question_terms) >= self.minimum_coverage


class DeterministicQueryRewriter:
    """Make an underspecified query more retrieval-oriented without an LLM."""

    _METHOD_TITLES = {
        "colbert": "ColBERT Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
        "dpr": "Dense Passage Retrieval for Open-Domain Question Answering",
        "hyde": "Precise Zero-Shot Dense Retrieval without Relevance Labels",
        "rag": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "sgpt": "SGPT GPT Sentence Embeddings for Semantic Search",
    }

    def rewrite(self, question: str, previous_query: str, attempt: int) -> str:
        lowered = question.lower()
        if re.search(r"\b(who wrote|who authored|authors? of)\b", lowered):
            title = next(
                (
                    paper_title
                    for method, paper_title in self._METHOD_TITLES.items()
                    if re.search(rf"\b{re.escape(method)}\b", lowered)
                ),
                None,
            )
            if title:
                return f"Who are the authors of {title}?"
            hint = "official paper title author names authorship"
        elif re.search(r"\b(stands for|acronym|abbreviation)\b", lowered):
            hint = "full name definition acronym expansion"
        elif re.search(r"\b(compare|difference|differ)\b", lowered):
            hint = "comparison similarities differences method details"
        else:
            hint = "relevant passage explicit evidence technical details"
        return f"{previous_query.strip()} {hint}"


class BoundedRetrievalGraph:
    """Compile and run a small graph; this is a workflow, not an agent."""

    def __init__(
        self,
        pipeline: LangChainRAGPipeline,
        *,
        evaluator: EvidenceEvaluator | None = None,
        rewriter: QueryRewriter | None = None,
        max_rewrites: int = 1,
    ) -> None:
        if not 0 <= max_rewrites <= 3:
            raise ValueError("max_rewrites must be between 0 and 3")
        self.pipeline = pipeline
        self.evaluator = evaluator or TermCoverageEvaluator()
        self.rewriter = rewriter or DeterministicQueryRewriter()
        self.max_rewrites = max_rewrites
        self.graph = self._compile()

    def _retrieve(self, state: RetrievalState) -> dict:
        return {"documents": self.pipeline.retrieve_documents(state["query"])}

    def _evaluate(self, state: RetrievalState) -> dict:
        return {
            "evidence_sufficient": self.evaluator.is_sufficient(
                state["question"], state["documents"]
            )
        }

    @staticmethod
    def _route_after_evaluation(
        state: RetrievalState,
    ) -> Literal["answer", "rewrite"]:
        if state["evidence_sufficient"]:
            return "answer"
        if state["rewrite_count"] >= state["max_rewrites"]:
            return "answer"
        return "rewrite"

    def _rewrite(self, state: RetrievalState) -> dict:
        next_count = state["rewrite_count"] + 1
        rewritten = self.rewriter.rewrite(
            state["question"], state["query"], next_count
        ).strip()
        if not rewritten:
            raise ValueError("query rewriter returned an empty query")
        return {
            "query": rewritten,
            "rewrite_count": next_count,
            "query_history": [*state["query_history"], rewritten],
        }

    def _answer(self, state: RetrievalState) -> dict:
        return {
            "answer": self.pipeline.answer_from_documents(
                state["question"], state["documents"]
            )
        }

    def _compile(self) -> CompiledStateGraph:
        builder = StateGraph(RetrievalState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("evaluate", self._evaluate)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("answer", self._answer)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "evaluate")
        builder.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {"answer": "answer", "rewrite": "rewrite"},
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("answer", END)
        return builder.compile()

    def invoke(self, question: str) -> RetrievalState:
        if not question.strip():
            raise ValueError("question cannot be empty")
        initial: RetrievalState = {
            "question": question.strip(),
            "query": question.strip(),
            "documents": [],
            "evidence_sufficient": False,
            "rewrite_count": 0,
            "max_rewrites": self.max_rewrites,
            "query_history": [question.strip()],
            "answer": None,
        }
        return self.graph.invoke(initial)
