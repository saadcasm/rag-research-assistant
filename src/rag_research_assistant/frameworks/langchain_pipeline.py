"""A small LangChain composition around the project's existing retriever."""

from dataclasses import asdict
from typing import Any, List, Mapping, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict

from ..context import build_context
from ..generation import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, DEFAULT_TEMPERATURE
from ..models import Chunk, GroundedAnswer, SearchResult
from ..prompting import GROUNDED_PROMPT_TEMPLATE
from ..rag import INSUFFICIENT_CONTEXT_ANSWER
from ..retrievers import Retriever


class ChainPayload(BaseModel):
    """Typed shape produced by the LCEL chain before domain conversion."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    documents: List[Document]
    context: str
    answer: str


def build_ollama_chat_model(
    model_name: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatOllama:
    """Construct LangChain's wrapper with the manual path's established defaults."""

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        reasoning=False,
        validate_model_on_init=True,
    )


def search_result_to_document(result: SearchResult) -> Document:
    """Adapt a transparent project result to LangChain's document contract."""

    chunk = result.chunk
    metadata = asdict(chunk)
    metadata.pop("text")
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.text,
        metadata={
            **metadata,
            "score": result.score,
        },
    )


def document_to_search_result(document: Document) -> SearchResult:
    """Recover the domain record, including provenance, from a Document."""

    metadata = document.metadata
    required = {
        "chunk_id",
        "document",
        "page_number",
        "chunk_index",
        "char_start",
        "char_end",
        "score",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(
            "LangChain document is missing provenance fields: " + ", ".join(missing)
        )
    chunk = Chunk(
        chunk_id=str(metadata["chunk_id"]),
        document=str(metadata["document"]),
        page_number=int(metadata["page_number"]),
        chunk_index=int(metadata["chunk_index"]),
        char_start=int(metadata["char_start"]),
        char_end=int(metadata["char_end"]),
        text=document.page_content,
        start_page=int(metadata.get("start_page") or metadata["page_number"]),
        end_page=int(metadata.get("end_page") or metadata["page_number"]),
        section_title=(
            None
            if metadata.get("section_title") is None
            else str(metadata["section_title"])
        ),
        chunking_strategy=str(metadata.get("chunking_strategy", "legacy")),
    )
    return SearchResult(score=float(metadata["score"]), chunk=chunk)


def documents_to_results(documents: Sequence[Document]) -> List[SearchResult]:
    return [document_to_search_result(document) for document in documents]


def documents_to_context(documents: Sequence[Document]) -> str:
    """Reuse the manual formatter after crossing the Document boundary."""

    context, _ = build_context(documents_to_results(documents))
    return context


class LangChainRAGPipeline:
    """LCEL prompt/model composition without replacing manual retrieval."""

    def __init__(self, retriever: Retriever, model: Runnable[Any, Any], top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.prompt = PromptTemplate.from_template(GROUNDED_PROMPT_TEMPLATE)

        self.retriever_runnable: Runnable[str, List[Document]] = RunnableLambda(
            self.retrieve_documents
        )
        prepared = RunnableParallel(
            question=RunnableLambda(lambda question: question),
            documents=self.retriever_runnable,
        ) | RunnableLambda(self._add_context)
        self.answer_chain = self.prompt | self.model | StrOutputParser()
        self.chain = prepared.assign(answer=RunnableLambda(self._generate_answer))

    def retrieve_documents(self, question: str) -> List[Document]:
        """Expose the existing Retriever through LangChain's Runnable interface."""

        if not question.strip():
            raise ValueError("question cannot be empty")
        return [
            search_result_to_document(result)
            for result in self.retriever.search(question.strip(), top_k=self.top_k)
        ]

    @staticmethod
    def _add_context(value: Mapping[str, Any]) -> dict[str, Any]:
        documents = value["documents"]
        return {
            **value,
            "context": documents_to_context(documents),
        }

    def _generate_answer(self, value: Mapping[str, Any]) -> str:
        if not value["documents"]:
            return INSUFFICIENT_CONTEXT_ANSWER
        return self.answer_chain.invoke(value)

    def answer_from_documents(
        self, question: str, documents: Sequence[Document]
    ) -> GroundedAnswer:
        """Generate once from already-retrieved documents."""

        results = documents_to_results(documents)
        if not results:
            return GroundedAnswer(INSUFFICIENT_CONTEXT_ANSWER, [], [])
        context, sources = build_context(results)
        prompt_value = self.prompt.invoke(
            {"question": question.strip(), "context": context}
        )
        answer = StrOutputParser().invoke(self.model.invoke(prompt_value))
        return GroundedAnswer(answer.strip(), sources, results)

    def invoke(self, question: str) -> GroundedAnswer:
        """Run retrieve -> context -> prompt -> model and return domain types."""

        payload = ChainPayload.model_validate(self.chain.invoke(question.strip()))
        results = documents_to_results(payload.documents)
        _, sources = build_context(results)
        return GroundedAnswer(payload.answer.strip(), sources, results)
