"""Experimental Hypothetical Document Embeddings (HyDE) retrieval."""

import re
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..embeddings import Embedder
from ..generation import TextGenerator
from ..hybrid import reciprocal_rank_fusion
from ..models import SearchResult
from ..reranking import Reranker
from ..retrievers import Retriever
from .rewriting import extract_protected_terms


HYDE_PROMPT_TEMPLATE = """You create a retrieval probe for a research-paper search system.

Write a concise technical passage that could plausibly appear in a relevant research paper and contain the information needed to answer the question.

Rules:
1. Write an answer-like research passage, not a search query and not instructions.
2. Preserve named entities, method names, acronyms, model names, and dataset names from the question exactly.
3. Use 2 to 5 sentences and approximately 60 to 150 words.
4. Do not fabricate or include citations, references, source identifiers, or quotations.
5. Do not say that the passage is hypothetical.
6. Do not mention this prompt, retrieval, or your role.
7. Output only the passage, with no title, label, JSON, or meta commentary.

QUESTION
{question}

TECHNICAL PASSAGE
"""


_WORD = re.compile(r"\S+")
_CONTENT_WORD = re.compile(r"[A-Za-z0-9]+(?:[-./:][A-Za-z0-9]+)*")
_ENTITY = re.compile(r"(?<!\w)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}[A-Z0-9-]*)(?!\w)")
_NUMBER = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")
_PARENTHETICAL = re.compile(r"\(([^()]{2,100})\)")
_CITATION_LIKE = re.compile(
    r"\[\d+(?:\s*[-,]\s*\d+)*\]|\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s+\d{4}\)"
)
_STOPWORDS = {
    "a", "an", "and", "are", "by", "does", "for", "from", "how", "in",
    "is", "of", "on", "or", "the", "to", "what", "when", "which", "why",
    "with",
}
_ENTITY_STOPWORDS = {
    "A", "An", "How", "In", "The", "What", "When", "Where", "Which", "Who", "Why",
}


def build_hyde_prompt(question: str) -> str:
    """Build an answer-like probe prompt without asking for a final answer."""

    if not question.strip():
        raise ValueError("question cannot be empty")
    return HYDE_PROMPT_TEMPLATE.format(question=question.strip())


def normalize_hypothetical_document(text: str, *, max_words: int = 150) -> tuple[str, bool]:
    """Normalize whitespace and cap probe length without rewriting its content."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("HyDE generator returned an empty response")
    normalized = " ".join(text.split())
    fenced = re.fullmatch(r"```(?:text|markdown)?\s*(.*?)\s*```", normalized, re.I | re.S)
    if fenced:
        normalized = " ".join(fenced.group(1).split())
    normalized = re.sub(
        r"^(?:technical passage|hypothetical document|answer)\s*:\s*",
        "",
        normalized,
        flags=re.I,
    ).strip()
    if not normalized:
        raise ValueError("HyDE generator returned no usable passage")
    matches = list(_WORD.finditer(normalized))
    if len(matches) <= max_words:
        return normalized, False
    return normalized[: matches[max_words - 1].end()].rstrip(), True


def _entity_terms(text: str) -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            match.group(0)
            for match in _ENTITY.finditer(text)
            if match.group(0) not in _ENTITY_STOPWORDS
        )
    )


def _content_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _CONTENT_WORD.findall(text)
        if len(token) > 2 and token.casefold() not in _STOPWORDS
    }


@dataclass(frozen=True)
class HyDEDiagnostics:
    protected_terms: Tuple[str, ...]
    preserved_protected_terms: Tuple[str, ...]
    missing_protected_terms: Tuple[str, ...]
    introduced_named_entities: Tuple[str, ...]
    introduced_numbers: Tuple[str, ...]
    suspicious_expansions: Tuple[str, ...]
    citation_like_patterns: Tuple[str, ...]
    query_term_coverage: float
    lexical_jaccard: float
    warnings: Tuple[str, ...]


def diagnose_hypothetical_document(question: str, document: str) -> HyDEDiagnostics:
    """Surface drift indicators for audit; never block a generated probe."""

    protected = tuple(dict.fromkeys((*extract_protected_terms(question), *_entity_terms(question))))
    lowered = document.casefold()
    preserved = tuple(term for term in protected if term.casefold() in lowered)
    missing = tuple(term for term in protected if term.casefold() not in lowered)
    question_entities = {term.casefold() for term in _entity_terms(question)}
    introduced_entities = tuple(
        term for term in _entity_terms(document) if term.casefold() not in question_entities
    )
    question_numbers = {value.casefold() for value in _NUMBER.findall(question)}
    document_without_citations = _CITATION_LIKE.sub("", document)
    introduced_numbers = tuple(
        dict.fromkeys(
            value
            for value in _NUMBER.findall(document_without_citations)
            if value.casefold() not in question_numbers
        )
    )
    question_text = question.casefold()
    suspicious_expansions = tuple(
        dict.fromkeys(
            match.group(1).strip()
            for match in _PARENTHETICAL.finditer(document)
            if match.group(1).strip().casefold() not in question_text
        )
    )
    citation_like = tuple(dict.fromkeys(_CITATION_LIKE.findall(document)))
    query_terms = _content_terms(question)
    document_terms = _content_terms(document)
    coverage = len(query_terms & document_terms) / len(query_terms) if query_terms else 0.0
    union = query_terms | document_terms
    jaccard = len(query_terms & document_terms) / len(union) if union else 0.0
    warnings: List[str] = []
    if missing:
        warnings.append("missing_protected_terms:" + ",".join(missing))
    if introduced_numbers:
        warnings.append("introduced_numeric_detail:" + ",".join(introduced_numbers))
    if suspicious_expansions:
        warnings.append("suspicious_expansion:" + ",".join(suspicious_expansions))
    if citation_like:
        warnings.append("citation_like_output:" + ",".join(citation_like))
    if coverage == 0.0 and query_terms:
        warnings.append("no_query_content_term_overlap")
    return HyDEDiagnostics(
        protected,
        preserved,
        missing,
        introduced_entities,
        introduced_numbers,
        suspicious_expansions,
        citation_like,
        coverage,
        jaccard,
        tuple(warnings),
    )


@dataclass(frozen=True)
class HypotheticalDocument:
    question: str
    text: str
    model_name: str
    word_count: int
    character_count: int
    generation_seconds: float
    truncated: bool
    fallback: bool
    error: Optional[str]
    diagnostics: Optional[HyDEDiagnostics]


class HyDEDocumentGenerator:
    """Generate one compact answer-like retrieval probe with safe fallback."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        temperature: float = 0.0,
        max_words: int = 150,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        self.generator = generator
        self.model_name = generator.model_name
        self.temperature = temperature
        self.max_words = max_words

    def generate(self, question: str) -> HypotheticalDocument:
        original = question.strip()
        prompt = build_hyde_prompt(original)
        started = perf_counter()
        text = ""
        truncated = False
        error = None
        diagnostics = None
        try:
            raw = self.generator.generate(prompt, temperature=self.temperature)
            text, truncated = normalize_hypothetical_document(
                raw, max_words=self.max_words
            )
            diagnostics = diagnose_hypothetical_document(original, text)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return HypotheticalDocument(
            question=original,
            text=text,
            model_name=self.model_name,
            word_count=len(text.split()),
            character_count=len(text),
            generation_seconds=perf_counter() - started,
            truncated=truncated,
            fallback=not bool(text),
            error=error,
            diagnostics=diagnostics,
        )


class VectorDenseRetriever(Protocol):
    embedder: Embedder

    def search_vector(self, vector: np.ndarray, *, top_k: int = 5) -> List[SearchResult]: ...


@dataclass(frozen=True)
class HyDERetrievalResult:
    baseline_candidates: Tuple[SearchResult, ...]
    baseline_results: Tuple[SearchResult, ...]
    hyde_dense_results: Tuple[SearchResult, ...]
    hyde_only_results: Tuple[SearchResult, ...]
    fused_candidates: Tuple[SearchResult, ...]
    fused_results: Tuple[SearchResult, ...]
    baseline_seconds: float
    embedding_seconds: float
    dense_seconds: float
    hyde_reranking_seconds: float
    fusion_seconds: float
    fused_reranking_seconds: float
    hyde_only_total_seconds: float
    fused_total_seconds: float
    fallback: bool
    error: Optional[str]


class HyDEExperimentalRetriever:
    """Compare HyDE branches while keeping final relevance tied to the question."""

    def __init__(
        self,
        baseline: Retriever,
        dense: VectorDenseRetriever,
        reranker: Reranker,
        *,
        candidate_depth: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if candidate_depth <= 0:
            raise ValueError("candidate_depth must be positive")
        self.baseline = baseline
        self.dense = dense
        self.reranker = reranker
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k

    def search(
        self,
        original_question: str,
        hypothetical: HypotheticalDocument,
        *,
        top_k: int = 5,
    ) -> HyDERetrievalResult:
        if not original_question.strip():
            raise ValueError("original_question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        baseline_started = perf_counter()
        baseline_candidates = self.baseline.search(
            original_question, top_k=self.candidate_depth
        )
        baseline_seconds = perf_counter() - baseline_started
        baseline_results = tuple(baseline_candidates[:top_k])
        if hypothetical.fallback:
            return self._fallback(
                baseline_candidates, baseline_results, baseline_seconds,
                hypothetical.error or "HyDE generation produced no usable text",
                generation_seconds=hypothetical.generation_seconds,
            )
        embedding_seconds = dense_seconds = hyde_reranking_seconds = 0.0
        fusion_seconds = fused_reranking_seconds = 0.0
        try:
            started = perf_counter()
            vector = np.asarray(
                self.dense.embedder.embed_query(hypothetical.text), dtype=np.float32
            )
            embedding_seconds = perf_counter() - started
            started = perf_counter()
            hyde_dense = self.dense.search_vector(
                vector, top_k=self.candidate_depth
            )
            dense_seconds = perf_counter() - started
            started = perf_counter()
            hyde_only = self.reranker.rerank(
                original_question, hyde_dense, top_k
            )
            hyde_reranking_seconds = perf_counter() - started
            started = perf_counter()
            fused_candidates = reciprocal_rank_fusion(
                [baseline_candidates, hyde_dense],
                top_k=self.candidate_depth,
                rrf_k=self.rrf_k,
            )
            fusion_seconds = perf_counter() - started
            started = perf_counter()
            fused = self.reranker.rerank(
                original_question, fused_candidates, top_k
            )
            fused_reranking_seconds = perf_counter() - started
        except Exception as exc:
            return self._fallback(
                baseline_candidates,
                baseline_results,
                baseline_seconds,
                f"{type(exc).__name__}: {exc}",
                generation_seconds=hypothetical.generation_seconds,
                embedding_seconds=embedding_seconds,
                dense_seconds=dense_seconds,
            )
        return HyDERetrievalResult(
            tuple(baseline_candidates), baseline_results, tuple(hyde_dense),
            tuple(hyde_only), tuple(fused_candidates), tuple(fused),
            baseline_seconds, embedding_seconds, dense_seconds,
            hyde_reranking_seconds, fusion_seconds, fused_reranking_seconds,
            hypothetical.generation_seconds + embedding_seconds + dense_seconds + hyde_reranking_seconds,
            baseline_seconds + hypothetical.generation_seconds + embedding_seconds
            + dense_seconds + fusion_seconds + fused_reranking_seconds,
            False, None,
        )

    @staticmethod
    def _fallback(
        baseline_candidates: Sequence[SearchResult],
        baseline_results: Sequence[SearchResult],
        baseline_seconds: float,
        error: str,
        *,
        generation_seconds: float = 0.0,
        embedding_seconds: float = 0.0,
        dense_seconds: float = 0.0,
    ) -> HyDERetrievalResult:
        fallback = tuple(baseline_results)
        return HyDERetrievalResult(
            tuple(baseline_candidates), fallback, (), fallback,
            tuple(baseline_candidates), fallback,
            baseline_seconds, embedding_seconds, dense_seconds,
            0.0, 0.0, 0.0,
            baseline_seconds + generation_seconds + embedding_seconds + dense_seconds,
            baseline_seconds + generation_seconds + embedding_seconds + dense_seconds,
            True, error,
        )
