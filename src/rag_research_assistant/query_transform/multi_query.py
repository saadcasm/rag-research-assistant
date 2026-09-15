"""Experimental corpus-grounded multi-query generation and rank fusion."""

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import List, Protocol, Sequence, Tuple

from ..generation import TextGenerator
from ..hybrid import reciprocal_rank_fusion
from ..models import SearchResult
from ..retrievers import RerankingRetriever
from .rewriting import (
    IdentityDiagnostics,
    diagnose_identity_preservation,
    extract_protected_terms,
    format_preliminary_evidence,
)


MULTI_QUERY_PROMPT_TEMPLATE = """You generate alternative search queries for a research-paper retrieval system.

Generate exactly {requested_count} meaningfully different retrieval queries. The original query is retained separately, so do not repeat it.

Rules:
1. Preserve the user's intent and every important named entity, paper, method, model, dataset, acronym, and quoted term.
2. Do not answer the question.
3. Add only facts supported by the preliminary evidence.
4. Never replace a specific entity with a generic concept.
5. Prefer retrieval-oriented formulations, not superficial punctuation or word-order changes.
6. If the evidence does not justify an expansion, keep the original terminology.
7. Return only valid JSON in this exact shape: {{"queries": ["query 1", "query 2"]}}

ORIGINAL QUERY
{original_query}

PRELIMINARY EVIDENCE
{evidence}

JSON
"""


@dataclass(frozen=True)
class RejectedQuery:
    query: str
    reason: str


@dataclass(frozen=True)
class VariantDiagnostics:
    query: str
    identity: IdentityDiagnostics
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class MultiQueryGenerationResult:
    """Auditable outcome of one generation call, including safe fallback state."""

    original_query: str
    requested_count: int
    generated_queries: Tuple[str, ...]
    valid_queries: Tuple[str, ...]
    rejected_queries: Tuple[RejectedQuery, ...]
    model_name: str
    latency_seconds: float
    fallback: bool
    error: str | None
    diagnostics: Tuple[VariantDiagnostics, ...]

    @property
    def queries(self) -> Tuple[str, ...]:
        """The actual retrieval queries; the original is invariantly first."""

        return (self.original_query, *self.valid_queries)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_count - len(self.valid_queries))


class MultiQueryGenerator(Protocol):
    model_name: str

    def generate_queries(
        self,
        original_query: str,
        preliminary_results: Sequence[SearchResult],
        *,
        count: int,
    ) -> MultiQueryGenerationResult: ...


def _normalize_query(value: str) -> str:
    return " ".join(value.split()).casefold()


def parse_query_variants(raw_response: str) -> List[str]:
    """Parse the deliberately small JSON contract returned by the generator."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("multi-query generator returned an empty response")
    value = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.I)
    if fenced:
        value = fenced.group(1)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"multi-query generator returned invalid JSON: {exc.msg}") from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("queries")
    if not isinstance(parsed, list):
        raise ValueError("multi-query JSON must contain a queries array")
    if any(not isinstance(item, str) for item in parsed):
        raise ValueError("every generated query must be a string")
    return parsed


def build_multi_query_prompt(
    original_query: str,
    preliminary_results: Sequence[SearchResult],
    *,
    requested_count: int,
    max_chars_per_chunk: int,
) -> str:
    if not original_query.strip():
        raise ValueError("original_query cannot be empty")
    if requested_count <= 0:
        raise ValueError("requested_count must be positive")
    evidence = format_preliminary_evidence(
        preliminary_results, max_chars_per_chunk=max_chars_per_chunk
    )
    return MULTI_QUERY_PROMPT_TEMPLATE.format(
        original_query=original_query.strip(),
        requested_count=requested_count,
        evidence=evidence,
    )


_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9./:-]*")
_PARENTHETICAL = re.compile(r"\(([^()]{2,100})\)")
_STOPWORDS = {
    "a", "an", "and", "are", "by", "does", "for", "from", "how", "in",
    "is", "of", "on", "or", "paper", "the", "to", "what", "when", "which",
    "who", "why", "with",
}


def diagnose_variant(
    original_query: str,
    variant: str,
    preliminary_results: Sequence[SearchResult],
) -> VariantDiagnostics:
    """Surface explainable drift signals without accepting/rejecting by a score."""

    identity = diagnose_identity_preservation(original_query, variant)
    warnings: List[str] = []
    if identity.missing_terms:
        warnings.append("missing_protected_terms:" + ",".join(identity.missing_terms))

    evidence_text = " ".join(
        [original_query, *(result.chunk.text for result in preliminary_results)]
    ).casefold()
    unsupported = [
        term
        for term in extract_protected_terms(variant)
        if term.casefold() not in evidence_text
    ]
    if unsupported:
        warnings.append("unsupported_identity_detail:" + ",".join(unsupported))
    unsupported_expansions = [
        match.group(1).strip()
        for match in _PARENTHETICAL.finditer(variant)
        if match.group(1).strip().casefold() not in evidence_text
    ]
    if unsupported_expansions:
        warnings.append("suspicious_expansion:" + ",".join(unsupported_expansions))

    def content_words(text: str) -> set[str]:
        return {
            token.casefold()
            for token in _WORD.findall(text)
            if len(token) > 2 and token.casefold() not in _STOPWORDS
        }

    if content_words(original_query) and not (
        content_words(original_query) & content_words(variant)
    ):
        warnings.append("no_lexical_identity_overlap")
    return VariantDiagnostics(variant, identity, tuple(warnings))


class CorpusGroundedMultiQueryGenerator:
    """Generate alternatives once and retain every outcome needed for diagnosis."""

    def __init__(
        self,
        generator: TextGenerator,
        *,
        temperature: float = 0.0,
        max_chars_per_chunk: int = 1_000,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if max_chars_per_chunk <= 0:
            raise ValueError("max_chars_per_chunk must be positive")
        self.generator = generator
        self.model_name = generator.model_name
        self.temperature = temperature
        self.max_chars_per_chunk = max_chars_per_chunk

    def generate_queries(
        self,
        original_query: str,
        preliminary_results: Sequence[SearchResult],
        *,
        count: int,
    ) -> MultiQueryGenerationResult:
        original = original_query.strip()
        prompt = build_multi_query_prompt(
            original,
            preliminary_results,
            requested_count=count,
            max_chars_per_chunk=self.max_chars_per_chunk,
        )
        started = perf_counter()
        raw_queries: List[str] = []
        valid: List[str] = []
        rejected: List[RejectedQuery] = []
        error = None
        try:
            raw_queries = parse_query_variants(
                self.generator.generate(prompt, temperature=self.temperature)
            )
            seen = {_normalize_query(original)}
            for raw_query in raw_queries:
                query = " ".join(raw_query.split())
                normalized = _normalize_query(query)
                if not query:
                    rejected.append(RejectedQuery(raw_query, "empty"))
                elif len(query) > 4_000:
                    rejected.append(RejectedQuery(query, "too_long"))
                elif normalized in seen:
                    reason = (
                        "duplicates_original"
                        if normalized == _normalize_query(original)
                        else "duplicate_variant"
                    )
                    rejected.append(RejectedQuery(query, reason))
                elif len(valid) >= count:
                    rejected.append(RejectedQuery(query, "exceeds_requested_count"))
                else:
                    seen.add(normalized)
                    valid.append(query)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if not valid and error is None:
            error = "no valid query variants generated"
        diagnostics = tuple(
            diagnose_variant(original, query, preliminary_results) for query in valid
        )
        return MultiQueryGenerationResult(
            original_query=original,
            requested_count=count,
            generated_queries=tuple(raw_queries),
            valid_queries=tuple(valid),
            rejected_queries=tuple(rejected),
            model_name=self.model_name,
            latency_seconds=perf_counter() - started,
            fallback=not valid,
            error=error,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class MultiQueryRetrievalResult:
    queries: Tuple[str, ...]
    query_rankings: Tuple[Tuple[SearchResult, ...], ...]
    per_query_seconds: Tuple[float, ...]
    fused_candidates: Tuple[SearchResult, ...]
    final_results: Tuple[SearchResult, ...]
    deduplicated_occurrences: int
    candidate_count_before_rerank: int
    retrieval_seconds: float
    fusion_seconds: float
    reranking_seconds: float


class MultiQueryRetriever:
    """Fuse per-query hybrid rankings, then apply the existing reranker once."""

    def __init__(self, retriever: RerankingRetriever, *, rrf_k: int = 60) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        self.retriever = retriever
        self.rrf_k = rrf_k

    def search(
        self,
        original_query: str,
        alternative_queries: Sequence[str],
        *,
        top_k: int = 5,
    ) -> MultiQueryRetrievalResult:
        if not original_query.strip():
            raise ValueError("original_query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        queries = (original_query.strip(), *alternative_queries)
        rankings: List[Tuple[SearchResult, ...]] = []
        query_times: List[float] = []
        for query in queries:
            started = perf_counter()
            ranking = self.retriever.base.search(
                query, top_k=max(top_k, self.retriever.candidate_depth)
            )
            query_times.append(perf_counter() - started)
            rankings.append(tuple(ranking))

        fusion_started = perf_counter()
        fused = reciprocal_rank_fusion(
            rankings,
            top_k=self.retriever.candidate_depth,
            rrf_k=self.rrf_k,
        )
        fusion_seconds = perf_counter() - fusion_started
        rerank_started = perf_counter()
        final = self.retriever.reranker.rerank(original_query, fused, top_k)
        reranking_seconds = perf_counter() - rerank_started
        unique_ids = {
            result.chunk.chunk_id for ranking in rankings for result in ranking
        }
        occurrences = sum(len(ranking) for ranking in rankings)
        return MultiQueryRetrievalResult(
            queries=queries,
            query_rankings=tuple(rankings),
            per_query_seconds=tuple(query_times),
            fused_candidates=tuple(fused),
            final_results=tuple(final),
            deduplicated_occurrences=occurrences - len(unique_ids),
            candidate_count_before_rerank=len(fused),
            retrieval_seconds=sum(query_times),
            fusion_seconds=fusion_seconds,
            reranking_seconds=reranking_seconds,
        )
