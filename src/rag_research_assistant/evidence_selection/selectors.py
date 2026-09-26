"""Deterministic, gold-free evidence-set selectors for Phase 10J."""

import re
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Mapping, Sequence, Tuple

import numpy as np

from ..models import SearchResult
from ..query_transform.rewriting import extract_protected_terms


_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-./:][A-Za-z0-9]+)*\b")
_GENERIC = {"How", "What", "Which", "Why", "When", "Where", "Compare", "Explain"}


@dataclass(frozen=True)
class SelectionDecision:
    selection_step: int
    chunk_id: str
    original_rank: int
    cross_encoder_score: float
    normalized_relevance: float
    facets_covered: Tuple[str, ...]
    new_facets_added: Tuple[str, ...]
    redundancy_score: float
    diversity_contribution: float
    coverage_contribution: float
    final_selection_score: float


@dataclass(frozen=True)
class SelectionResult:
    strategy: str
    facets: Tuple[str, ...]
    selected: Tuple[SearchResult, ...]
    decisions: Tuple[SelectionDecision, ...]
    latency_seconds: float

    def to_dict(self):
        return {
            "strategy": self.strategy, "facets": list(self.facets),
            "selected_chunk_ids": [item.chunk.chunk_id for item in self.selected],
            "decisions": [asdict(item) for item in self.decisions],
            "latency_seconds": self.latency_seconds,
        }


def extract_query_facets(question: str) -> Tuple[str, ...]:
    """Extract transparent identity-bearing terms without an LLM or labels."""

    if not question.strip():
        raise ValueError("question cannot be empty")
    values = list(extract_protected_terms(question))
    for token in _TOKEN.findall(question):
        if token in _GENERIC:
            continue
        if token.isupper() or any(character.isupper() for character in token[1:]):
            values.append(token)
    deduplicated = []
    seen = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            deduplicated.append(value)
    return tuple(deduplicated)


def rank_normalized_relevance(count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.float32)
    if count == 1:
        return np.ones(1, dtype=np.float32)
    return np.linspace(1.0, 0.0, count, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def selection_diagnostics(selected: Sequence[SearchResult], vectors: Mapping[str, np.ndarray]):
    similarities = [
        cosine_similarity(vectors[left.chunk.chunk_id], vectors[right.chunk.chunk_id])
        for index, left in enumerate(selected)
        for right in selected[index + 1 :]
    ]
    documents = [item.chunk.document for item in selected]
    pages = [(item.chunk.document, item.chunk.start_page, item.chunk.end_page) for item in selected]
    return {
        "average_pairwise_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "maximum_pairwise_similarity": max(similarities, default=0.0),
        "unique_documents": len(set(documents)),
        "repeated_document_count": len(documents) - len(set(documents)),
        "repeated_page_count": len(pages) - len(set(pages)),
    }


def select_evidence(
    question: str,
    candidates: Sequence[SearchResult],
    vectors: Mapping[str, np.ndarray],
    *,
    top_k: int = 10,
    strategy: str,
    diversity_weight: float = 0.0,
    coverage_weight: float = 0.0,
    anchor_top_1: bool = True,
) -> SelectionResult:
    """Greedily select a set using only question, candidates, scores, and vectors."""

    if strategy not in {"baseline", "diversity", "coverage", "coverage_diversity"}:
        raise ValueError("unknown selection strategy")
    if top_k <= 0 or diversity_weight < 0 or coverage_weight < 0:
        raise ValueError("selection parameters must be non-negative and top_k positive")
    if any(item.chunk.chunk_id not in vectors for item in candidates):
        raise ValueError("candidate embedding is missing")
    started = perf_counter()
    facets = extract_query_facets(question)
    relevance = rank_normalized_relevance(len(candidates))
    facet_map = {
        index: tuple(facet for facet in facets if facet.casefold() in item.chunk.text.casefold())
        for index, item in enumerate(candidates)
    }
    selected_indexes = []
    covered = set()
    decisions = []
    while len(selected_indexes) < min(top_k, len(candidates)):
        remaining = [index for index in range(len(candidates)) if index not in selected_indexes]
        if not selected_indexes and anchor_top_1:
            chosen = 0
            redundancy = 0.0
            new_facets = tuple(facet_map[chosen])
            coverage = coverage_weight * (len(new_facets) / len(facets) if facets else 0.0)
            score = float(relevance[chosen]) + coverage
        else:
            scored = []
            for index in remaining:
                redundancy = max(
                    (cosine_similarity(vectors[candidates[index].chunk.chunk_id], vectors[candidates[prior].chunk.chunk_id]) for prior in selected_indexes),
                    default=0.0,
                )
                new_facets = tuple(facet for facet in facet_map[index] if facet.casefold() not in covered)
                coverage = coverage_weight * (len(new_facets) / len(facets) if facets else 0.0)
                score = float(relevance[index]) + coverage - diversity_weight * redundancy
                scored.append((score, -index, index, redundancy, new_facets, coverage))
            score, _, chosen, redundancy, new_facets, coverage = max(scored)
        selected_indexes.append(chosen)
        covered.update(facet.casefold() for facet in facet_map[chosen])
        decisions.append(SelectionDecision(
            len(selected_indexes), candidates[chosen].chunk.chunk_id, chosen + 1,
            candidates[chosen].score, float(relevance[chosen]), facet_map[chosen], new_facets,
            redundancy, -diversity_weight * redundancy, coverage, score,
        ))
    return SelectionResult(
        strategy, facets, tuple(candidates[index] for index in selected_indexes),
        tuple(decisions), perf_counter() - started,
    )
