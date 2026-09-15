"""Experimental multi-branch retrieval and auditable hop-coverage metrics."""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence, Tuple

from .hybrid import reciprocal_rank_fusion
from .models import Chunk, SearchResult
from .query_transform.decomposition import DecompositionResult
from .retrievers import RerankingRetriever


@dataclass(frozen=True)
class HopEvidence:
    document: str
    pages: Tuple[int, ...]
    passage: str = ""


@dataclass(frozen=True)
class ExpectedHop:
    description: str
    expected_key_facts: Tuple[str, ...]
    evidence: Tuple[HopEvidence, ...]
    evidence_available: bool = True


@dataclass(frozen=True)
class MultiHopExample:
    id: str
    question: str
    answerability: str
    category: str
    hops: Tuple[ExpectedHop, ...]
    verification_status: str
    notes: str = ""


@dataclass(frozen=True)
class HopCoverage:
    retrieved_hop_indexes: Tuple[int, ...]
    total_hops: int
    recovered_hops: int
    full: bool
    partial: float


@dataclass(frozen=True)
class EvidenceValidationIssue:
    question_id: str
    hop_index: int
    document: str
    pages: Tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class BranchRanking:
    query: str
    results: Tuple[SearchResult, ...]
    latency_seconds: float


@dataclass(frozen=True)
class FusedCondition:
    branch_rankings: Tuple[BranchRanking, ...]
    candidates: Tuple[SearchResult, ...]
    results: Tuple[SearchResult, ...]
    contributions: Mapping[str, Tuple[str, ...]]
    deduplicated_occurrences: int
    fusion_seconds: float
    reranking_seconds: float


@dataclass(frozen=True)
class MultiHopRetrievalResult:
    decomposition: DecompositionResult
    baseline: FusedCondition
    always_decompose: FusedCondition
    original_plus_decomposed: FusedCondition
    fallback: bool
    error: str | None


def _string(value: Any, field: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line}: {field} must be a non-empty string")
    return value.strip()


def load_multihop_dataset(path: Path) -> list[MultiHopExample]:
    examples: list[MultiHopExample] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            identifier = _string(value.get("id"), "id", line_number)
            if identifier in seen:
                raise ValueError(f"line {line_number}: duplicate id {identifier!r}")
            seen.add(identifier)
            answerability = _string(value.get("answerability"), "answerability", line_number)
            if answerability not in {"answerable", "partially_answerable"}:
                raise ValueError(f"line {line_number}: unsupported answerability")
            raw_hops = value.get("hops")
            if not isinstance(raw_hops, list) or len(raw_hops) < 2:
                raise ValueError(f"line {line_number}: multi-hop questions require at least two hops")
            hops: list[ExpectedHop] = []
            for raw_hop in raw_hops:
                evidence_available = raw_hop.get("evidence_available", True)
                if not isinstance(evidence_available, bool):
                    raise ValueError(f"line {line_number}: evidence_available must be boolean")
                raw_evidence = raw_hop.get("evidence", [])
                if not isinstance(raw_evidence, list):
                    raise ValueError(f"line {line_number}: evidence must be a list")
                evidence = []
                for item in raw_evidence:
                    pages = item.get("pages")
                    if not isinstance(pages, list) or not pages or any(
                        isinstance(page, bool) or not isinstance(page, int) or page <= 0
                        for page in pages
                    ):
                        raise ValueError(f"line {line_number}: evidence pages must be positive integers")
                    evidence.append(HopEvidence(
                        _string(item.get("document"), "document", line_number),
                        tuple(pages),
                        _string(item.get("passage", ""), "passage", line_number),
                    ))
                if evidence_available and not evidence:
                    raise ValueError(f"line {line_number}: available hop needs evidence")
                if not evidence_available and evidence:
                    raise ValueError(f"line {line_number}: unavailable hop cannot contain evidence")
                facts = raw_hop.get("expected_key_facts", [])
                if not isinstance(facts, list) or any(not isinstance(fact, str) or not fact.strip() for fact in facts):
                    raise ValueError(f"line {line_number}: expected_key_facts must be strings")
                hops.append(ExpectedHop(
                    _string(raw_hop.get("description"), "description", line_number),
                    tuple(fact.strip() for fact in facts), tuple(evidence), evidence_available,
                ))
            if answerability == "answerable" and not all(hop.evidence_available for hop in hops):
                raise ValueError(f"line {line_number}: answerable question has unavailable hop")
            if answerability == "partially_answerable" and all(hop.evidence_available for hop in hops):
                raise ValueError(f"line {line_number}: partially answerable question needs a missing hop")
            examples.append(MultiHopExample(
                identifier, _string(value.get("question"), "question", line_number),
                answerability, _string(value.get("category"), "category", line_number),
                tuple(hops), _string(value.get("verification_status"), "verification_status", line_number),
                str(value.get("notes", "")).strip(),
            ))
    if not examples:
        raise ValueError(f"multi-hop dataset is empty: {path}")
    return examples


def hop_matches_result(hop: ExpectedHop, result: SearchResult) -> bool:
    return any(
        evidence.document == result.chunk.document
        and any(result.chunk.start_page <= page <= result.chunk.end_page for page in evidence.pages)
        and _normalized_extracted_text(evidence.passage)
        in _normalized_extracted_text(result.chunk.text)
        for evidence in hop.evidence
    )


def _normalized_extracted_text(text: str) -> str:
    """Normalize deterministic PDF artifacts without semantically rewriting text."""

    text = re.sub(r"-\s+", "", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return " ".join(
        re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", text).casefold())
    )


def verify_evidence_passages(
    examples: Sequence[MultiHopExample], chunks: Sequence[Chunk]
) -> list[EvidenceValidationIssue]:
    """Re-locate every stored passage in current extracted chunk text.

    Matching is exact after whitespace, line-hyphenation, common ligature, and
    punctuation normalization.  There is deliberately no fuzzy threshold that
    could silently bless a merely similar passage.
    """

    issues: list[EvidenceValidationIssue] = []
    for example in examples:
        for hop_index, hop in enumerate(example.hops, 1):
            for evidence in hop.evidence:
                relevant = [
                    chunk.text
                    for chunk in chunks
                    if chunk.document == evidence.document
                    and any(chunk.start_page <= page <= chunk.end_page for page in evidence.pages)
                ]
                reason = None
                if not relevant:
                    reason = "document_or_page_not_found"
                elif _normalized_extracted_text(evidence.passage) not in _normalized_extracted_text(" ".join(relevant)):
                    reason = "passage_not_found"
                if reason:
                    issues.append(EvidenceValidationIssue(
                        example.id, hop_index, evidence.document, evidence.pages, reason
                    ))
    return issues


def evaluate_hop_coverage(
    example: MultiHopExample, results: Sequence[SearchResult], *, top_k: int
) -> HopCoverage:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    recovered = tuple(
        index for index, hop in enumerate(example.hops)
        if any(hop_matches_result(hop, result) for result in results[:top_k])
    )
    total = len(example.hops)
    return HopCoverage(recovered, total, len(recovered), len(recovered) == total, len(recovered) / total)


def aggregate_hop_coverage(
    examples: Sequence[MultiHopExample], rankings: Sequence[Sequence[SearchResult]], *, top_k: int
) -> dict[str, Any]:
    if len(examples) != len(rankings):
        raise ValueError("examples and rankings must have equal lengths")
    values = [evaluate_hop_coverage(example, results, top_k=top_k) for example, results in zip(examples, rankings)]
    total = len(values)
    return {
        "questions": total,
        "full_hop_coverage": sum(value.full for value in values) / total if total else 0.0,
        "partial_hop_coverage": sum(value.partial for value in values) / total if total else 0.0,
        "average_hops_recovered": sum(value.recovered_hops for value in values) / total if total else 0.0,
        "zero_hops": sum(value.recovered_hops == 0 for value in values),
        "one_hop": sum(value.recovered_hops == 1 for value in values),
        "all_hops": sum(value.full for value in values),
    }


def _contributions(rankings: Sequence[BranchRanking]) -> dict[str, Tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    for branch in rankings:
        for result in branch.results:
            labels = values.setdefault(result.chunk.chunk_id, [])
            if branch.query not in labels:
                labels.append(branch.query)
    return {key: tuple(value) for key, value in values.items()}


class MultiHopRetriever:
    """Compare baseline, decomposed-only, and guarded original+decomposed fusion."""

    def __init__(self, retriever: RerankingRetriever, *, rrf_k: int = 60) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        self.retriever = retriever
        self.rrf_k = rrf_k

    def _branch(self, query: str, top_k: int) -> BranchRanking:
        started = perf_counter()
        values = self.retriever.base.search(query, top_k=max(top_k, self.retriever.candidate_depth))
        return BranchRanking(query, tuple(values), perf_counter() - started)

    def _condition(self, original: str, branches: Sequence[BranchRanking], top_k: int) -> FusedCondition:
        fusion_started = perf_counter()
        candidates = reciprocal_rank_fusion(
            [branch.results for branch in branches],
            top_k=self.retriever.candidate_depth,
            rrf_k=self.rrf_k,
        )
        fusion_seconds = perf_counter() - fusion_started
        rerank_started = perf_counter()
        results = self.retriever.reranker.rerank(original, candidates, top_k)
        reranking_seconds = perf_counter() - rerank_started
        occurrences = sum(len(branch.results) for branch in branches)
        unique = {result.chunk.chunk_id for branch in branches for result in branch.results}
        return FusedCondition(
            tuple(branches), tuple(candidates), tuple(results), _contributions(branches),
            occurrences - len(unique), fusion_seconds, reranking_seconds,
        )

    def search(self, original_question: str, decomposition: DecompositionResult, *, top_k: int = 10) -> MultiHopRetrievalResult:
        if not original_question.strip():
            raise ValueError("original_question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        original = original_question.strip()
        original_branch = self._branch(original, top_k)
        baseline = self._condition(original, [original_branch], top_k)
        if not decomposition.needs_decomposition or len(decomposition.subquestions) < 2:
            return MultiHopRetrievalResult(
                decomposition, baseline, baseline, baseline,
                decomposition.fallback, decomposition.error,
            )
        try:
            sub_branches = [self._branch(query, top_k) for query in decomposition.subquestions]
            decomposed = self._condition(original, sub_branches, top_k)
            guarded = self._condition(original, [original_branch, *sub_branches], top_k)
            return MultiHopRetrievalResult(decomposition, baseline, decomposed, guarded, False, None)
        except Exception as exc:
            return MultiHopRetrievalResult(
                decomposition, baseline, baseline, baseline, True,
                f"{type(exc).__name__}: {exc}",
            )
