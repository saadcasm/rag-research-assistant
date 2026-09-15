"""Cheap, explainable signals available before multi-query generation."""

import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..query_transform.rewriting import extract_protected_terms


_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "paper", "the",
    "their", "this", "to", "was", "what", "when", "where", "which", "who", "why",
    "with",
}


@dataclass(frozen=True)
class RankedCandidate:
    score: float
    document: str
    start_page: int
    end_page: int
    chunk_id: str
    text: str


@dataclass(frozen=True)
class BaselineSignals:
    top_reranker_score: Optional[float]
    reranker_gap_1_2: Optional[float]
    reranker_gap_1_3: Optional[float]
    reranker_score_spread: Optional[float]
    reranker_score_stddev: Optional[float]
    unique_documents_top_k: int
    dominant_document_share: float
    documents_with_multiple_chunks: int
    query_term_coverage_top_1: float
    query_term_coverage_top_k: float
    protected_term_count: int
    protected_term_coverage_top_k: Optional[float]
    hybrid_reranked_chunk_overlap_at_5: float
    hybrid_reranked_document_overlap_at_5: float
    hybrid_top_chunk_retained: bool
    hybrid_top_document_retained: bool
    hybrid_mean_rank_shift: Optional[float]

    def numeric(self) -> Dict[str, float]:
        values: Dict[str, float] = {}
        for name, value in self.__dict__.items():
            if value is not None and isinstance(value, (int, float, bool)):
                values[name] = float(value)
        return values


def _content_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(text)
        if len(token) > 1 and token.casefold() not in _STOPWORDS
    }


def _coverage(terms: set[str], candidates: Sequence[RankedCandidate]) -> float:
    if not terms:
        return 1.0
    evidence = _content_terms(" ".join(candidate.text for candidate in candidates))
    return len(terms & evidence) / len(terms)


def _set_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def extract_baseline_signals(
    question: str,
    reranked: Sequence[RankedCandidate],
    hybrid: Sequence[RankedCandidate],
) -> BaselineSignals:
    """Extract only signals observable after the first baseline retrieval."""

    if not question.strip():
        raise ValueError("question cannot be empty")
    scores = [item.score for item in reranked]
    documents = [item.document for item in reranked]
    document_counts = {document: documents.count(document) for document in set(documents)}
    query_terms = _content_terms(question)
    protected = extract_protected_terms(question)
    protected_evidence = " ".join(item.text for item in reranked).casefold()
    protected_coverage = (
        sum(term.casefold() in protected_evidence for term in protected) / len(protected)
        if protected
        else None
    )
    hybrid_ids = [item.chunk_id for item in hybrid[:5]]
    reranked_ids = [item.chunk_id for item in reranked[:5]]
    hybrid_docs = [item.document for item in hybrid[:5]]
    reranked_docs = [item.document for item in reranked[:5]]
    reranked_positions = {chunk_id: rank for rank, chunk_id in enumerate(reranked_ids, 1)}
    shifts = [
        abs(rank - reranked_positions[chunk_id])
        for rank, chunk_id in enumerate(hybrid_ids, 1)
        if chunk_id in reranked_positions
    ]
    return BaselineSignals(
        top_reranker_score=scores[0] if scores else None,
        reranker_gap_1_2=scores[0] - scores[1] if len(scores) >= 2 else None,
        reranker_gap_1_3=scores[0] - scores[2] if len(scores) >= 3 else None,
        reranker_score_spread=max(scores) - min(scores) if scores else None,
        reranker_score_stddev=statistics.pstdev(scores) if scores else None,
        unique_documents_top_k=len(set(documents)),
        dominant_document_share=(max(document_counts.values()) / len(documents) if documents else 0.0),
        documents_with_multiple_chunks=sum(count > 1 for count in document_counts.values()),
        query_term_coverage_top_1=_coverage(query_terms, reranked[:1]),
        query_term_coverage_top_k=_coverage(query_terms, reranked),
        protected_term_count=len(protected),
        protected_term_coverage_top_k=protected_coverage,
        hybrid_reranked_chunk_overlap_at_5=_set_overlap(hybrid_ids, reranked_ids),
        hybrid_reranked_document_overlap_at_5=_set_overlap(hybrid_docs, reranked_docs),
        hybrid_top_chunk_retained=bool(hybrid_ids and hybrid_ids[0] in reranked_ids),
        hybrid_top_document_retained=bool(hybrid_docs and hybrid_docs[0] in reranked_docs),
        hybrid_mean_rank_shift=statistics.mean(shifts) if shifts else None,
    )


@dataclass(frozen=True)
class RoutingExample:
    question_id: str
    mq2_outcome: str
    baseline_first_relevant_rank: Optional[int]
    mq2_first_relevant_rank: Optional[int]
    expected_sources: Tuple[Mapping[str, object], ...]
    baseline_results: Tuple[Mapping[str, object], ...]
    mq2_results: Tuple[Mapping[str, object], ...]
    signals: BaselineSignals


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mean_first_relevant_rank: Optional[float]


@dataclass(frozen=True)
class RoutingMetrics:
    total_questions: int
    helpful_questions: int
    escalated_questions: int
    rescued_questions: int
    missed_rescues: int
    unnecessary_escalations: int
    escalation_rate: float
    rescue_recall: float
    routing_precision: float
    estimated_average_latency_seconds: float
    retrieval: RetrievalMetrics


@dataclass(frozen=True)
class ThresholdRule:
    signal: str
    direction: str
    threshold: float

    def routes(self, signals: BaselineSignals) -> bool:
        value = signals.numeric().get(self.signal)
        if value is None:
            return False
        if self.direction == "at_or_below":
            return value <= self.threshold
        if self.direction == "at_or_above":
            return value >= self.threshold
        raise ValueError(f"unknown threshold direction: {self.direction}")


def join_mq2_outcomes(
    signals: Mapping[str, BaselineSignals], outcomes: Mapping[str, str]
) -> Dict[str, Tuple[BaselineSignals, str]]:
    """Join by stable question ID and reject silent label/signal mismatches."""

    if set(signals) != set(outcomes):
        missing_labels = sorted(set(signals) - set(outcomes))
        missing_signals = sorted(set(outcomes) - set(signals))
        raise ValueError(
            f"signal/outcome IDs differ; missing_labels={missing_labels}, "
            f"missing_signals={missing_signals}"
        )
    return {question_id: (signals[question_id], outcomes[question_id]) for question_id in signals}


def _source_matches(source: Mapping[str, object], result: Mapping[str, object]) -> bool:
    if source["document"] != result["document"]:
        return False
    page = source.get("page_number")
    if page is not None and not (int(result["start_page"]) <= int(page) <= int(result["end_page"])):
        return False
    chunk_id = source.get("chunk_id")
    return chunk_id is None or chunk_id == result["chunk_id"]


def _retrieval_metrics(examples: Sequence[RoutingExample], decisions: Sequence[bool]) -> RetrievalMetrics:
    hits = {1: [], 3: [], 5: []}
    recalls = {1: [], 3: [], 5: []}
    ranks: List[int] = []
    for example, escalate in zip(examples, decisions):
        results = example.mq2_results if escalate else example.baseline_results
        rank = example.mq2_first_relevant_rank if escalate else example.baseline_first_relevant_rank
        if rank is not None:
            ranks.append(rank)
        for k in (1, 3, 5):
            matched = {
                index
                for index, source in enumerate(example.expected_sources)
                if any(_source_matches(source, result) for result in results[:k])
            }
            hits[k].append(bool(matched))
            recalls[k].append(len(matched) / len(example.expected_sources))
    average = lambda values: sum(values) / len(values) if values else 0.0
    return RetrievalMetrics(
        hit_at_1=average(hits[1]), hit_at_3=average(hits[3]), hit_at_5=average(hits[5]),
        recall_at_1=average(recalls[1]), recall_at_3=average(recalls[3]), recall_at_5=average(recalls[5]),
        mean_first_relevant_rank=average(ranks) if ranks else None,
    )


def evaluate_routing_decisions(
    examples: Sequence[RoutingExample],
    decisions: Sequence[bool],
    *,
    baseline_latency_seconds: float,
    mq2_latency_seconds: float,
) -> RoutingMetrics:
    if not examples or len(examples) != len(decisions):
        raise ValueError("examples and decisions must have the same non-zero length")
    if baseline_latency_seconds < 0 or mq2_latency_seconds < 0:
        raise ValueError("latencies cannot be negative")
    helpful = [example.mq2_outcome == "improved" for example in examples]
    escalated = sum(decisions)
    rescued = sum(route and benefit for route, benefit in zip(decisions, helpful))
    helpful_count = sum(helpful)
    unnecessary = sum(route and not benefit for route, benefit in zip(decisions, helpful))
    rate = escalated / len(examples)
    return RoutingMetrics(
        total_questions=len(examples),
        helpful_questions=helpful_count,
        escalated_questions=escalated,
        rescued_questions=rescued,
        missed_rescues=helpful_count - rescued,
        unnecessary_escalations=unnecessary,
        escalation_rate=rate,
        rescue_recall=rescued / helpful_count if helpful_count else 0.0,
        routing_precision=rescued / escalated if escalated else 0.0,
        estimated_average_latency_seconds=(1.0 - rate) * baseline_latency_seconds + rate * mq2_latency_seconds,
        retrieval=_retrieval_metrics(examples, decisions),
    )


def evaluate_threshold_rule(
    examples: Sequence[RoutingExample],
    rule: ThresholdRule,
    *,
    baseline_latency_seconds: float,
    mq2_latency_seconds: float,
) -> RoutingMetrics:
    return evaluate_routing_decisions(
        examples,
        [rule.routes(example.signals) for example in examples],
        baseline_latency_seconds=baseline_latency_seconds,
        mq2_latency_seconds=mq2_latency_seconds,
    )
