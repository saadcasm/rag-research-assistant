"""Label-safe diagnostics for dense, BM25, and pre-rerank RRF candidates."""

from dataclasses import dataclass
from statistics import median
from time import perf_counter
from typing import Any, Mapping, Sequence, Tuple

from ..bm25 import tokenize
from ..hybrid import reciprocal_rank_fusion
from ..models import SearchResult
from ..multihop import ExpectedHop, MultiHopExample, hop_matches_result
from ..query_transform.rewriting import extract_protected_terms
from ..retrievers import Retriever


BRANCHES = ("dense", "bm25", "hybrid")
DEFAULT_DEPTHS = (5, 10, 20, 50, 100)
RRF_DEMOTION_THRESHOLD = 10


@dataclass(frozen=True)
class ComponentRankings:
    """Rankings collected without access to expected documents or passages."""

    question_id: str
    question: str
    dense: Tuple[SearchResult, ...]
    bm25: Tuple[SearchResult, ...]
    hybrid: Tuple[SearchResult, ...]
    latency_seconds: Mapping[str, float]
    max_depth: int
    rrf_k: int

    def branch(self, name: str) -> Tuple[SearchResult, ...]:
        if name not in BRANCHES:
            raise ValueError(f"unknown retrieval branch: {name}")
        return getattr(self, name)


def retrieve_component_rankings(
    question_id: str,
    question: str,
    dense: Retriever,
    bm25: Retriever,
    *,
    max_depth: int = 100,
    rrf_k: int = 60,
) -> ComponentRankings:
    """Run the original question once per branch, then fuse without gold labels."""

    if not question_id.strip():
        raise ValueError("question_id cannot be empty")
    if not question.strip():
        raise ValueError("question cannot be empty")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    started = perf_counter()
    dense_results = tuple(dense.search(question, top_k=max_depth))
    dense_seconds = perf_counter() - started
    started = perf_counter()
    bm25_results = tuple(bm25.search(question, top_k=max_depth))
    bm25_seconds = perf_counter() - started
    started = perf_counter()
    hybrid_results = tuple(
        reciprocal_rank_fusion(
            [dense_results, bm25_results], top_k=max_depth, rrf_k=rrf_k
        )
    )
    hybrid_seconds = perf_counter() - started
    return ComponentRankings(
        question_id.strip(),
        question.strip(),
        dense_results,
        bm25_results,
        hybrid_results,
        {
            "dense": dense_seconds,
            "bm25": bm25_seconds,
            "rrf": hybrid_seconds,
            "total": dense_seconds + bm25_seconds + hybrid_seconds,
        },
        max_depth,
        rrf_k,
    )


def _first_passage_hit(
    hop: ExpectedHop, ranking: Sequence[SearchResult]
) -> tuple[int | None, SearchResult | None]:
    return next(
        (
            (rank, result)
            for rank, result in enumerate(ranking, 1)
            if hop_matches_result(hop, result)
        ),
        (None, None),
    )


def _expected_documents(hop: ExpectedHop) -> set[str]:
    return {evidence.document for evidence in hop.evidence}


def _first_document_hit(
    hop: ExpectedHop, ranking: Sequence[SearchResult]
) -> tuple[int | None, SearchResult | None]:
    documents = _expected_documents(hop)
    return next(
        (
            (rank, result)
            for rank, result in enumerate(ranking, 1)
            if result.chunk.document in documents
        ),
        (None, None),
    )


def _rank_for_chunk(chunk_id: str, ranking: Sequence[SearchResult]) -> int | None:
    return next(
        (rank for rank, result in enumerate(ranking, 1) if result.chunk.chunk_id == chunk_id),
        None,
    )


def _lexical_diagnostics(question: str, hop: ExpectedHop) -> dict[str, Any]:
    evidence_text = " ".join(evidence.passage for evidence in hop.evidence)
    question_tokens = set(tokenize(question))
    evidence_tokens = set(tokenize(evidence_text))
    shared = sorted(question_tokens & evidence_tokens)
    union = question_tokens | evidence_tokens
    protected = extract_protected_terms(question)
    evidence_folded = evidence_text.casefold()
    return {
        "question_token_count": len(question_tokens),
        "evidence_token_count": len(evidence_tokens),
        "shared_terms": shared,
        "shared_term_count": len(shared),
        "query_term_coverage": (
            len(shared) / len(question_tokens) if question_tokens else 1.0
        ),
        "jaccard_overlap": len(shared) / len(union) if union else 1.0,
        "query_terms_absent_from_evidence": sorted(question_tokens - evidence_tokens),
        "protected_terms": list(protected),
        "protected_terms_in_evidence": [
            term for term in protected if term.casefold() in evidence_folded
        ],
        "protected_terms_missing_from_evidence": [
            term for term in protected if term.casefold() not in evidence_folded
        ],
        "possible_pdf_artifacts": {
            "line_hyphenation": "-\n" in evidence_text or "- \n" in evidence_text,
            "ligatures": any(character in evidence_text for character in "ﬁﬂﬀﬃﬄ"),
            "replacement_character": "�" in evidence_text,
        },
    }


def _branch_outcome(dense_rank: int | None, bm25_rank: int | None) -> str:
    if dense_rank is not None and bm25_rank is not None:
        return "DENSE_AND_BM25_FOUND"
    if dense_rank is not None:
        return "DENSE_ONLY_FOUND"
    if bm25_rank is not None:
        return "BM25_ONLY_FOUND"
    return "FULL_MISS"


def _hop_diagnostic(
    question: str,
    hop_index: int,
    hop: ExpectedHop,
    rankings: ComponentRankings,
    depths: Sequence[int],
) -> dict[str, Any]:
    if not hop.evidence_available:
        return {
            "hop_index": hop_index,
            "hop_description": hop.description,
            "evidence_available": False,
            "diagnostic_classification": "INTENTIONALLY_UNAVAILABLE",
            "diagnostic_flags": [],
        }

    passage_hits: dict[str, tuple[int | None, SearchResult | None]] = {}
    document_hits: dict[str, tuple[int | None, SearchResult | None]] = {}
    for branch in BRANCHES:
        ranking = rankings.branch(branch)
        passage_hits[branch] = _first_passage_hit(hop, ranking)
        document_hits[branch] = _first_document_hit(hop, ranking)

    dense_rank, dense_result = passage_hits["dense"]
    bm25_rank, bm25_result = passage_hits["bm25"]
    hybrid_rank, hybrid_result = passage_hits["hybrid"]
    found_branch_ranks = [rank for rank in (dense_rank, bm25_rank) if rank is not None]
    best_branch_rank = min(found_branch_ranks) if found_branch_ranks else None
    rrf_rank_shift = (
        hybrid_rank - best_branch_rank
        if hybrid_rank is not None and best_branch_rank is not None
        else None
    )
    rrf_demoted = bool(
        best_branch_rank is not None
        and best_branch_rank <= 20
        and (
            hybrid_rank is None
            or hybrid_rank - best_branch_rank >= RRF_DEMOTION_THRESHOLD
        )
    )
    all_passage_ranks = [
        rank for rank in (dense_rank, bm25_rank, hybrid_rank) if rank is not None
    ]
    any_passage_rank = min(all_passage_ranks) if all_passage_ranks else None
    found_only_deep = bool(
        any(
            rank is not None and rank > 20
            for rank in (dense_rank, bm25_rank, hybrid_rank)
        )
        and all(rank is None or rank > 20 for rank in (dense_rank, bm25_rank, hybrid_rank))
    )
    document_found_passage_missed = bool(
        dense_rank is None
        and bm25_rank is None
        and any(rank is not None for rank, _ in document_hits.values())
    )
    flags = []
    if found_only_deep:
        flags.append("FOUND_ONLY_DEEP")
    if document_found_passage_missed:
        flags.append("DOCUMENT_FOUND_PASSAGE_MISSED")
    if rrf_demoted:
        flags.append("BOTH_FOUND_BUT_RRF_DEMOTED")

    hybrid_chunk_id = hybrid_result.chunk.chunk_id if hybrid_result else None
    branch_contribution = {
        "hybrid_matching_chunk_id": hybrid_chunk_id,
        "dense_rank_for_hybrid_chunk": (
            _rank_for_chunk(hybrid_chunk_id, rankings.dense) if hybrid_chunk_id else None
        ),
        "bm25_rank_for_hybrid_chunk": (
            _rank_for_chunk(hybrid_chunk_id, rankings.bm25) if hybrid_chunk_id else None
        ),
        "contributing_branches": (
            [
                branch
                for branch, ranking in (("dense", rankings.dense), ("bm25", rankings.bm25))
                if hybrid_chunk_id and _rank_for_chunk(hybrid_chunk_id, ranking) is not None
            ]
        ),
        "rrf_score": hybrid_result.score if hybrid_result else None,
        "best_branch_passage_rank": best_branch_rank,
        "rrf_passage_rank_shift": rrf_rank_shift,
        "significantly_demoted": rrf_demoted,
    }
    expected_documents = sorted(_expected_documents(hop))
    expected_pages = sorted(
        {page for evidence in hop.evidence for page in evidence.pages}
    )
    evidence_passages = [evidence.passage for evidence in hop.evidence]
    return {
        "hop_index": hop_index,
        "hop_description": hop.description,
        "evidence_available": True,
        "expected_documents": expected_documents,
        "expected_pages": expected_pages,
        "evidence_passages": evidence_passages,
        "first_passage_rank": {
            branch: passage_hits[branch][0] for branch in BRANCHES
        },
        "first_passage_score": {
            branch: (passage_hits[branch][1].score if passage_hits[branch][1] else None)
            for branch in BRANCHES
        },
        "first_document_rank": {
            branch: document_hits[branch][0] for branch in BRANCHES
        },
        "passage_found_at": {
            branch: {
                str(depth): (
                    passage_hits[branch][0] is not None
                    and passage_hits[branch][0] <= depth
                )
                for depth in depths
            }
            for branch in BRANCHES
        },
        "document_found_at": {
            branch: {
                str(depth): (
                    document_hits[branch][0] is not None
                    and document_hits[branch][0] <= depth
                )
                for depth in depths
            }
            for branch in BRANCHES
        },
        "document_found_passage_missing_at": {
            branch: {
                str(depth): (
                    document_hits[branch][0] is not None
                    and document_hits[branch][0] <= depth
                    and (
                        passage_hits[branch][0] is None
                        or passage_hits[branch][0] > depth
                    )
                )
                for depth in depths
            }
            for branch in BRANCHES
        },
        "diagnostic_classification": _branch_outcome(dense_rank, bm25_rank),
        "diagnostic_flags": flags,
        "first_passage_rank_any_branch": any_passage_rank,
        "branch_contribution": branch_contribution,
        "lexical_diagnostics": _lexical_diagnostics(question, hop),
    }


def _coverage_for_branch(
    hops: Sequence[dict[str, Any]], branch: str, depth: int
) -> dict[str, Any]:
    available = [hop for hop in hops if hop["evidence_available"]]
    recovered = [
        hop["hop_index"]
        for hop in available
        if hop["passage_found_at"][branch][str(depth)]
    ]
    total = len(available)
    return {
        "available_hops": total,
        "recovered_hop_indexes": recovered,
        "recovered_hops": len(recovered),
        "full_hop_availability": len(recovered) == total,
        "partial_hop_availability": len(recovered) / total if total else 1.0,
    }


def evaluate_component_rankings(
    example: MultiHopExample,
    rankings: ComponentRankings,
    *,
    depths: Sequence[int] = DEFAULT_DEPTHS,
) -> dict[str, Any]:
    """Apply gold evidence only after all three rankings have been frozen."""

    if example.id != rankings.question_id or example.question != rankings.question:
        raise ValueError("rankings do not correspond to the evaluation example")
    normalized_depths = tuple(dict.fromkeys(depths))
    if not normalized_depths or any(depth <= 0 for depth in normalized_depths):
        raise ValueError("depths must contain positive integers")
    if max(normalized_depths) > rankings.max_depth:
        raise ValueError("requested evaluation depth exceeds retrieved depth")
    hops = [
        _hop_diagnostic(example.question, index, hop, rankings, normalized_depths)
        for index, hop in enumerate(example.hops, 1)
    ]
    return {
        "question_id": example.id,
        "question": example.question,
        "answerability": example.answerability,
        "category": example.category,
        "required_hop_count": len(example.hops),
        "available_hop_count": sum(hop.evidence_available for hop in example.hops),
        "rankings": {
            branch: [
                {
                    "rank": rank,
                    "score": result.score,
                    "chunk_id": result.chunk.chunk_id,
                    "document": result.chunk.document,
                    "start_page": result.chunk.start_page,
                    "end_page": result.chunk.end_page,
                }
                for rank, result in enumerate(rankings.branch(branch), 1)
            ]
            for branch in BRANCHES
        },
        "latency_seconds": dict(rankings.latency_seconds),
        "hops": hops,
        "coverage": {
            branch: {
                str(depth): _coverage_for_branch(hops, branch, depth)
                for depth in normalized_depths
            }
            for branch in BRANCHES
        },
    }


def _nearest_rank_percentile(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int((percentile * len(ordered) + 0.999999999) // 1) - 1)
    return float(ordered[min(index, len(ordered) - 1)])


def aggregate_component_diagnostics(
    records: Sequence[Mapping[str, Any]], *, depths: Sequence[int] = DEFAULT_DEPTHS
) -> dict[str, Any]:
    """Aggregate passage/document recall and available-hop question coverage."""

    available_hops = [
        hop
        for record in records
        for hop in record["hops"]
        if hop["evidence_available"]
    ]
    hop_count = len(available_hops)
    branch_metrics: dict[str, Any] = {}
    for branch in BRANCHES:
        found_ranks = [
            hop["first_passage_rank"][branch]
            for hop in available_hops
            if hop["first_passage_rank"][branch] is not None
        ]
        branch_metrics[branch] = {
            "passage_recall": {
                str(depth): (
                    sum(hop["passage_found_at"][branch][str(depth)] for hop in available_hops)
                    / hop_count
                    if hop_count
                    else 0.0
                )
                for depth in depths
            },
            "document_recall": {
                str(depth): (
                    sum(hop["document_found_at"][branch][str(depth)] for hop in available_hops)
                    / hop_count
                    if hop_count
                    else 0.0
                )
                for depth in depths
            },
            "document_found_passage_missing_rate": {
                str(depth): (
                    sum(
                        hop["document_found_passage_missing_at"][branch][str(depth)]
                        for hop in available_hops
                    )
                    / hop_count
                    if hop_count
                    else 0.0
                )
                for depth in depths
            },
            "first_passage_rank": {
                "found_hops": len(found_ranks),
                "median": float(median(found_ranks)) if found_ranks else None,
                "p75": _nearest_rank_percentile(found_ranks, 0.75),
                "p90": _nearest_rank_percentile(found_ranks, 0.90),
            },
            "question_coverage": {
                str(depth): _aggregate_question_coverage(records, branch, depth)
                for depth in depths
            },
        }
    classifications = {
        name: sum(hop["diagnostic_classification"] == name for hop in available_hops)
        for name in (
            "DENSE_AND_BM25_FOUND",
            "DENSE_ONLY_FOUND",
            "BM25_ONLY_FOUND",
            "FULL_MISS",
        )
    }
    return {
        "questions": len(records),
        "available_hops": hop_count,
        "branches": branch_metrics,
        "classification_counts": classifications,
        "classification_fractions": {
            name: count / hop_count if hop_count else 0.0
            for name, count in classifications.items()
        },
        "flag_counts": {
            name: sum(name in hop["diagnostic_flags"] for hop in available_hops)
            for name in (
                "FOUND_ONLY_DEEP",
                "DOCUMENT_FOUND_PASSAGE_MISSED",
                "BOTH_FOUND_BUT_RRF_DEMOTED",
            )
        },
    }


def _aggregate_question_coverage(
    records: Sequence[Mapping[str, Any]], branch: str, depth: int
) -> dict[str, Any]:
    values = [record["coverage"][branch][str(depth)] for record in records]
    total = len(values)
    recovered = [value["recovered_hops"] for value in values]
    return {
        "questions": total,
        "full_hop_availability": (
            sum(value["full_hop_availability"] for value in values) / total
            if total
            else 0.0
        ),
        "partial_hop_availability": (
            sum(value["partial_hop_availability"] for value in values) / total
            if total
            else 0.0
        ),
        "average_recovered_hops": sum(recovered) / total if total else 0.0,
        "zero_hop_questions": sum(value == 0 for value in recovered),
        "one_hop_questions": sum(value == 1 for value in recovered),
        "all_hop_questions": sum(
            value["full_hop_availability"] for value in values
        ),
    }
