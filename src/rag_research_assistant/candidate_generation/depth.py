"""Controlled candidate-depth experiment primitives for Phase 10I-2."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence, Tuple

from ..hybrid import reciprocal_rank_fusion
from ..models import SearchResult
from ..multihop import MultiHopExample, hop_matches_result
from ..reranking import Reranker
from ..retrievers import Retriever


DEPTHS = (20, 50, 100)
FINAL_TOP_K = 10
RRF_K = 60


@dataclass(frozen=True)
class CandidateDepthRun:
    """One label-free retrieval and reranking run at an explicit branch depth."""

    question: str
    depth: int
    dense: Tuple[SearchResult, ...]
    bm25: Tuple[SearchResult, ...]
    fused: Tuple[SearchResult, ...]
    final: Tuple[SearchResult, ...]
    latency_seconds: Mapping[str, float]
    reranked_candidate_count: int
    final_top_k: int


def run_candidate_depth(
    question: str,
    dense: Retriever,
    bm25: Retriever,
    reranker: Reranker,
    *,
    depth: int,
    final_top_k: int = FINAL_TOP_K,
    rrf_k: int = RRF_K,
) -> CandidateDepthRun:
    """Vary only branch/fusion depth; no labels enter this function."""

    if not question.strip():
        raise ValueError("question cannot be empty")
    if depth <= 0 or final_top_k <= 0:
        raise ValueError("depth and final_top_k must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")
    started = perf_counter()
    dense_results = tuple(dense.search(question, top_k=depth))
    dense_seconds = perf_counter() - started
    started = perf_counter()
    bm25_results = tuple(bm25.search(question, top_k=depth))
    bm25_seconds = perf_counter() - started
    started = perf_counter()
    fused = tuple(
        reciprocal_rank_fusion(
            [dense_results, bm25_results], top_k=depth, rrf_k=rrf_k
        )
    )
    rrf_seconds = perf_counter() - started
    started = perf_counter()
    final = tuple(reranker.rerank(question, fused, final_top_k))
    reranker_seconds = perf_counter() - started
    return CandidateDepthRun(
        question.strip(), depth, dense_results, bm25_results, fused, final,
        {
            "dense": dense_seconds,
            "bm25": bm25_seconds,
            "rrf": rrf_seconds,
            "reranker": reranker_seconds,
            "total": dense_seconds + bm25_seconds + rrf_seconds + reranker_seconds,
        },
        len(fused), final_top_k,
    )


def _first_rank(hop, ranking: Sequence[SearchResult]) -> int | None:
    return next(
        (rank for rank, result in enumerate(ranking, 1) if hop_matches_result(hop, result)),
        None,
    )


def _coverage(example: MultiHopExample, ranking: Sequence[SearchResult], k: int) -> dict[str, Any]:
    available = [
        (index, hop) for index, hop in enumerate(example.hops, 1) if hop.evidence_available
    ]
    recovered = [
        index
        for index, hop in available
        if any(hop_matches_result(hop, result) for result in ranking[:k])
    ]
    total = len(available)
    return {
        "available_hops": total,
        "recovered_hop_indexes": recovered,
        "recovered_hops": len(recovered),
        "full": len(recovered) == total,
        "partial": len(recovered) / total if total else 1.0,
    }


def evaluate_depth_run(
    example: MultiHopExample,
    run: CandidateDepthRun,
    *,
    final_ks: Sequence[int] = (3, 5, 10),
) -> dict[str, Any]:
    """Apply gold hops after the depth-controlled rankings are frozen."""

    if example.question != run.question:
        raise ValueError("run does not correspond to the evaluation example")
    if any(k <= 0 or k > run.final_top_k for k in final_ks):
        raise ValueError("final evaluation depths must fit final_top_k")
    hops = []
    for index, hop in enumerate(example.hops, 1):
        if not hop.evidence_available:
            hops.append({
                "hop_index": index,
                "description": hop.description,
                "evidence_available": False,
            })
            continue
        hops.append({
            "hop_index": index,
            "description": hop.description,
            "evidence_available": True,
            "dense_rank": _first_rank(hop, run.dense),
            "bm25_rank": _first_rank(hop, run.bm25),
            "pre_rerank_rrf_rank": _first_rank(hop, run.fused),
            "post_rerank_rank": _first_rank(hop, run.final),
        })
    return {
        "depth": run.depth,
        "dense_candidate_count": len(run.dense),
        "bm25_candidate_count": len(run.bm25),
        "fused_candidate_count": len(run.fused),
        "reranked_candidate_count": run.reranked_candidate_count,
        "final_result_count": len(run.final),
        "pre_rerank": _rows(run.fused),
        "post_rerank": _rows(run.final),
        "hops": hops,
        "oracle_candidate_coverage": _coverage(example, run.fused, len(run.fused)),
        "final_coverage": {
            str(k): _coverage(example, run.final, k) for k in final_ks
        },
        "first_relevant_passage_rank": next(
            (
                rank
                for rank, result in enumerate(run.final, 1)
                if any(
                    hop.evidence_available and hop_matches_result(hop, result)
                    for hop in example.hops
                )
            ),
            None,
        ),
        "latency_seconds": dict(run.latency_seconds),
    }


def _rows(ranking: Sequence[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "score": result.score,
            "chunk_id": result.chunk.chunk_id,
            "document": result.chunk.document,
            "start_page": result.chunk.start_page,
            "end_page": result.chunk.end_page,
        }
        for rank, result in enumerate(ranking, 1)
    ]


def classify_movement(baseline: Mapping[str, Any], compared: Mapping[str, Any]) -> dict[str, Any]:
    """Classify final top-10 available-hop movement relative to depth 20."""

    before = baseline["final_coverage"]["10"]
    after = compared["final_coverage"]["10"]
    if after["recovered_hops"] > before["recovered_hops"]:
        outcome = "improved"
    elif after["recovered_hops"] < before["recovered_hops"]:
        outcome = "degraded"
    else:
        outcome = "unchanged"
    if before["recovered_hops"] == 0 and after["recovered_hops"] == 1:
        transition = "0-hop_to_1-hop"
    elif not before["full"] and before["recovered_hops"] == 1 and after["full"]:
        transition = "1-hop_to_full-hop"
    elif before["full"] and not after["full"]:
        transition = "full-hop_to_partial-hop"
    elif before["recovered_hops"] > 0 and after["recovered_hops"] == 0:
        transition = "partial-hop_to_0-hop"
    else:
        transition = f"{before['recovered_hops']}-hop_to_{after['recovered_hops']}-hop"
    rescued = sorted(set(after["recovered_hop_indexes"]) - set(before["recovered_hop_indexes"]))
    regressed = sorted(set(before["recovered_hop_indexes"]) - set(after["recovered_hop_indexes"]))
    hop_by_index = {hop["hop_index"]: hop for hop in compared["hops"]}
    return {
        "outcome": outcome,
        "transition": transition,
        "rescued_hop_indexes": rescued,
        "regressed_hop_indexes": regressed,
        "rescued_hop_promotions": [hop_by_index[index] for index in rescued],
    }


def aggregate_multihop(records: Sequence[Mapping[str, Any]], depth: int, k: int) -> dict[str, Any]:
    values = [record["conditions"][str(depth)]["final_coverage"][str(k)] for record in records]
    total = len(values)
    return {
        "questions": total,
        "full_hop_coverage": sum(value["full"] for value in values) / total if total else 0.0,
        "partial_hop_coverage": sum(value["partial"] for value in values) / total if total else 0.0,
        "average_recovered_hops": sum(value["recovered_hops"] for value in values) / total if total else 0.0,
        "zero_hop_questions": sum(value["recovered_hops"] == 0 for value in values),
        "one_hop_questions": sum(value["recovered_hops"] == 1 for value in values),
        "all_hop_questions": sum(value["full"] for value in values),
    }
