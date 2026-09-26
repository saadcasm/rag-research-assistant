import json

import pytest

from rag_research_assistant.candidate_generation.diagnostics import (
    ComponentRankings,
    aggregate_component_diagnostics,
    evaluate_component_rankings,
    retrieve_component_rankings,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.multihop import ExpectedHop, HopEvidence, MultiHopExample


def result(identifier: str, document: str, page: int, text: str, score=1.0) -> SearchResult:
    return SearchResult(
        score,
        Chunk(identifier, document, page, 1, 0, len(text), text),
    )


def example(*, partial: bool = False) -> MultiHopExample:
    return MultiHopExample(
        "q1",
        "How do DPR and ANCE differ?",
        "partially_answerable" if partial else "answerable",
        "cross_paper_comparison",
        (
            ExpectedHop(
                "DPR evidence",
                ("DPR fact",),
                (HopEvidence("dpr.pdf", (3,), "DPR uses in-batch negatives"),),
            ),
            ExpectedHop(
                "ANCE evidence",
                ("ANCE fact",),
                () if partial else (HopEvidence("ance.pdf", (5,), "ANCE mines negatives"),),
                not partial,
            ),
        ),
        "verified",
    )


def rankings(dense=(), bm25=(), hybrid=(), max_depth=50) -> ComponentRankings:
    return ComponentRankings(
        "q1",
        "How do DPR and ANCE differ?",
        tuple(dense),
        tuple(bm25),
        tuple(hybrid),
        {"dense": 0.1, "bm25": 0.2, "rrf": 0.01, "total": 0.31},
        max_depth,
        60,
    )


class FakeRetriever:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.values[:top_k]


def test_label_free_retrieval_collects_original_query_and_three_rankings() -> None:
    dense_values = [result("d", "dpr.pdf", 3, "DPR uses in-batch negatives")]
    bm25_values = [result("b", "ance.pdf", 5, "ANCE mines negatives")]
    dense = FakeRetriever(dense_values)
    bm25 = FakeRetriever(bm25_values)

    collected = retrieve_component_rankings(
        "q1", "How do DPR and ANCE differ?", dense, bm25, max_depth=10
    )

    assert dense.calls == [("How do DPR and ANCE differ?", 10)]
    assert bm25.calls == [("How do DPR and ANCE differ?", 10)]
    assert [item.chunk.chunk_id for item in collected.dense] == ["d"]
    assert [item.chunk.chunk_id for item in collected.bm25] == ["b"]
    assert {item.chunk.chunk_id for item in collected.hybrid} == {"d", "b"}
    assert collected.latency_seconds["total"] >= 0.0


def test_first_passage_and_document_ranks_are_independent_per_branch() -> None:
    noise = result("n", "noise.pdf", 1, "noise")
    wrong_dpr_passage = result("dw", "dpr.pdf", 1, "unrelated DPR text")
    dpr = result("d", "dpr.pdf", 3, "DPR uses in-batch negatives")
    record = evaluate_component_rankings(
        example(),
        rankings(
            dense=[noise, dpr],
            bm25=[wrong_dpr_passage, noise, dpr],
            hybrid=[noise, wrong_dpr_passage, dpr],
        ),
        depths=(1, 3),
    )
    hop = record["hops"][0]

    assert hop["first_passage_rank"] == {"dense": 2, "bm25": 3, "hybrid": 3}
    assert hop["first_document_rank"] == {"dense": 2, "bm25": 1, "hybrid": 2}
    assert hop["passage_found_at"]["dense"] == {"1": False, "3": True}
    assert hop["document_found_at"]["bm25"]["1"] is True


@pytest.mark.parametrize(
    ("dense", "bm25", "expected"),
    [
        (True, False, "DENSE_ONLY_FOUND"),
        (False, True, "BM25_ONLY_FOUND"),
        (True, True, "DENSE_AND_BM25_FOUND"),
        (False, False, "FULL_MISS"),
    ],
)
def test_branch_classifications(dense, bm25, expected) -> None:
    passage = result("d", "dpr.pdf", 3, "DPR uses in-batch negatives")
    noise = result("n", "noise.pdf", 1, "noise")
    record = evaluate_component_rankings(
        example(),
        rankings(
            dense=[passage] if dense else [noise],
            bm25=[passage] if bm25 else [noise],
            hybrid=[passage] if (dense or bm25) else [noise],
        ),
        depths=(1,),
    )
    assert record["hops"][0]["diagnostic_classification"] == expected


def test_document_found_passage_missing_is_explicit() -> None:
    wrong = result("wrong", "dpr.pdf", 3, "correct paper but unrelated paragraph")
    record = evaluate_component_rankings(
        example(), rankings(dense=[wrong], bm25=[], hybrid=[wrong]), depths=(1,)
    )
    hop = record["hops"][0]

    assert hop["diagnostic_classification"] == "FULL_MISS"
    assert "DOCUMENT_FOUND_PASSAGE_MISSED" in hop["diagnostic_flags"]
    assert hop["document_found_passage_missing_at"]["dense"]["1"] is True


def test_deep_only_and_rrf_rank_shift_diagnostics() -> None:
    noise = [result(f"n{i}", "noise.pdf", 1, f"noise {i}") for i in range(24)]
    passage = result("d", "dpr.pdf", 3, "DPR uses in-batch negatives")
    deep_record = evaluate_component_rankings(
        example(),
        rankings(
            dense=[*noise[:20], passage],
            bm25=noise,
            hybrid=[*noise[:20], passage],
        ),
        depths=(20, 50),
    )
    assert "FOUND_ONLY_DEEP" in deep_record["hops"][0]["diagnostic_flags"]

    demoted_record = evaluate_component_rankings(
        example(),
        rankings(
            dense=[passage],
            bm25=[passage],
            hybrid=[*noise[:14], passage],
        ),
        depths=(20,),
    )
    contribution = demoted_record["hops"][0]["branch_contribution"]
    assert contribution["rrf_passage_rank_shift"] == 14
    assert contribution["significantly_demoted"] is True
    assert "BOTH_FOUND_BUT_RRF_DEMOTED" in demoted_record["hops"][0]["diagnostic_flags"]


def test_available_hop_coverage_excludes_intentionally_unavailable_hop() -> None:
    passage = result("d", "dpr.pdf", 3, "DPR uses in-batch negatives")
    record = evaluate_component_rankings(
        example(partial=True),
        rankings(dense=[passage], bm25=[], hybrid=[passage]),
        depths=(1,),
    )

    assert record["available_hop_count"] == 1
    assert record["hops"][1]["diagnostic_classification"] == "INTENTIONALLY_UNAVAILABLE"
    assert record["coverage"]["dense"]["1"]["full_hop_availability"] is True
    assert record["coverage"]["bm25"]["1"]["partial_hop_availability"] == 0.0


def test_aggregation_reports_recall_coverage_classification_and_serializes() -> None:
    dpr = result("d", "dpr.pdf", 3, "DPR uses in-batch negatives")
    ance = result("a", "ance.pdf", 5, "ANCE mines negatives")
    record = evaluate_component_rankings(
        example(),
        rankings(dense=[dpr, ance], bm25=[dpr], hybrid=[dpr, ance]),
        depths=(1, 2),
    )
    summary = aggregate_component_diagnostics([record], depths=(1, 2))

    assert summary["available_hops"] == 2
    assert summary["branches"]["dense"]["passage_recall"] == {"1": 0.5, "2": 1.0}
    assert summary["branches"]["dense"]["question_coverage"]["2"]["all_hop_questions"] == 1
    assert summary["classification_counts"]["DENSE_AND_BM25_FOUND"] == 1
    assert summary["classification_counts"]["DENSE_ONLY_FOUND"] == 1
    assert json.loads(json.dumps(record))["hops"][0]["first_passage_rank"]["dense"] == 1


def test_requested_depth_cannot_exceed_frozen_ranking_depth() -> None:
    with pytest.raises(ValueError, match="exceeds retrieved depth"):
        evaluate_component_rankings(example(), rankings(max_depth=20), depths=(50,))
