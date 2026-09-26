import json

import pytest

from rag_research_assistant.evidence_selection.candidate_coverage import (
    CandidatePool,
    aggregate_candidate_coverage,
    evaluate_candidate_pool,
    load_phase10g_candidate_pools,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.multihop import ExpectedHop, HopEvidence, MultiHopExample


def candidate(identifier: str, document: str, page: int, text: str) -> SearchResult:
    return SearchResult(
        1.0,
        Chunk(identifier, document, page, 0, 0, len(text), text),
    )


def example(*, partially_answerable: bool = False) -> MultiHopExample:
    second = ExpectedHop(
        "second fact",
        ("beta",),
        () if partially_answerable else (HopEvidence("b.pdf", (2,), "beta evidence"),),
        not partially_answerable,
    )
    return MultiHopExample(
        "q1",
        "Compare alpha and beta",
        "partially_answerable" if partially_answerable else "answerable",
        "comparison",
        (
            ExpectedHop("first fact", ("alpha",), (HopEvidence("a.pdf", (1,), "alpha evidence"),)),
            second,
        ),
        "verified",
    )


def test_full_and_partial_candidate_pool_availability() -> None:
    pool = CandidatePool(
        "q1",
        "Compare alpha and beta",
        (
            candidate("noise", "x.pdf", 1, "noise"),
            candidate("alpha", "a.pdf", 1, "alpha evidence"),
            candidate("other", "x.pdf", 2, "other"),
            candidate("beta", "b.pdf", 2, "beta evidence"),
        ),
    )
    evaluated = evaluate_candidate_pool(example(), pool, requested_depths=(2, 4, 5))

    assert [hop.first_candidate_rank for hop in evaluated.hop_availability] == [2, 4]
    assert evaluated.depths[2].partial == 0.5
    assert evaluated.depths[2].full is False
    assert evaluated.depths[4].partial == 1.0
    assert evaluated.depths[4].full is True
    assert evaluated.depths[5].depth_available is False
    assert evaluated.depths[5].full is None


def test_partially_answerable_reports_required_and_available_coverage_separately() -> None:
    pool = CandidatePool(
        "q1",
        "Compare alpha and beta",
        (candidate("alpha", "a.pdf", 1, "alpha evidence"),),
    )
    evaluated = evaluate_candidate_pool(
        example(partially_answerable=True), pool, requested_depths=(1,)
    )
    row = evaluated.depths[1]
    assert row.full is False
    assert row.partial == 0.5
    assert row.available_full is True
    assert row.available_partial == 1.0


def test_evaluation_never_reorders_or_mutates_candidate_pool() -> None:
    candidates = (
        candidate("beta", "b.pdf", 2, "beta evidence"),
        candidate("alpha", "a.pdf", 1, "alpha evidence"),
    )
    pool = CandidatePool("q1", "Compare alpha and beta", candidates)
    before = tuple(item.chunk.chunk_id for item in pool.candidates)
    evaluated = evaluate_candidate_pool(example(), pool, requested_depths=(2,))

    assert tuple(item.chunk.chunk_id for item in pool.candidates) == before
    assert evaluated.candidate_chunk_ids == before
    assert evaluated.candidate_ranks == (1, 2)


def test_pool_loading_is_label_free_and_deduplicates_stably() -> None:
    chunks = [
        candidate("a", "a.pdf", 1, "alpha").chunk,
        candidate("b", "b.pdf", 2, "beta").chunk,
    ]
    artifact = {
        "questions": [{
            "question_id": "q1",
            "question": "Compare alpha and beta",
            # Deliberately poisonous gold-like data: the loader must ignore it.
            "expected_hops": [{"evidence": [{"document": "gold.pdf"}]}],
            "conditions": {"baseline": {"branch_rankings": [{"results": [
                {"rank": 1, "chunk_id": "b", "score": 0.9},
                {"rank": 2, "chunk_id": "a", "score": 0.8},
                {"rank": 3, "chunk_id": "b", "score": 0.7},
            ]}]}},
        }],
    }

    pool = load_phase10g_candidate_pools(artifact, chunks)[0]
    assert [item.chunk.chunk_id for item in pool.candidates] == ["b", "a"]
    assert pool.duplicate_chunk_ids == ("b",)


def test_aggregate_and_serialization_include_oracle_counts_and_latency() -> None:
    pool = CandidatePool(
        "q1",
        "Compare alpha and beta",
        (
            candidate("alpha", "a.pdf", 1, "alpha evidence"),
            candidate("beta", "b.pdf", 2, "beta evidence"),
        ),
    )
    evaluated = evaluate_candidate_pool(example(), pool, requested_depths=(2, 3))
    summary = aggregate_candidate_coverage([evaluated], 2)

    assert summary["oracle_full_hop_availability"] == 1.0
    assert summary["all_hops"] == 1
    assert evaluated.evaluation_seconds >= 0.0
    serialized = evaluated.to_dict()
    assert serialized["depths"]["2"]["available_full"] is True
    assert json.loads(json.dumps(serialized))["candidate_chunk_ids"] == ["alpha", "beta"]


def test_pool_and_example_identity_must_match() -> None:
    pool = CandidatePool("wrong", "Compare alpha and beta", ())
    with pytest.raises(ValueError, match="does not correspond"):
        evaluate_candidate_pool(example(), pool, requested_depths=(1,))
