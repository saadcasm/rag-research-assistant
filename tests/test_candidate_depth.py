import pytest

from rag_research_assistant.candidate_generation.depth import (
    CandidateDepthRun,
    classify_movement,
    evaluate_depth_run,
    run_candidate_depth,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.multihop import ExpectedHop, HopEvidence, MultiHopExample


def item(index, text=None, document="paper.pdf", page=1):
    value = text or f"noise {index}"
    return SearchResult(1.0 / (index + 1), Chunk(f"c{index}", document, page, index, 0, len(value), value))


class FakeRetriever:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.values[:top_k]


class FakeReranker:
    model_name = "fake-cross-encoder"

    def __init__(self):
        self.candidate_counts = []

    def rerank(self, query, candidates, top_k):
        self.candidate_counts.append(len(candidates))
        ranked = sorted(candidates, key=lambda row: ("gold evidence" not in row.chunk.text, row.chunk.chunk_id))
        return [SearchResult(float(len(ranked) - i), row.chunk) for i, row in enumerate(ranked[:top_k])]


def example(partial=False):
    return MultiHopExample(
        "q", "original question", "partially_answerable" if partial else "answerable", "comparison",
        (
            ExpectedHop("gold", (), (HopEvidence("paper.pdf", (1,), "gold evidence"),)),
            ExpectedHop("second", (), () if partial else (HopEvidence("other.pdf", (2,), "second evidence"),), not partial),
        ), "verified",
    )


@pytest.mark.parametrize("depth", [20, 50, 100])
def test_depth_controls_both_branches_rrf_reranker_and_fixed_final_top_k(depth):
    values = [item(i, "gold evidence" if i == depth - 1 else None) for i in range(100)]
    dense, bm25, reranker = FakeRetriever(values), FakeRetriever(list(reversed(values))), FakeReranker()

    run = run_candidate_depth("original question", dense, bm25, reranker, depth=depth, final_top_k=10)

    assert dense.calls == [("original question", depth)]
    assert bm25.calls == [("original question", depth)]
    assert len(run.dense) == depth
    assert len(run.bm25) == depth
    assert len(run.fused) == depth
    assert run.reranked_candidate_count == depth
    assert reranker.candidate_counts == [depth]
    assert len(run.final) == 10
    assert run.final_top_k == 10
    assert all(value >= 0 for value in run.latency_seconds.values())


def make_run(depth, fused, final):
    return CandidateDepthRun(
        "original question", depth, tuple(fused), tuple(fused), tuple(fused), tuple(final),
        {"dense": 0.1, "bm25": 0.1, "rrf": 0.01, "reranker": 0.2, "total": 0.41},
        len(fused), 10,
    )


def test_deep_passage_is_recorded_as_promoted_by_cross_encoder():
    noise = [item(i) for i in range(20)]
    gold = item(99, "gold evidence")
    second = item(100, "second evidence", "other.pdf", 2)
    run = make_run(50, [*noise, gold, second], [gold, second, *noise[:8]])

    record = evaluate_depth_run(example(), run)

    assert record["hops"][0]["pre_rerank_rrf_rank"] == 21
    assert record["hops"][0]["post_rerank_rank"] == 1
    assert record["oracle_candidate_coverage"]["full"] is True
    assert record["final_coverage"]["3"]["full"] is True
    assert record["first_relevant_passage_rank"] == 1


def test_partially_answerable_coverage_excludes_missing_hop():
    gold = item(1, "gold evidence")
    record = evaluate_depth_run(example(partial=True), make_run(20, [gold], [gold]))

    assert record["oracle_candidate_coverage"]["available_hops"] == 1
    assert record["oracle_candidate_coverage"]["full"] is True
    assert record["hops"][1]["evidence_available"] is False

    movement = classify_movement(record, record)
    assert movement["outcome"] == "unchanged"
    assert movement["transition"] == "1-hop_to_1-hop"


def test_question_movement_records_rescue_and_reranker_promotion():
    gold = item(1, "gold evidence")
    second = item(2, "second evidence", "other.pdf", 2)
    baseline = evaluate_depth_run(example(), make_run(20, [gold], [gold]))
    deeper = evaluate_depth_run(example(), make_run(50, [gold, second], [gold, second]))

    movement = classify_movement(baseline, deeper)

    assert movement["outcome"] == "improved"
    assert movement["transition"] == "1-hop_to_full-hop"
    assert movement["rescued_hop_indexes"] == [2]
    assert movement["rescued_hop_promotions"][0]["post_rerank_rank"] == 2


def test_question_movement_detects_deeper_pool_regression():
    gold = item(1, "gold evidence")
    baseline = evaluate_depth_run(example(partial=True), make_run(20, [gold], [gold]))
    deeper = evaluate_depth_run(example(partial=True), make_run(100, [gold], []))

    movement = classify_movement(baseline, deeper)

    assert movement["outcome"] == "degraded"
    assert movement["transition"] == "full-hop_to_partial-hop"
    assert movement["regressed_hop_indexes"] == [1]


def test_evaluator_rejects_wrong_question_and_preserves_exact_passage_matching():
    run = make_run(20, [item(1, "similar but not gold")], [item(1, "similar but not gold")])
    record = evaluate_depth_run(example(), run)
    assert record["final_coverage"]["10"]["recovered_hops"] == 0
    wrong = MultiHopExample("q", "different", "answerable", "x", example().hops, "verified")
    with pytest.raises(ValueError, match="does not correspond"):
        evaluate_depth_run(wrong, run)
