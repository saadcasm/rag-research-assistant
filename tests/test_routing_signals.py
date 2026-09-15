import pytest

from rag_research_assistant.routing.signals import (
    RankedCandidate,
    RoutingExample,
    ThresholdRule,
    evaluate_routing_decisions,
    evaluate_threshold_rule,
    extract_baseline_signals,
    join_mq2_outcomes,
)


def candidate(
    chunk_id: str,
    score: float,
    document: str,
    text: str,
    page: int = 1,
) -> RankedCandidate:
    return RankedCandidate(score, document, page, page, chunk_id, text)


def test_signal_extraction_calculates_gaps_documents_overlap_and_coverage() -> None:
    reranked = [
        candidate("a1", 8.0, "a.pdf", "DPR passage encoder"),
        candidate("b1", 5.0, "b.pdf", "unrelated text"),
        candidate("a2", 2.0, "a.pdf", "dense retrieval"),
    ]
    hybrid = [
        candidate("b1", 0.03, "b.pdf", "unrelated text"),
        candidate("a1", 0.02, "a.pdf", "DPR passage encoder"),
        candidate("c1", 0.01, "c.pdf", "other"),
    ]

    signals = extract_baseline_signals(
        "How does DPR represent a passage?", reranked, hybrid
    )

    assert signals.top_reranker_score == 8.0
    assert signals.reranker_gap_1_2 == 3.0
    assert signals.reranker_gap_1_3 == 6.0
    assert signals.reranker_score_spread == 6.0
    assert signals.unique_documents_top_k == 2
    assert signals.dominant_document_share == pytest.approx(2 / 3)
    assert signals.documents_with_multiple_chunks == 1
    assert signals.query_term_coverage_top_1 == pytest.approx(2 / 3)
    assert signals.query_term_coverage_top_k == pytest.approx(2 / 3)
    assert signals.protected_term_count == 1
    assert signals.protected_term_coverage_top_k == 1.0
    assert signals.hybrid_reranked_chunk_overlap_at_5 == pytest.approx(0.5)
    assert signals.hybrid_reranked_document_overlap_at_5 == pytest.approx(2 / 3)
    assert signals.hybrid_top_chunk_retained is True
    assert signals.hybrid_top_document_retained is True
    assert signals.hybrid_mean_rank_shift == 1.0


def test_empty_rankings_have_explicit_neutral_or_missing_values() -> None:
    signals = extract_baseline_signals("retrieval question", [], [])

    assert signals.top_reranker_score is None
    assert signals.reranker_gap_1_2 is None
    assert signals.unique_documents_top_k == 0
    assert signals.query_term_coverage_top_k == 0.0
    assert signals.hybrid_reranked_chunk_overlap_at_5 == 1.0


def test_label_join_rejects_question_id_mismatches() -> None:
    signals = extract_baseline_signals("question", [], [])
    assert join_mq2_outcomes({"q1": signals}, {"q1": "improved"})["q1"][1] == "improved"
    with pytest.raises(ValueError, match="IDs differ"):
        join_mq2_outcomes({"q1": signals}, {"q2": "unchanged"})


def routing_example(question_id: str, outcome: str, signal_value: float) -> RoutingExample:
    signals = extract_baseline_signals(
        "retrieval",
        [candidate(question_id, signal_value, f"{question_id}.pdf", "retrieval")],
        [candidate(question_id, 0.1, f"{question_id}.pdf", "retrieval")],
    )
    expected = ({"document": "right.pdf", "page_number": 1, "chunk_id": None},)
    wrong = ({"document": "wrong.pdf", "start_page": 1, "end_page": 1, "chunk_id": "wrong"},)
    right = ({"document": "right.pdf", "start_page": 1, "end_page": 1, "chunk_id": "right"},)
    return RoutingExample(
        question_id,
        outcome,
        None if outcome == "improved" else 1,
        1,
        expected,
        wrong if outcome == "improved" else right,
        right,
        signals,
    )


def test_routing_metrics_cover_rescues_precision_cost_and_retrieval() -> None:
    examples = [
        routing_example("q1", "improved", 1.0),
        routing_example("q2", "improved", 2.0),
        routing_example("q3", "unchanged", 9.0),
        routing_example("q4", "unchanged", 10.0),
    ]
    metrics = evaluate_routing_decisions(
        examples,
        [True, False, True, False],
        baseline_latency_seconds=0.2,
        mq2_latency_seconds=6.2,
    )

    assert metrics.escalated_questions == 2
    assert metrics.escalation_rate == 0.5
    assert metrics.rescued_questions == 1
    assert metrics.rescue_recall == 0.5
    assert metrics.routing_precision == 0.5
    assert metrics.missed_rescues == 1
    assert metrics.unnecessary_escalations == 1
    assert metrics.estimated_average_latency_seconds == pytest.approx(3.2)
    assert metrics.retrieval.hit_at_1 == 0.75
    assert metrics.retrieval.recall_at_5 == 0.75


def test_candidate_threshold_rule_routes_in_both_directions() -> None:
    examples = [
        routing_example("q1", "improved", 1.0),
        routing_example("q2", "unchanged", 10.0),
    ]
    low = ThresholdRule("top_reranker_score", "at_or_below", 5.0)
    high = ThresholdRule("top_reranker_score", "at_or_above", 5.0)

    low_metrics = evaluate_threshold_rule(
        examples, low, baseline_latency_seconds=0.1, mq2_latency_seconds=6.0
    )
    high_metrics = evaluate_threshold_rule(
        examples, high, baseline_latency_seconds=0.1, mq2_latency_seconds=6.0
    )

    assert low_metrics.rescue_recall == 1.0
    assert low_metrics.routing_precision == 1.0
    assert high_metrics.rescue_recall == 0.0
    with pytest.raises(ValueError, match="unknown threshold direction"):
        ThresholdRule("top_reranker_score", "sideways", 5.0).routes(examples[0].signals)
