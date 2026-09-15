from rag_research_assistant.experiments.phase10_routing_analysis import analyze_phase10b


def ranked(document: str, chunk: str, score: float):
    return {
        "rank": 1,
        "document": document,
        "start_page": 1,
        "end_page": 1,
        "chunk_id": chunk,
        "score": score,
        "text_preview": "DPR dense passage retrieval",
    }


def question(question_id: str, outcome: str, baseline_document: str):
    right = ranked("right.pdf", "right", 5.0)
    baseline = ranked(baseline_document, "baseline", 1.0)
    return {
        "question_id": question_id,
        "original_question": "How does DPR retrieve passages?",
        "answerability": "answerable",
        "expected_sources": [
            {"document": "right.pdf", "page_number": 1, "chunk_id": None}
        ],
        "multi_query_2_outcome": outcome,
        "baseline_first_relevant_rank": None if outcome == "improved" else 1,
        "multi_query_2_first_relevant_rank": 1,
        "baseline_results": [baseline if outcome == "improved" else right],
        "multi_query_2_results": [right],
        "query_level_candidates": {"multi_query_2": [[baseline]]},
    }


def test_offline_analysis_joins_labels_sweeps_rules_and_preserves_reference_metrics() -> None:
    source = {
        "dataset_path": "frozen.jsonl",
        "dataset_sha256": "abc",
        "summary": {
            "average_baseline_retrieval_seconds": 0.2,
            "average_multi_query_2_total_seconds": 6.4,
        },
        "questions": [
            question("helpful", "improved", "wrong.pdf"),
            question("neutral", "unchanged", "right.pdf"),
            {
                **question("unanswerable", "not_scored", "wrong.pdf"),
                "answerability": "unanswerable",
            },
        ],
    }

    report = analyze_phase10b(source, source_path="phase10b.json")

    assert report["summary"]["total_scored_questions"] == 2
    assert report["summary"]["mq2_helpful_questions"] == 1
    assert report["summary"]["mq2_unchanged_questions"] == 1
    assert report["summary"]["always_baseline"]["escalation_rate"] == 0.0
    assert report["summary"]["always_mq2"]["rescue_recall"] == 1.0
    assert report["summary"]["always_mq2"]["routing_precision"] == 0.5
    assert report["candidate_threshold_rules"]
    assert report["per_question"][0]["mq2_outcome"] == "improved"
    assert report["unavailable_signals"][0]["signal"] == "dense_bm25_rank_agreement"
