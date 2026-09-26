from pathlib import Path

from rag_research_assistant.experiments.phase10_candidate_depth_eval import (
    _regression_metrics,
)


class Evaluation:
    def __init__(self, rank, answerability="answerable"):
        self.answerability = answerability
        self.first_correct_rank = rank
        self.hit_at_1 = rank == 1 if rank else False
        self.hit_at_3 = bool(rank and rank <= 3)
        self.hit_at_5 = bool(rank and rank <= 5)
        self.expected_source_recall_at_1 = float(self.hit_at_1)
        self.expected_source_recall_at_3 = float(self.hit_at_3)
        self.expected_source_recall_at_5 = float(self.hit_at_5)


def test_frozen_regression_metrics_and_mfr_exclude_unanswerable():
    metrics = _regression_metrics([Evaluation(1), Evaluation(3), Evaluation(None, "unanswerable")])
    assert metrics["retrieval_scored_questions"] == 2
    assert metrics["unanswerable_questions"] == 1
    assert metrics["hit_at_1"] == 0.5
    assert metrics["hit_at_3"] == 1.0
    assert metrics["mean_first_correct_rank"] == 2.0


def test_phase10i2_is_not_imported_by_production_paths():
    root = Path(__file__).parents[1] / "src/rag_research_assistant"
    for relative in ("api.py", "application.py", "cli.py", "rag.py", "frameworks/langgraph_workflow.py"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "candidate_generation.depth" not in text
        assert "phase10_candidate_depth_eval" not in text
