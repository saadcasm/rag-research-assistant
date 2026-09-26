from pathlib import Path
from rag_research_assistant.experiments.phase10_coverage_reranking_eval import _regression_metrics


class E:
    answerability="answerable"; first_correct_rank=1; hit_at_1=True; hit_at_3=True; hit_at_5=True
    expected_source_recall_at_1=1.; expected_source_recall_at_3=1.; expected_source_recall_at_5=1.


def test_regression_metrics_remain_frozen_shape():
    row=_regression_metrics([E()]); assert row["hit_at_1"]==1.; assert row["mean_first_correct_rank"]==1.


def test_phase10j_not_imported_by_production():
    root=Path(__file__).parents[1]/"src/rag_research_assistant"
    for rel in ("api.py","application.py","cli.py","rag.py"):
        text=(root/rel).read_text(); assert "selectors" not in text; assert "phase10_coverage_reranking" not in text
