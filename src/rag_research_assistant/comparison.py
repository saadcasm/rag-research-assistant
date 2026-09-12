"""Compare evaluation reports without hiding per-question rank changes."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .evaluation import EvaluationReport


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    dense_backend: str | None
    hit_at_1: float | None
    hit_at_3: float | None
    hit_at_5: float | None
    mean_first_correct_rank: float | None
    expected_source_recall_at_1: float | None
    expected_source_recall_at_3: float | None
    expected_source_recall_at_5: float | None


@dataclass(frozen=True)
class RankChange:
    question_id: str
    from_rank: int | None
    to_rank: int | None


@dataclass(frozen=True)
class PairwiseComparison:
    baseline: str
    contender: str
    improved: List[RankChange]
    degraded: List[RankChange]


@dataclass(frozen=True)
class ComparisonReport:
    strategies: List[StrategyMetrics]
    comparisons: List[PairwiseComparison]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compare_pair(
    baseline: EvaluationReport, contender: EvaluationReport
) -> PairwiseComparison:
    baseline_questions = {question.id: question for question in baseline.questions}
    contender_questions = {question.id: question for question in contender.questions}
    if baseline_questions.keys() != contender_questions.keys():
        raise ValueError("comparison reports must contain the same question IDs")

    improved: List[RankChange] = []
    degraded: List[RankChange] = []
    for question_id, before in baseline_questions.items():
        if before.answerability == "unanswerable":
            continue
        after = contender_questions[question_id]
        before_rank = before.first_correct_rank
        after_rank = after.first_correct_rank
        change = RankChange(question_id, before_rank, after_rank)
        if after_rank is not None and (before_rank is None or after_rank < before_rank):
            improved.append(change)
        elif before_rank is not None and (after_rank is None or after_rank > before_rank):
            degraded.append(change)
    return PairwiseComparison(
        baseline.retrieval_strategy,
        contender.retrieval_strategy,
        improved,
        degraded,
    )


def compare_reports(reports: Sequence[EvaluationReport]) -> ComparisonReport:
    """Create aggregate and adjacent per-question comparisons."""

    if len(reports) < 2:
        raise ValueError("at least two reports are required for comparison")
    metrics = [
        StrategyMetrics(
            report.retrieval_strategy,
            report.dense_backend,
            report.summary.hit_at_1,
            report.summary.hit_at_3,
            report.summary.hit_at_5,
            report.summary.mean_first_correct_rank,
            report.summary.expected_source_recall_at_1,
            report.summary.expected_source_recall_at_3,
            report.summary.expected_source_recall_at_5,
        )
        for report in reports
    ]
    reports_by_strategy = {report.retrieval_strategy: report for report in reports}
    if len(reports_by_strategy) != len(reports):
        raise ValueError("retrieval strategy names must be unique")
    if {"dense", "hybrid", "hybrid+rerank"} <= reports_by_strategy.keys():
        comparisons = [
            _compare_pair(reports_by_strategy["dense"], reports_by_strategy["hybrid"]),
            _compare_pair(
                reports_by_strategy["hybrid"], reports_by_strategy["hybrid+rerank"]
            ),
        ]
    else:
        comparisons = [
            _compare_pair(reports[index - 1], reports[index])
            for index in range(1, len(reports))
        ]
    return ComparisonReport(metrics, comparisons)


def write_comparison_report(report: ComparisonReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(report.to_dict(), output, indent=2, ensure_ascii=False)
        output.write("\n")
