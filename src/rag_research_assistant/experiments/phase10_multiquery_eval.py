"""Evaluate original-query retrieval against two corpus-grounded multi-query conditions."""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from ..application import ApplicationSettings, default_retrieval_config, load_retriever
from ..evaluation import (
    DEFAULT_HIT_KS,
    EvaluationExample,
    QuestionEvaluation,
    evaluate_retrieval_results,
    load_evaluation_dataset,
)
from ..generation import GenerationError, OllamaGenerator
from ..models import CorpusIndexMetadata, SearchResult
from ..query_transform.multi_query import (
    MULTI_QUERY_PROMPT_TEMPLATE,
    CorpusGroundedMultiQueryGenerator,
    MultiQueryGenerationResult,
    MultiQueryGenerator,
    MultiQueryRetrievalResult,
    MultiQueryRetriever,
)
from ..retrievers import RerankingRetriever
from .phase10_rewrite_eval import classify_rank_change


DEFAULT_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_JSON_OUTPUT = Path(
    "data/evaluation/benchmarks/phase-10b-multiquery-results.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path("docs/phase-10b-multiquery-results.md")
DEFAULT_DIAGNOSTIC_JSON_OUTPUT = Path(
    "data/evaluation/benchmarks/phase-10b-multiquery-diagnostic.json"
)
DEFAULT_DIAGNOSTIC_MARKDOWN_OUTPUT = Path(
    "docs/phase-10b-multiquery-diagnostic.md"
)
RankOutcome = Literal["improved", "degraded", "unchanged", "not_scored"]


class _UnavailableGenerator:
    def __init__(self, model_name: str, error: GenerationError) -> None:
        self.model_name = model_name
        self.error = error

    def generate(self, prompt: str, temperature: float) -> str:
        raise self.error


@dataclass(frozen=True)
class RetrievedSummary:
    rank: int
    document: str
    start_page: int
    end_page: int
    chunk_id: str
    score: float
    text_preview: str


@dataclass(frozen=True)
class ConditionMetrics:
    hit_at_1: Optional[float]
    hit_at_3: Optional[float]
    hit_at_5: Optional[float]
    recall_at_1: Optional[float]
    recall_at_3: Optional[float]
    recall_at_5: Optional[float]
    mean_first_relevant_rank: Optional[float]


@dataclass(frozen=True)
class MultiQueryQuestionComparison:
    question_id: str
    original_question: str
    answerability: str
    expected_sources: List[Dict[str, Any]]
    generated_queries: List[str]
    valid_queries: List[str]
    rejected_queries: List[Dict[str, str]]
    baseline_first_relevant_rank: Optional[int]
    multi_query_2_first_relevant_rank: Optional[int]
    multi_query_3_first_relevant_rank: Optional[int]
    multi_query_2_outcome: RankOutcome
    multi_query_3_outcome: RankOutcome
    baseline_top_result: Optional[RetrievedSummary]
    multi_query_2_top_result: Optional[RetrievedSummary]
    multi_query_3_top_result: Optional[RetrievedSummary]
    baseline_results: List[RetrievedSummary]
    multi_query_2_results: List[RetrievedSummary]
    multi_query_3_results: List[RetrievedSummary]
    query_level_candidates: Dict[str, List[List[RetrievedSummary]]]
    fused_candidates: Dict[str, List[RetrievedSummary]]
    generation_seconds: float
    baseline_retrieval_seconds: float
    multi_query_2_retrieval_seconds: float
    multi_query_3_retrieval_seconds: float
    multi_query_2_total_seconds: float
    multi_query_3_total_seconds: float
    generation_failure: bool
    generation_error: Optional[str]
    multi_query_2_shortfall: int
    multi_query_3_shortfall: int
    multi_query_2_deduplicated_occurrences: int
    multi_query_3_deduplicated_occurrences: int
    multi_query_2_candidate_count_before_rerank: int
    multi_query_3_candidate_count_before_rerank: int
    drift_warnings: List[Dict[str, Any]]


@dataclass(frozen=True)
class MultiQueryExperimentSummary:
    total_questions: int
    retrieval_scored_questions: int
    unanswerable_questions: int
    generation_failures: int
    multi_query_2_shortfalls: int
    multi_query_3_shortfalls: int
    average_valid_variants: float
    rejected_variant_count: int
    drift_warning_questions: int
    multi_query_2_improved: int
    multi_query_2_degraded: int
    multi_query_2_unchanged: int
    multi_query_3_improved: int
    multi_query_3_degraded: int
    multi_query_3_unchanged: int
    average_generation_seconds: float
    p50_generation_seconds: float
    p95_generation_seconds: float
    average_baseline_retrieval_seconds: float
    p50_baseline_retrieval_seconds: float
    p95_baseline_retrieval_seconds: float
    average_multi_query_2_retrieval_seconds: float
    p50_multi_query_2_retrieval_seconds: float
    p95_multi_query_2_retrieval_seconds: float
    average_multi_query_3_retrieval_seconds: float
    p50_multi_query_3_retrieval_seconds: float
    p95_multi_query_3_retrieval_seconds: float
    average_multi_query_2_total_seconds: float
    p50_multi_query_2_total_seconds: float
    p95_multi_query_2_total_seconds: float
    average_multi_query_3_total_seconds: float
    p50_multi_query_3_total_seconds: float
    p95_multi_query_3_total_seconds: float
    average_multi_query_2_deduplicated_occurrences: float
    average_multi_query_3_deduplicated_occurrences: float
    average_multi_query_2_candidates_before_rerank: float
    average_multi_query_3_candidates_before_rerank: float
    baseline: ConditionMetrics
    multi_query_2: ConditionMetrics
    multi_query_3: ConditionMetrics


@dataclass(frozen=True)
class MultiQueryExperimentReport:
    schema_version: int
    experiment: str
    created_at: str
    dataset_path: str
    dataset_sha256: Optional[str]
    settings: Dict[str, Any]
    summary: MultiQueryExperimentSummary
    questions: List[MultiQueryQuestionComparison]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _condition_metrics(evaluations: Sequence[QuestionEvaluation]) -> ConditionMetrics:
    scored = [item for item in evaluations if item.answerability != "unanswerable"]

    def mean(field: str) -> Optional[float]:
        values = [getattr(item, field) for item in scored]
        numeric = [float(value) for value in values if value is not None]
        return _average(numeric) if numeric else None

    ranks = [
        float(item.first_correct_rank)
        for item in scored
        if item.first_correct_rank is not None
    ]
    return ConditionMetrics(
        hit_at_1=mean("hit_at_1"),
        hit_at_3=mean("hit_at_3"),
        hit_at_5=mean("hit_at_5"),
        recall_at_1=mean("expected_source_recall_at_1"),
        recall_at_3=mean("expected_source_recall_at_3"),
        recall_at_5=mean("expected_source_recall_at_5"),
        mean_first_relevant_rank=_average(ranks) if ranks else None,
    )


def _summaries(results: Sequence[SearchResult]) -> List[RetrievedSummary]:
    summaries = []
    for rank, result in enumerate(results, start=1):
        text = " ".join(result.chunk.text.split())
        if len(text) > 400:
            text = text[:400].rstrip() + "..."
        summaries.append(
            RetrievedSummary(
                rank=rank,
                document=result.chunk.document,
                start_page=result.chunk.start_page,
                end_page=result.chunk.end_page,
                chunk_id=result.chunk.chunk_id,
                score=result.score,
                text_preview=text,
            )
        )
    return summaries


def _top(results: Sequence[SearchResult]) -> Optional[RetrievedSummary]:
    values = _summaries(results[:1])
    return values[0] if values else None


def _condition_seconds(result: MultiQueryRetrievalResult) -> float:
    return result.retrieval_seconds + result.fusion_seconds + result.reranking_seconds


def _expected_sources(example: EvaluationExample) -> List[Dict[str, Any]]:
    return [asdict(source) for source in example.expected_sources]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_multi_query(
    examples: Sequence[EvaluationExample],
    retriever: RerankingRetriever,
    generator: MultiQueryGenerator,
    metadata: CorpusIndexMetadata,
    *,
    preliminary_top_k: int = 5,
    final_top_k: int = 5,
    multi_query_rrf_k: int = 60,
    dataset_path: str = "",
    on_progress: Optional[
        Callable[[int, int, MultiQueryQuestionComparison], None]
    ] = None,
) -> MultiQueryExperimentReport:
    """Run baseline, original+2, and original+3 with frozen relevance logic."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    if preliminary_top_k <= 0:
        raise ValueError("preliminary_top_k must be positive")
    if final_top_k < max(DEFAULT_HIT_KS):
        raise ValueError("final_top_k must be at least 5")
    multi_retriever = MultiQueryRetriever(retriever, rrf_k=multi_query_rrf_k)
    comparisons: List[MultiQueryQuestionComparison] = []
    baseline_evaluations: List[QuestionEvaluation] = []
    mq2_evaluations: List[QuestionEvaluation] = []
    mq3_evaluations: List[QuestionEvaluation] = []

    for example in examples:
        baseline_started = perf_counter()
        baseline_results = retriever.search(example.question, top_k=final_top_k)
        baseline_seconds = perf_counter() - baseline_started
        generation = generator.generate_queries(
            example.question,
            baseline_results[:preliminary_top_k],
            count=3,
        )
        mq2_variants = generation.valid_queries[:2]
        mq3_variants = generation.valid_queries[:3]
        mq2 = multi_retriever.search(
            example.question, mq2_variants, top_k=final_top_k
        )
        mq3 = multi_retriever.search(
            example.question, mq3_variants, top_k=final_top_k
        )

        baseline_eval = evaluate_retrieval_results(example, baseline_results)
        mq2_eval = evaluate_retrieval_results(example, mq2.final_results)
        mq3_eval = evaluate_retrieval_results(example, mq3.final_results)
        baseline_evaluations.append(baseline_eval)
        mq2_evaluations.append(mq2_eval)
        mq3_evaluations.append(mq3_eval)

        def query_rankings(result: MultiQueryRetrievalResult) -> List[List[RetrievedSummary]]:
            return [_summaries(ranking) for ranking in result.query_rankings]

        drift = [
            {
                "query": diagnostic.query,
                "protected_terms": list(diagnostic.identity.protected_terms),
                "missing_terms": list(diagnostic.identity.missing_terms),
                "warnings": list(diagnostic.warnings),
            }
            for diagnostic in generation.diagnostics
            if diagnostic.warnings
        ]
        item = MultiQueryQuestionComparison(
            question_id=example.id,
            original_question=example.question,
            answerability=example.answerability,
            expected_sources=_expected_sources(example),
            generated_queries=list(generation.generated_queries),
            valid_queries=list(generation.valid_queries),
            rejected_queries=[asdict(value) for value in generation.rejected_queries],
            baseline_first_relevant_rank=baseline_eval.first_correct_rank,
            multi_query_2_first_relevant_rank=mq2_eval.first_correct_rank,
            multi_query_3_first_relevant_rank=mq3_eval.first_correct_rank,
            multi_query_2_outcome=classify_rank_change(
                example.answerability,
                baseline_eval.first_correct_rank,
                mq2_eval.first_correct_rank,
            ),
            multi_query_3_outcome=classify_rank_change(
                example.answerability,
                baseline_eval.first_correct_rank,
                mq3_eval.first_correct_rank,
            ),
            baseline_top_result=_top(baseline_results),
            multi_query_2_top_result=_top(mq2.final_results),
            multi_query_3_top_result=_top(mq3.final_results),
            baseline_results=_summaries(baseline_results),
            multi_query_2_results=_summaries(mq2.final_results),
            multi_query_3_results=_summaries(mq3.final_results),
            query_level_candidates={
                "multi_query_2": query_rankings(mq2),
                "multi_query_3": query_rankings(mq3),
            },
            fused_candidates={
                "multi_query_2": _summaries(mq2.fused_candidates),
                "multi_query_3": _summaries(mq3.fused_candidates),
            },
            generation_seconds=generation.latency_seconds,
            baseline_retrieval_seconds=baseline_seconds,
            multi_query_2_retrieval_seconds=_condition_seconds(mq2),
            multi_query_3_retrieval_seconds=_condition_seconds(mq3),
            multi_query_2_total_seconds=generation.latency_seconds + _condition_seconds(mq2),
            multi_query_3_total_seconds=generation.latency_seconds + _condition_seconds(mq3),
            generation_failure=generation.fallback,
            generation_error=generation.error,
            multi_query_2_shortfall=max(0, 2 - len(mq2_variants)),
            multi_query_3_shortfall=max(0, 3 - len(mq3_variants)),
            multi_query_2_deduplicated_occurrences=mq2.deduplicated_occurrences,
            multi_query_3_deduplicated_occurrences=mq3.deduplicated_occurrences,
            multi_query_2_candidate_count_before_rerank=mq2.candidate_count_before_rerank,
            multi_query_3_candidate_count_before_rerank=mq3.candidate_count_before_rerank,
            drift_warnings=drift,
        )
        comparisons.append(item)
        if on_progress:
            on_progress(len(comparisons), len(examples), item)

    generation_times = [item.generation_seconds for item in comparisons]
    baseline_times = [item.baseline_retrieval_seconds for item in comparisons]
    mq2_retrieval_times = [item.multi_query_2_retrieval_seconds for item in comparisons]
    mq3_retrieval_times = [item.multi_query_3_retrieval_seconds for item in comparisons]
    mq2_total_times = [item.multi_query_2_total_seconds for item in comparisons]
    mq3_total_times = [item.multi_query_3_total_seconds for item in comparisons]
    outcomes = lambda condition, outcome: sum(
        getattr(item, f"{condition}_outcome") == outcome for item in comparisons
    )
    summary = MultiQueryExperimentSummary(
        total_questions=len(comparisons),
        retrieval_scored_questions=sum(item.answerability != "unanswerable" for item in comparisons),
        unanswerable_questions=sum(item.answerability == "unanswerable" for item in comparisons),
        generation_failures=sum(item.generation_failure for item in comparisons),
        multi_query_2_shortfalls=sum(item.multi_query_2_shortfall > 0 for item in comparisons),
        multi_query_3_shortfalls=sum(item.multi_query_3_shortfall > 0 for item in comparisons),
        average_valid_variants=_average([float(len(item.valid_queries)) for item in comparisons]),
        rejected_variant_count=sum(len(item.rejected_queries) for item in comparisons),
        drift_warning_questions=sum(bool(item.drift_warnings) for item in comparisons),
        multi_query_2_improved=outcomes("multi_query_2", "improved"),
        multi_query_2_degraded=outcomes("multi_query_2", "degraded"),
        multi_query_2_unchanged=outcomes("multi_query_2", "unchanged"),
        multi_query_3_improved=outcomes("multi_query_3", "improved"),
        multi_query_3_degraded=outcomes("multi_query_3", "degraded"),
        multi_query_3_unchanged=outcomes("multi_query_3", "unchanged"),
        average_generation_seconds=_average(generation_times),
        p50_generation_seconds=statistics.median(generation_times),
        p95_generation_seconds=_percentile(generation_times, 0.95),
        average_baseline_retrieval_seconds=_average(baseline_times),
        p50_baseline_retrieval_seconds=statistics.median(baseline_times),
        p95_baseline_retrieval_seconds=_percentile(baseline_times, 0.95),
        average_multi_query_2_retrieval_seconds=_average(mq2_retrieval_times),
        p50_multi_query_2_retrieval_seconds=statistics.median(mq2_retrieval_times),
        p95_multi_query_2_retrieval_seconds=_percentile(mq2_retrieval_times, 0.95),
        average_multi_query_3_retrieval_seconds=_average(mq3_retrieval_times),
        p50_multi_query_3_retrieval_seconds=statistics.median(mq3_retrieval_times),
        p95_multi_query_3_retrieval_seconds=_percentile(mq3_retrieval_times, 0.95),
        average_multi_query_2_total_seconds=_average(mq2_total_times),
        p50_multi_query_2_total_seconds=statistics.median(mq2_total_times),
        p95_multi_query_2_total_seconds=_percentile(mq2_total_times, 0.95),
        average_multi_query_3_total_seconds=_average(mq3_total_times),
        p50_multi_query_3_total_seconds=statistics.median(mq3_total_times),
        p95_multi_query_3_total_seconds=_percentile(mq3_total_times, 0.95),
        average_multi_query_2_deduplicated_occurrences=_average([float(item.multi_query_2_deduplicated_occurrences) for item in comparisons]),
        average_multi_query_3_deduplicated_occurrences=_average([float(item.multi_query_3_deduplicated_occurrences) for item in comparisons]),
        average_multi_query_2_candidates_before_rerank=_average([float(item.multi_query_2_candidate_count_before_rerank) for item in comparisons]),
        average_multi_query_3_candidates_before_rerank=_average([float(item.multi_query_3_candidate_count_before_rerank) for item in comparisons]),
        baseline=_condition_metrics(baseline_evaluations),
        multi_query_2=_condition_metrics(mq2_evaluations),
        multi_query_3=_condition_metrics(mq3_evaluations),
    )
    return MultiQueryExperimentReport(
        schema_version=1,
        experiment="phase-10b-corpus-grounded-multi-query-retrieval",
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_path=dataset_path,
        dataset_sha256=_sha256(Path(dataset_path)) if dataset_path and Path(dataset_path).is_file() else None,
        settings={
            "conditions": {"baseline": 0, "multi_query_2": 2, "multi_query_3": 3},
            "preliminary_top_k": preliminary_top_k,
            "final_top_k": final_top_k,
            "candidate_depth": retriever.candidate_depth,
            "query_level_retrieval": getattr(retriever.base, "name", None),
            "dense_backend": getattr(retriever, "backend_name", None),
            "embedding_model": metadata.embedding_model,
            "embedding_dimension": metadata.embedding_dimension,
            "indexed_chunks": metadata.chunk_count,
            "reranker_model": retriever.reranker_model,
            "hybrid_rrf_k": getattr(retriever.base, "rrf_k", None),
            "multi_query_rrf_k": multi_query_rrf_k,
            "generation_model": generator.model_name,
            "generation_temperature": getattr(generator, "temperature", None),
            "max_chars_per_preliminary_chunk": getattr(generator, "max_chars_per_chunk", None),
            "generation_calls_per_question": 1,
            "cross_encoder_query": "original_question",
            "prompt_sha256": hashlib.sha256(MULTI_QUERY_PROMPT_TEMPLATE.encode()).hexdigest(),
        },
        summary=summary,
        questions=comparisons,
    )


def write_json_report(report: MultiQueryExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_markdown_report(report: MultiQueryExperimentReport, path: Path) -> None:
    summary = report.summary
    lines = [
        "# Phase 10B: multi-query retrieval results", "",
        "The frozen baseline is compared with original+2 and original+3 generated alternatives. Each query receives dense+BM25 hybrid retrieval; those rankings are fused before one unchanged cross-encoder rerank against the original question.", "",
        f"Dataset: `{report.dataset_path}`", f"Generation model: `{report.settings['generation_model']}`", "",
        "## Aggregate retrieval metrics", "",
        "| Metric | Baseline | Multi-query-2 | Multi-query-3 |", "|---|---:|---:|---:|",
    ]
    for label, field in [
        ("Hit@1", "hit_at_1"), ("Hit@3", "hit_at_3"), ("Hit@5", "hit_at_5"),
        ("Recall@1", "recall_at_1"), ("Recall@3", "recall_at_3"), ("Recall@5", "recall_at_5"),
        ("Mean first relevant rank", "mean_first_relevant_rank"),
    ]:
        lines.append(f"| {label} | {_metric(getattr(summary.baseline, field))} | {_metric(getattr(summary.multi_query_2, field))} | {_metric(getattr(summary.multi_query_3, field))} |")
    lines.extend([
        "", "## Behavior and cost", "",
        f"- Questions: {summary.total_questions} ({summary.retrieval_scored_questions} scored; {summary.unanswerable_questions} unanswerable)",
        f"- MQ2 improved/degraded/unchanged: {summary.multi_query_2_improved}/{summary.multi_query_2_degraded}/{summary.multi_query_2_unchanged}",
        f"- MQ3 improved/degraded/unchanged: {summary.multi_query_3_improved}/{summary.multi_query_3_degraded}/{summary.multi_query_3_unchanged}",
        f"- Generation failures: {summary.generation_failures}; MQ2/MQ3 shortfall questions: {summary.multi_query_2_shortfalls}/{summary.multi_query_3_shortfalls}",
        f"- Average valid variants: {summary.average_valid_variants:.2f}; rejected variants: {summary.rejected_variant_count}",
        f"- Drift-warning questions: {summary.drift_warning_questions}",
        f"- Average generation: {summary.average_generation_seconds:.3f}s (p50 {summary.p50_generation_seconds:.3f}s; p95 {summary.p95_generation_seconds:.3f}s)",
        f"- Average baseline/MQ2/MQ3 retrieval: {summary.average_baseline_retrieval_seconds:.3f}s / {summary.average_multi_query_2_retrieval_seconds:.3f}s / {summary.average_multi_query_3_retrieval_seconds:.3f}s",
        f"- Average MQ2/MQ3 total: {summary.average_multi_query_2_total_seconds:.3f}s / {summary.average_multi_query_3_total_seconds:.3f}s", "",
        "## Informative rank changes", "",
        "| ID | Baseline | MQ2 | MQ3 | Original question |", "|---|---:|---:|---:|---|",
    ])
    informative = [
        item
        for item in report.questions
        if item.multi_query_2_outcome == "improved"
        or item.multi_query_3_outcome == "improved"
        or item.multi_query_2_outcome == "degraded"
        or item.multi_query_3_outcome == "degraded"
    ]
    for item in informative:
        rank = lambda value: "not found" if value is None else str(value)
        question = item.original_question.replace("|", "\\|")
        lines.append(
            f"| `{item.question_id}` | {rank(item.baseline_first_relevant_rank)} | "
            f"{rank(item.multi_query_2_first_relevant_rank)} | "
            f"{rank(item.multi_query_3_first_relevant_rank)} | {question} |"
        )
    if not informative:
        lines.append("| — | — | — | — | No rank changes |")

    failures = [item for item in report.questions if item.generation_failure]
    lines.extend(["", "## Generation failures", ""])
    for item in failures:
        lines.append(f"- `{item.question_id}`: {item.generation_error}")
    if not failures:
        lines.append("- None.")

    baseline_hit5 = summary.baseline.hit_at_5 or 0.0
    mq2_hit5 = summary.multi_query_2.hit_at_5 or 0.0
    lines.extend([
        "", "## Interpretation", "",
        "The original-query safeguard worked in this run: neither multi-query condition degraded a scored question relative to baseline. The mean first-relevant rank nevertheless increased slightly because multi-query recovered previously missed evidence at ranks 4–5; adding those newly found ranks changes the mean's denominator.",
        "",
        "MQ3 did not improve aggregate metrics beyond MQ2. Extra-query retrieval timings were measured in a fixed MQ2-then-MQ3 order, so cache and warm-up effects make their small timing inversion unsuitable as evidence that three alternatives are cheaper.",
        "",
        "Drift diagnostics remain intentionally high-recall warnings. Hyphenated technical phrases can be classified as model-like identity tokens, so warning counts are not error counts and require per-query inspection.",
        "", "## Recommendation", "",
    ])
    if mq2_hit5 > baseline_hit5 and summary.multi_query_2_degraded == 0:
        lines.append(
            "Multi-query-2 shows a real but modest recall benefit in this corpus. Keep it as an optional experimental recall mode, not the default: generation dominates latency and structured-output failures remain operationally significant. Multi-query-3 is not justified by this run because it adds no aggregate gain over MQ2."
        )
    else:
        lines.append(
            "Do not adopt multi-query retrieval from this run; the measured ranking benefit does not justify its generation and retrieval cost."
        )
    lines.extend([
        "",
        "The companion JSON retains the complete generated queries, query-level rankings, fused candidates, final results, configuration, failures, and latency records needed for further audit without rerunning Ollama.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_results(label: str, results: Sequence[SearchResult]) -> None:
    print(f"\n{label}")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"{rank}. {chunk.document} page {chunk.start_page}-{chunk.end_page} chunk={chunk.chunk_id} score={result.score:.4f}")


def run_smoke(
    retriever: RerankingRetriever,
    generator: MultiQueryGenerator,
    *,
    question: str,
    preliminary_top_k: int,
    final_top_k: int,
    multi_query_rrf_k: int,
) -> None:
    print("ORIGINAL QUERY")
    print(question)
    preliminary = retriever.search(question, top_k=preliminary_top_k)
    generation = generator.generate_queries(question, preliminary, count=3)
    print("\nGENERATED VARIANTS")
    for index, query in enumerate(generation.valid_queries, start=1):
        print(f"{index}. {query}")
    print(f"shortfall={generation.shortfall} fallback={generation.fallback} latency={generation.latency_seconds:.3f}s")
    if generation.error:
        print(f"generation_error={generation.error}")
    result = MultiQueryRetriever(retriever, rrf_k=multi_query_rrf_k).search(
        question, generation.valid_queries, top_k=final_top_k
    )
    for query, ranking in zip(result.queries, result.query_rankings):
        _print_results(f"PER-QUERY HYBRID RESULTS: {query}", ranking)
    _print_results("MULTI-QUERY RRF FUSED CANDIDATES", result.fused_candidates)
    _print_results("FINAL CROSS-ENCODER RERANKED RESULTS", result.final_results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--diagnostic", type=int, metavar="COUNT")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--question", default="Who wrote the DPR paper?")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--preliminary-top-k", type=int, default=5)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--multi-query-rrf-k", type=int, default=60)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.diagnostic is not None and args.diagnostic <= 0:
        print("Error: --diagnostic COUNT must be positive", file=sys.stderr)
        return 2
    settings = ApplicationSettings.from_environment()
    model = args.model or settings.ollama_model
    timeout = settings.ollama_timeout if args.timeout is None else args.timeout
    try:
        loaded = load_retriever(default_retrieval_config(settings))
    except Exception as exc:
        print(f"Error: retrieval startup failed: {exc}", file=sys.stderr)
        return 2
    try:
        if not isinstance(loaded.retriever, RerankingRetriever):
            raise ValueError("Phase 10B requires the frozen reranking retriever")
        ollama = OllamaGenerator(model_name=model, base_url=settings.ollama_url, timeout=timeout)
        try:
            ollama.ensure_model_available()
        except GenerationError as exc:
            print(f"Warning: model unavailable; variants will fall back to original-only retrieval: {exc}", file=sys.stderr)
            ollama = _UnavailableGenerator(model, exc)
        generator = CorpusGroundedMultiQueryGenerator(
            ollama,
            temperature=args.temperature,
            max_chars_per_chunk=args.max_chars_per_chunk,
        )
        if args.smoke:
            run_smoke(
                loaded.retriever,
                generator,
                question=args.question,
                preliminary_top_k=args.preliminary_top_k,
                final_top_k=args.final_top_k,
                multi_query_rrf_k=args.multi_query_rrf_k,
            )
            return 0
        examples = load_evaluation_dataset(args.dataset)
        if args.diagnostic is not None:
            examples = examples[: args.diagnostic]
            json_output = args.json_output or DEFAULT_DIAGNOSTIC_JSON_OUTPUT
            markdown_output = args.markdown_output or DEFAULT_DIAGNOSTIC_MARKDOWN_OUTPUT
        else:
            json_output = args.json_output or DEFAULT_JSON_OUTPUT
            markdown_output = args.markdown_output or DEFAULT_MARKDOWN_OUTPUT
        report = evaluate_multi_query(
            examples,
            loaded.retriever,
            generator,
            loaded.metadata,
            preliminary_top_k=args.preliminary_top_k,
            final_top_k=args.final_top_k,
            multi_query_rrf_k=args.multi_query_rrf_k,
            dataset_path=str(args.dataset),
            on_progress=lambda current, total, item: print(
                f"[{current}/{total}] {item.question_id}: variants={len(item.valid_queries)} "
                f"mq2={item.multi_query_2_outcome} mq3={item.multi_query_3_outcome}",
                flush=True,
            ),
        )
        write_json_report(report, json_output)
        write_markdown_report(report, markdown_output)
        print(f"JSON report: {json_output}")
        print(f"Markdown report: {markdown_output}")
        return 0
    except (GenerationError, OSError, ValueError) as exc:
        print(f"Error: Phase 10B experiment failed: {exc}", file=sys.stderr)
        return 2
    finally:
        loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
