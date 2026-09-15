"""Evaluate baseline retrieval against corpus-grounded single-query rewriting."""

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

from ..application import (
    ApplicationSettings,
    default_retrieval_config,
    load_retriever,
)
from ..evaluation import (
    DEFAULT_HIT_KS,
    EvaluationExample,
    QuestionEvaluation,
    evaluate_retrieval_results,
    load_evaluation_dataset,
)
from ..generation import GenerationError, OllamaGenerator
from ..models import CorpusIndexMetadata, SearchResult
from ..query_transform.rewriting import (
    CorpusGroundedLLMRewriter,
    QueryRewriter,
    REWRITE_PROMPT_TEMPLATE,
    RewriteResult,
)
from ..retrievers import Retriever


DEFAULT_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_JSON_OUTPUT = Path(
    "data/evaluation/benchmarks/phase-10a-rewrite-results.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path("docs/phase-10a-query-rewriting-results.md")
RankOutcome = Literal["improved", "degraded", "unchanged", "not_scored"]


class _UnavailableGenerator:
    """Turn one startup dependency error into fast, explicit rewrite fallbacks."""

    def __init__(self, model_name: str, error: GenerationError) -> None:
        self.model_name = model_name
        self.error = error

    def generate(self, prompt: str, temperature: float) -> str:
        raise self.error


@dataclass(frozen=True)
class RetrievedSummary:
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
class RewriteQuestionComparison:
    question_id: str
    original_question: str
    answerability: str
    expected_sources: List[Dict[str, Any]]
    rewritten_query: str
    changed: bool
    baseline_first_relevant_rank: Optional[int]
    rewritten_first_relevant_rank: Optional[int]
    outcome: RankOutcome
    baseline_top_result: Optional[RetrievedSummary]
    preliminary_results: List[RetrievedSummary]
    rewritten_top_result: Optional[RetrievedSummary]
    baseline_retrieval_seconds: float
    preliminary_retrieval_seconds: float
    rewrite_seconds: float
    final_retrieval_seconds: float
    rewrite_condition_total_seconds: float
    fallback: bool
    rewrite_error: Optional[str]
    identity_protected_terms: List[str]
    identity_missing_terms: List[str]


@dataclass(frozen=True)
class RewriteExperimentSummary:
    total_questions: int
    retrieval_scored_questions: int
    unanswerable_questions: int
    queries_changed: int
    changed_percentage: float
    queries_unchanged: int
    improved_questions: int
    degraded_questions: int
    unchanged_questions: int
    rewrite_failures: int
    invalid_or_empty_responses: int
    fallbacks: int
    identity_warning_questions: int
    average_rewrite_seconds: float
    p50_rewrite_seconds: float
    p95_rewrite_seconds: float
    average_baseline_retrieval_seconds: float
    average_rewrite_condition_total_seconds: float
    baseline: ConditionMetrics
    rewritten: ConditionMetrics


@dataclass(frozen=True)
class RewriteExperimentReport:
    schema_version: int
    experiment: str
    created_at: str
    dataset_path: str
    dataset_sha256: Optional[str]
    settings: Dict[str, Any]
    summary: RewriteExperimentSummary
    questions: List[RewriteQuestionComparison]

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


def classify_rank_change(
    answerability: str,
    baseline_rank: Optional[int],
    rewritten_rank: Optional[int],
) -> RankOutcome:
    """Classify rank movement while excluding unanswerable questions."""

    if answerability == "unanswerable":
        return "not_scored"
    if rewritten_rank is not None and (
        baseline_rank is None or rewritten_rank < baseline_rank
    ):
        return "improved"
    if baseline_rank is not None and (
        rewritten_rank is None or rewritten_rank > baseline_rank
    ):
        return "degraded"
    return "unchanged"


def _top_result(results: Sequence[SearchResult]) -> Optional[RetrievedSummary]:
    if not results:
        return None
    result = results[0]
    return _retrieved_summary(results[0])


def _retrieved_summary(result: SearchResult) -> RetrievedSummary:
    preview = " ".join(result.chunk.text.split())
    if len(preview) > 400:
        preview = preview[:400].rstrip() + "..."
    return RetrievedSummary(
        document=result.chunk.document,
        start_page=result.chunk.start_page,
        end_page=result.chunk.end_page,
        chunk_id=result.chunk.chunk_id,
        score=result.score,
        text_preview=preview,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_metrics(
    evaluations: Sequence[QuestionEvaluation],
) -> ConditionMetrics:
    scored = [result for result in evaluations if result.answerability != "unanswerable"]

    def average_field(field: str) -> Optional[float]:
        values = [getattr(result, field) for result in scored]
        numeric = [float(value) for value in values if value is not None]
        return _average(numeric) if numeric else None

    found_ranks = [
        float(result.first_correct_rank)
        for result in scored
        if result.first_correct_rank is not None
    ]
    return ConditionMetrics(
        hit_at_1=average_field("hit_at_1"),
        hit_at_3=average_field("hit_at_3"),
        hit_at_5=average_field("hit_at_5"),
        recall_at_1=average_field("expected_source_recall_at_1"),
        recall_at_3=average_field("expected_source_recall_at_3"),
        recall_at_5=average_field("expected_source_recall_at_5"),
        mean_first_relevant_rank=(
            _average(found_ranks) if found_ranks else None
        ),
    )


def _expected_sources(example: EvaluationExample) -> List[Dict[str, Any]]:
    return [asdict(source) for source in example.expected_sources]


def evaluate_rewriting(
    examples: Sequence[EvaluationExample],
    retriever: Retriever,
    rewriter: QueryRewriter,
    metadata: CorpusIndexMetadata,
    *,
    preliminary_top_k: int = 5,
    final_top_k: int = 5,
    dataset_path: str = "",
    on_progress: Optional[Callable[[int, int, RewriteQuestionComparison], None]] = None,
) -> RewriteExperimentReport:
    """Run both conditions with shared retrieval and frozen relevance logic."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    if preliminary_top_k <= 0:
        raise ValueError("preliminary_top_k must be positive")
    if final_top_k < max(DEFAULT_HIT_KS):
        raise ValueError("final_top_k must be at least 5")
    if preliminary_top_k > final_top_k:
        raise ValueError("preliminary_top_k cannot exceed final_top_k")

    comparisons: List[RewriteQuestionComparison] = []
    baseline_evaluations: List[QuestionEvaluation] = []
    rewritten_evaluations: List[QuestionEvaluation] = []

    for example in examples:
        baseline_started = perf_counter()
        baseline_results = retriever.search(example.question, top_k=final_top_k)
        baseline_seconds = perf_counter() - baseline_started
        preliminary_results = baseline_results[:preliminary_top_k]

        rewrite = rewriter.rewrite(example.question, preliminary_results)

        final_started = perf_counter()
        final_results = retriever.search(
            rewrite.rewritten_query, top_k=final_top_k
        )
        final_seconds = perf_counter() - final_started

        baseline_evaluation = evaluate_retrieval_results(example, baseline_results)
        rewritten_evaluation = evaluate_retrieval_results(example, final_results)
        baseline_evaluations.append(baseline_evaluation)
        rewritten_evaluations.append(rewritten_evaluation)
        outcome = classify_rank_change(
            example.answerability,
            baseline_evaluation.first_correct_rank,
            rewritten_evaluation.first_correct_rank,
        )
        comparisons.append(
            RewriteQuestionComparison(
                question_id=example.id,
                original_question=example.question,
                answerability=example.answerability,
                expected_sources=_expected_sources(example),
                rewritten_query=rewrite.rewritten_query,
                changed=rewrite.changed,
                baseline_first_relevant_rank=baseline_evaluation.first_correct_rank,
                rewritten_first_relevant_rank=rewritten_evaluation.first_correct_rank,
                outcome=outcome,
                baseline_top_result=_top_result(baseline_results),
                preliminary_results=[
                    _retrieved_summary(result) for result in preliminary_results
                ],
                rewritten_top_result=_top_result(final_results),
                baseline_retrieval_seconds=baseline_seconds,
                preliminary_retrieval_seconds=baseline_seconds,
                rewrite_seconds=rewrite.latency_seconds,
                final_retrieval_seconds=final_seconds,
                rewrite_condition_total_seconds=(
                    baseline_seconds + rewrite.latency_seconds + final_seconds
                ),
                fallback=rewrite.fallback,
                rewrite_error=rewrite.error,
                identity_protected_terms=list(rewrite.identity.protected_terms),
                identity_missing_terms=list(rewrite.identity.missing_terms),
            )
        )
        if on_progress is not None:
            on_progress(len(comparisons), len(examples), comparisons[-1])

    rewrite_times = [item.rewrite_seconds for item in comparisons]
    changed = sum(item.changed for item in comparisons)
    summary = RewriteExperimentSummary(
        total_questions=len(comparisons),
        retrieval_scored_questions=sum(
            item.answerability != "unanswerable" for item in comparisons
        ),
        unanswerable_questions=sum(
            item.answerability == "unanswerable" for item in comparisons
        ),
        queries_changed=changed,
        changed_percentage=100.0 * changed / len(comparisons),
        queries_unchanged=len(comparisons) - changed,
        improved_questions=sum(item.outcome == "improved" for item in comparisons),
        degraded_questions=sum(item.outcome == "degraded" for item in comparisons),
        unchanged_questions=sum(item.outcome == "unchanged" for item in comparisons),
        rewrite_failures=sum(item.rewrite_error is not None for item in comparisons),
        invalid_or_empty_responses=sum(
            _invalid_rewrite_result(item) for item in comparisons
        ),
        fallbacks=sum(item.fallback for item in comparisons),
        identity_warning_questions=sum(
            bool(item.identity_missing_terms) for item in comparisons
        ),
        average_rewrite_seconds=_average(rewrite_times),
        p50_rewrite_seconds=statistics.median(rewrite_times),
        p95_rewrite_seconds=_percentile(rewrite_times, 0.95),
        average_baseline_retrieval_seconds=_average(
            [item.baseline_retrieval_seconds for item in comparisons]
        ),
        average_rewrite_condition_total_seconds=_average(
            [item.rewrite_condition_total_seconds for item in comparisons]
        ),
        baseline=_condition_metrics(baseline_evaluations),
        rewritten=_condition_metrics(rewritten_evaluations),
    )
    return RewriteExperimentReport(
        schema_version=1,
        experiment="phase-10a-corpus-grounded-single-query-rewriting",
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_path=dataset_path,
        dataset_sha256=(
            _sha256(Path(dataset_path))
            if dataset_path and Path(dataset_path).is_file()
            else None
        ),
        settings={
            "preliminary_top_k": preliminary_top_k,
            "final_top_k": final_top_k,
            "retrieval_strategy": retriever.name,
            "retrieval_score_type": retriever.score_name,
            "dense_backend": getattr(retriever, "backend_name", None),
            "embedding_model": metadata.embedding_model,
            "embedding_dimension": metadata.embedding_dimension,
            "indexed_chunks": metadata.chunk_count,
            "rewrite_model": rewriter.model_name,
            "rewrite_temperature": getattr(rewriter, "temperature", None),
            "max_chars_per_preliminary_chunk": getattr(
                rewriter, "max_chars_per_chunk", None
            ),
            "preliminary_results_reuse_baseline_call": True,
            "candidate_depth": getattr(retriever, "candidate_depth", None),
            "reranker_model": getattr(retriever, "reranker_model", None),
            "rrf_k": getattr(getattr(retriever, "base", None), "rrf_k", None),
            "rewrite_prompt_sha256": hashlib.sha256(
                REWRITE_PROMPT_TEMPLATE.encode("utf-8")
            ).hexdigest(),
        },
        summary=summary,
        questions=comparisons,
    )


def _invalid_rewrite_result(item: RewriteQuestionComparison) -> bool:
    if not item.rewrite_error:
        return False
    lowered = item.rewrite_error.lower()
    return "empty" in lowered or "multiple lines" in lowered


def write_json_report(report: RewriteExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(report.to_dict(), output, indent=2, ensure_ascii=False)
        output.write("\n")


def _metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _rank(value: Optional[int]) -> str:
    return "not found" if value is None else str(value)


def _recommendation(summary: RewriteExperimentSummary) -> str:
    baseline_hit5 = summary.baseline.hit_at_5 or 0.0
    rewritten_hit5 = summary.rewritten.hit_at_5 or 0.0
    if rewritten_hit5 < baseline_hit5 or summary.degraded_questions > summary.improved_questions:
        return "Reject as the default path; regressions outweigh the measured benefit."
    if summary.improved_questions > summary.degraded_questions:
        return "Continue experimenting; the signal is positive but does not justify production adoption yet."
    return "Do not adopt yet; the measured retrieval benefit is neutral."


def write_markdown_report(report: RewriteExperimentReport, path: Path) -> None:
    """Write an auditable summary plus the most informative rank movements."""

    summary = report.summary
    lines = [
        "# Phase 10A: corpus-grounded LLM query rewriting",
        "",
        "## Experiment",
        "",
        "The frozen Phase 7.5 retriever is compared with the same retriever preceded by one corpus-grounded LLM rewrite. The original question remains separate and is never replaced for answer generation.",
        "",
        f"Dataset: `{report.dataset_path}`",
        f"Rewrite model: `{report.settings['rewrite_model']}`",
        f"Preliminary/final top-k: {report.settings['preliminary_top_k']}/{report.settings['final_top_k']}",
        f"Retrieval: `{report.settings['retrieval_strategy']}` using `{report.settings['dense_backend']}`",
        "",
        "## Aggregate retrieval metrics",
        "",
        "| Metric | Baseline | Rewritten |",
        "|---|---:|---:|",
    ]
    for label, field in [
        ("Hit@1", "hit_at_1"),
        ("Hit@3", "hit_at_3"),
        ("Hit@5", "hit_at_5"),
        ("Recall@1", "recall_at_1"),
        ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"),
        ("Mean first relevant rank", "mean_first_relevant_rank"),
    ]:
        lines.append(
            f"| {label} | {_metric(getattr(summary.baseline, field))} | {_metric(getattr(summary.rewritten, field))} |"
        )
    lines.extend(
        [
            "",
            "## Rewrite behavior and latency",
            "",
            f"- Questions: {summary.total_questions} ({summary.retrieval_scored_questions} scored, {summary.unanswerable_questions} unanswerable)",
            f"- Changed: {summary.queries_changed} ({summary.changed_percentage:.1f}%); unchanged: {summary.queries_unchanged}",
            f"- Improved/degraded/unchanged: {summary.improved_questions}/{summary.degraded_questions}/{summary.unchanged_questions}",
            f"- Failures/fallbacks: {summary.rewrite_failures}/{summary.fallbacks}; invalid or empty: {summary.invalid_or_empty_responses}",
            f"- Identity warnings: {summary.identity_warning_questions}",
            f"- Average baseline retrieval: {summary.average_baseline_retrieval_seconds:.3f}s",
            f"- Average rewrite: {summary.average_rewrite_seconds:.3f}s (p50 {summary.p50_rewrite_seconds:.3f}s, p95 {summary.p95_rewrite_seconds:.3f}s)",
            f"- Average rewrite-condition total: {summary.average_rewrite_condition_total_seconds:.3f}s",
            "",
        ]
    )
    for heading, outcome, reverse in [
        ("Biggest improvements", "improved", True),
        ("Biggest regressions", "degraded", True),
    ]:
        selected = [item for item in report.questions if item.outcome == outcome]

        def magnitude(item: RewriteQuestionComparison) -> int:
            before = item.baseline_first_relevant_rank or 99
            after = item.rewritten_first_relevant_rank or 99
            return abs(before - after)

        selected.sort(key=magnitude, reverse=reverse)
        lines.extend(
            [
                f"## {heading}",
                "",
                "| ID | Original → rewritten | Rank change | Identity loss |",
                "|---|---|---:|---|",
            ]
        )
        for item in selected[:10]:
            query = f"{item.original_question} → {item.rewritten_query}".replace("|", "\\|")
            missing = ", ".join(item.identity_missing_terms) or "none"
            lines.append(
                f"| `{item.question_id}` | {query} | {_rank(item.baseline_first_relevant_rank)} → {_rank(item.rewritten_first_relevant_rank)} | {missing} |"
            )
        if not selected:
            lines.append("| — | None | — | — |")
        lines.append("")
    identity_items = [item for item in report.questions if item.identity_missing_terms]
    lines.extend(
        [
            "## Semantic-drift diagnostics",
            "",
            "Missing quoted phrases, acronyms, and model-like tokens are warnings, not relevance thresholds.",
            "",
        ]
    )
    for item in identity_items[:20]:
        lines.append(
            f"- `{item.question_id}` lost {', '.join(item.identity_missing_terms)}: `{item.rewritten_query}`"
        )
    if not identity_items:
        lines.append("- No protected-token losses detected.")
    failures = [item for item in report.questions if item.fallback]
    lines.extend(["", "## Failures and fallbacks", ""])
    for item in failures:
        lines.append(f"- `{item.question_id}`: {item.rewrite_error}")
    if not failures:
        lines.append("- No rewrite failures.")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            _recommendation(summary),
            "",
            "This experiment does not use cross-encoder scores as probabilities and introduces no confidence threshold. Per-question records are in the companion JSON report.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_results(label: str, results: Sequence[SearchResult]) -> None:
    print(f"\n{label}")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"{rank}. {chunk.document} page {chunk.start_page}-{chunk.end_page} "
            f"chunk={chunk.chunk_id} score={result.score:.4f}"
        )


def run_smoke(
    retriever: Retriever,
    rewriter: QueryRewriter,
    *,
    question: str,
    preliminary_top_k: int,
    final_top_k: int,
) -> None:
    print("ORIGINAL QUERY")
    print(question)
    preliminary = retriever.search(question, top_k=preliminary_top_k)
    _print_results("PRELIMINARY RESULTS", preliminary)
    rewrite = rewriter.rewrite(question, preliminary)
    print("\nREWRITTEN QUERY")
    print(rewrite.rewritten_query)
    print(
        f"changed={rewrite.changed} latency={rewrite.latency_seconds:.3f}s "
        f"identity_missing={list(rewrite.identity.missing_terms)}"
    )
    if rewrite.error:
        print(f"fallback_error={rewrite.error}")
    final = retriever.search(rewrite.rewritten_query, top_k=final_top_k)
    _print_results("FINAL RESULTS", final)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--question", default="Who wrote the DPR paper?")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--preliminary-top-k", type=int, default=5)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    settings = ApplicationSettings.from_environment()
    model = args.model or settings.ollama_model
    timeout = settings.ollama_timeout if args.timeout is None else args.timeout
    try:
        loaded = load_retriever(default_retrieval_config(settings))
    except Exception as exc:
        print(f"Error: retrieval startup failed: {exc}", file=sys.stderr)
        return 2
    try:
        generator = OllamaGenerator(
            model_name=model,
            base_url=settings.ollama_url,
            timeout=timeout,
        )
        try:
            generator.ensure_model_available()
        except GenerationError as exc:
            print(
                "Warning: rewrite model is unavailable; all rewrites will fall "
                f"back to the original query: {exc}",
                file=sys.stderr,
            )
            generator = _UnavailableGenerator(model, exc)
        rewriter = CorpusGroundedLLMRewriter(
            generator,
            temperature=args.temperature,
            max_chars_per_chunk=args.max_chars_per_chunk,
        )
        if args.smoke:
            run_smoke(
                loaded.retriever,
                rewriter,
                question=args.question,
                preliminary_top_k=args.preliminary_top_k,
                final_top_k=args.final_top_k,
            )
            return 0
        examples = load_evaluation_dataset(args.dataset)
        report = evaluate_rewriting(
            examples,
            loaded.retriever,
            rewriter,
            loaded.metadata,
            preliminary_top_k=args.preliminary_top_k,
            final_top_k=args.final_top_k,
            dataset_path=str(args.dataset),
            on_progress=lambda current, total, item: print(
                f"[{current}/{total}] {item.question_id}: "
                f"changed={item.changed} outcome={item.outcome} "
                f"fallback={item.fallback}",
                flush=True,
            ),
        )
        write_json_report(report, args.json_output)
        write_markdown_report(report, args.markdown_output)
        print(f"JSON report: {args.json_output}")
        print(f"Markdown report: {args.markdown_output}")
        return 0
    except (GenerationError, OSError, ValueError) as exc:
        print(f"Error: Phase 10A experiment failed: {exc}", file=sys.stderr)
        return 2
    finally:
        loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
