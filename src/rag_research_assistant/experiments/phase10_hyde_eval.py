"""Evaluate HyDE as an isolated retrieval probe over the frozen legacy index."""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..application import ApplicationSettings, default_retrieval_config, load_retriever
from ..evaluation import (
    EvaluationExample,
    QuestionEvaluation,
    evaluate_retrieval_results,
    load_evaluation_dataset,
)
from ..experiments.phase10_rewrite_eval import classify_rank_change
from ..generation import OllamaGenerator
from ..models import SearchResult
from ..qdrant_store import QdrantDenseRetriever
from ..query_transform.hyde import (
    HyDEDocumentGenerator,
    HyDEExperimentalRetriever,
)
from ..reranking import CrossEncoderReranker
from ..retrievers import RerankingRetriever


DEFAULT_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_JSON_OUTPUT = Path("data/evaluation/benchmarks/phase-10f-hyde-results.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/phase-10f-hyde-results.md")
DEFAULT_DIAGNOSTIC_JSON = Path("data/evaluation/benchmarks/phase-10f-hyde-diagnostic.json")
DEFAULT_DIAGNOSTIC_MARKDOWN = Path("docs/phase-10f-hyde-diagnostic.md")
EXPECTED_LEGACY_CHUNKS = 3_793


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _retrieval_metrics(evaluations: Sequence[QuestionEvaluation]) -> Dict[str, Any]:
    scored = [item for item in evaluations if item.answerability != "unanswerable"]

    def metric(field: str) -> Optional[float]:
        values = [float(getattr(item, field)) for item in scored if getattr(item, field) is not None]
        return _mean(values) if values else None

    ranks = [float(item.first_correct_rank) for item in scored if item.first_correct_rank is not None]
    return {
        "hit_at_1": metric("hit_at_1"),
        "hit_at_3": metric("hit_at_3"),
        "hit_at_5": metric("hit_at_5"),
        "recall_at_1": metric("expected_source_recall_at_1"),
        "recall_at_3": metric("expected_source_recall_at_3"),
        "recall_at_5": metric("expected_source_recall_at_5"),
        "mean_first_relevant_rank": _mean(ranks) if ranks else None,
    }


def _result_rows(values: Sequence[SearchResult]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": rank,
            "score": result.score,
            "chunk_id": result.chunk.chunk_id,
            "document": result.chunk.document,
            "start_page": result.chunk.start_page,
            "end_page": result.chunk.end_page,
            "text_preview": " ".join(result.chunk.text.split())[:500],
        }
        for rank, result in enumerate(values, start=1)
    ]


def _overlap_at_five(left: Sequence[SearchResult], right: Sequence[SearchResult]) -> int:
    return len(
        {result.chunk.chunk_id for result in left[:5]}
        & {result.chunk.chunk_id for result in right[:5]}
    )


def evaluate_experiment(
    examples: Sequence[EvaluationExample],
    generator: HyDEDocumentGenerator,
    retriever: HyDEExperimentalRetriever,
    *,
    question_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    dataset_path: str = "",
    configuration: Optional[Dict[str, Any]] = None,
    on_progress=None,
) -> Dict[str, Any]:
    """Generate once per question and persist every controlled retrieval branch."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    baseline_evaluations: List[QuestionEvaluation] = []
    hyde_evaluations: List[QuestionEvaluation] = []
    fused_evaluations: List[QuestionEvaluation] = []
    records = []
    for example in examples:
        metadata = (question_metadata or {}).get(example.id, {})
        hypothetical = generator.generate(example.question)
        result = retriever.search(example.question, hypothetical, top_k=5)
        baseline_eval = evaluate_retrieval_results(example, list(result.baseline_results))
        hyde_eval = evaluate_retrieval_results(example, list(result.hyde_only_results))
        fused_eval = evaluate_retrieval_results(example, list(result.fused_results))
        baseline_evaluations.append(baseline_eval)
        hyde_evaluations.append(hyde_eval)
        fused_evaluations.append(fused_eval)
        hyde_outcome = classify_rank_change(
            example.answerability, baseline_eval.first_correct_rank, hyde_eval.first_correct_rank
        )
        fused_outcome = classify_rank_change(
            example.answerability, baseline_eval.first_correct_rank, fused_eval.first_correct_rank
        )
        records.append(
            {
                "question_id": example.id,
                "original_question": example.question,
                "answerability": example.answerability,
                "category": metadata.get("category"),
                "difficulty": metadata.get("difficulty"),
                "expected_sources": [asdict(source) for source in example.expected_sources],
                "hypothetical_document": asdict(hypothetical),
                "fallback": result.fallback,
                "error": result.error,
                "baseline_first_relevant_rank": baseline_eval.first_correct_rank,
                "hyde_only_first_relevant_rank": hyde_eval.first_correct_rank,
                "fused_first_relevant_rank": fused_eval.first_correct_rank,
                "hyde_only_vs_baseline": hyde_outcome,
                "fused_vs_baseline": fused_outcome,
                "baseline_results": _result_rows(result.baseline_results),
                "hyde_dense_results": _result_rows(result.hyde_dense_results),
                "hyde_only_reranked_results": _result_rows(result.hyde_only_results),
                "fused_candidates": _result_rows(result.fused_candidates),
                "fused_final_results": _result_rows(result.fused_results),
                "hyde_dense_baseline_overlap_at_5": _overlap_at_five(
                    result.baseline_results, result.hyde_dense_results
                ),
                "hyde_only_baseline_overlap_at_5": _overlap_at_five(
                    result.baseline_results, result.hyde_only_results
                ),
                "fused_baseline_overlap_at_5": _overlap_at_five(
                    result.baseline_results, result.fused_results
                ),
                "baseline_retrieval_seconds": result.baseline_seconds,
                "hyde_generation_seconds": hypothetical.generation_seconds,
                "hyde_embedding_seconds": result.embedding_seconds,
                "hyde_dense_retrieval_seconds": result.dense_seconds,
                "hyde_only_reranking_seconds": result.hyde_reranking_seconds,
                "fusion_seconds": result.fusion_seconds,
                "fused_final_reranking_seconds": result.fused_reranking_seconds,
                "hyde_only_total_seconds": result.hyde_only_total_seconds,
                "fused_total_seconds": result.fused_total_seconds,
            }
        )
        if on_progress:
            on_progress(len(records), len(examples), records[-1])

    scored_records = [record for record in records if record["answerability"] != "unanswerable"]
    unanswerable = [record for record in records if record["answerability"] == "unanswerable"]

    def outcome_count(field: str, outcome: str) -> int:
        return sum(record[field] == outcome for record in scored_records)

    def found_transition(field: str, *, rescued: bool) -> int:
        return sum(
            (
                record["baseline_first_relevant_rank"] is None and record[field] is not None
            )
            if rescued
            else (
                record["baseline_first_relevant_rank"] is not None and record[field] is None
            )
            for record in scored_records
        )

    diagnostics = [
        record["hypothetical_document"]["diagnostics"]
        for record in records
        if record["hypothetical_document"]["diagnostics"] is not None
    ]
    generated = [
        record["hypothetical_document"]
        for record in records
        if not record["hypothetical_document"]["fallback"]
    ]
    category_metrics = {}
    categories = sorted(
        {record["category"] or "uncategorized" for record in scored_records}
    )
    for category in categories:
        indexes = [
            index
            for index, record in enumerate(records)
            if record["answerability"] != "unanswerable"
            and (record["category"] or "uncategorized") == category
        ]
        category_metrics[category] = {
            "questions": len(indexes),
            "baseline": _retrieval_metrics([baseline_evaluations[index] for index in indexes]),
            "hyde_only": _retrieval_metrics([hyde_evaluations[index] for index in indexes]),
            "fused": _retrieval_metrics([fused_evaluations[index] for index in indexes]),
        }
    summary = {
        "total_questions": len(records),
        "retrieval_scored_questions": len(scored_records),
        "unanswerable_questions": len(unanswerable),
        "baseline": _retrieval_metrics(baseline_evaluations),
        "hyde_only": _retrieval_metrics(hyde_evaluations),
        "fused": _retrieval_metrics(fused_evaluations),
        "hyde_only_improved": outcome_count("hyde_only_vs_baseline", "improved"),
        "hyde_only_unchanged": outcome_count("hyde_only_vs_baseline", "unchanged"),
        "hyde_only_degraded": outcome_count("hyde_only_vs_baseline", "degraded"),
        "fused_improved": outcome_count("fused_vs_baseline", "improved"),
        "fused_unchanged": outcome_count("fused_vs_baseline", "unchanged"),
        "fused_degraded": outcome_count("fused_vs_baseline", "degraded"),
        "hyde_only_not_found_to_found": found_transition("hyde_only_first_relevant_rank", rescued=True),
        "hyde_only_found_to_not_found": found_transition("hyde_only_first_relevant_rank", rescued=False),
        "fused_not_found_to_found": found_transition("fused_first_relevant_rank", rescued=True),
        "fused_found_to_not_found": found_transition("fused_first_relevant_rank", rescued=False),
        "generation_fallbacks": sum(record["fallback"] for record in records),
        "generation_or_branch_errors": sum(record["error"] is not None for record in records),
        "average_hypothetical_words": _mean([float(value["word_count"]) for value in generated]),
        "average_hypothetical_characters": _mean([float(value["character_count"]) for value in generated]),
        "truncated_hypothetical_documents": sum(value["truncated"] for value in generated),
        "missing_protected_term_questions": sum(bool(value["missing_protected_terms"]) for value in diagnostics),
        "introduced_numeric_detail_questions": sum(bool(value["introduced_numbers"]) for value in diagnostics),
        "citation_like_output_questions": sum(bool(value["citation_like_patterns"]) for value in diagnostics),
        "introduced_named_entity_questions": sum(bool(value["introduced_named_entities"]) for value in diagnostics),
        "questions_with_drift_warnings": sum(bool(value["warnings"]) for value in diagnostics),
        "average_query_term_coverage": _mean([value["query_term_coverage"] for value in diagnostics]),
        "average_lexical_jaccard": _mean([value["lexical_jaccard"] for value in diagnostics]),
        "average_baseline_retrieval_seconds": _mean([record["baseline_retrieval_seconds"] for record in records]),
        "average_hyde_generation_seconds": _mean([record["hyde_generation_seconds"] for record in records]),
        "average_hyde_embedding_seconds": _mean([record["hyde_embedding_seconds"] for record in records]),
        "average_hyde_dense_retrieval_seconds": _mean([record["hyde_dense_retrieval_seconds"] for record in records]),
        "average_hyde_only_reranking_seconds": _mean([record["hyde_only_reranking_seconds"] for record in records]),
        "average_fusion_seconds": _mean([record["fusion_seconds"] for record in records]),
        "average_fused_final_reranking_seconds": _mean([record["fused_final_reranking_seconds"] for record in records]),
        "average_hyde_only_total_seconds": _mean([record["hyde_only_total_seconds"] for record in records]),
        "average_fused_total_seconds": _mean([record["fused_total_seconds"] for record in records]),
        "unanswerable_hyde_top_result_changed": sum(
            [value["chunk_id"] for value in record["baseline_results"][:1]]
            != [value["chunk_id"] for value in record["hyde_only_reranked_results"][:1]]
            for record in unanswerable
        ),
        "unanswerable_fused_top_result_changed": sum(
            [value["chunk_id"] for value in record["baseline_results"][:1]]
            != [value["chunk_id"] for value in record["fused_final_results"][:1]]
            for record in unanswerable
        ),
        "average_unanswerable_fused_overlap_at_5": _mean(
            [float(record["fused_baseline_overlap_at_5"]) for record in unanswerable]
        ),
        "category_metrics": category_metrics,
    }
    return {
        "schema_version": 1,
        "experiment": "phase-10f-hypothetical-document-embeddings",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_sha256": (
            hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
            if dataset_path and Path(dataset_path).is_file() else None
        ),
        "configuration": configuration or {},
        "interpretation_notes": {
            "hypothetical_document": "retrieval probe only; never evidence",
            "reranking": "all final reranking uses the original question against real chunks",
            "unanswerable": "retained for drift inspection and excluded from retrieval metrics",
            "diagnostics": "audit signals only; no threshold blocks a probe",
        },
        "summary": summary,
        "questions": records,
    }


def write_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_question_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load reporting-only category fields without changing evaluation labels."""

    values = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            values[row["id"]] = {
                "category": row.get("category"),
                "difficulty": row.get("difficulty"),
            }
    return values


def _metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Phase 10F: HyDE results", "",
        "HyDE generates one answer-like passage as a semantic probe. The hypothetical text is never evidence, and every final cross-encoder call uses the original question against real legacy chunks.", "",
        f"Questions: {summary['total_questions']} ({summary['retrieval_scored_questions']} scored; {summary['unanswerable_questions']} unanswerable)", "",
        "| Metric | Baseline | HyDE-only | Baseline + HyDE |", "|---|---:|---:|---:|",
    ]
    for label, field in [
        ("Hit@1", "hit_at_1"), ("Hit@3", "hit_at_3"), ("Hit@5", "hit_at_5"),
        ("Recall@1", "recall_at_1"), ("Recall@3", "recall_at_3"),
        ("Recall@5", "recall_at_5"), ("Mean first relevant rank", "mean_first_relevant_rank"),
    ]:
        lines.append(
            f"| {label} | {_metric(summary['baseline'][field])} | {_metric(summary['hyde_only'][field])} | {_metric(summary['fused'][field])} |"
        )
    lines.extend([
        "", "## Category Hit@5", "",
        "| Category | Questions | Baseline | HyDE-only | Fused |",
        "|---|---:|---:|---:|---:|",
    ])
    for category, values in summary["category_metrics"].items():
        lines.append(
            f"| {category} | {values['questions']} | "
            f"{_metric(values['baseline']['hit_at_5'])} | "
            f"{_metric(values['hyde_only']['hit_at_5'])} | "
            f"{_metric(values['fused']['hit_at_5'])} |"
        )
    lines.extend([
        "", "## Rank movement", "",
        f"- HyDE-only improved/unchanged/degraded: {summary['hyde_only_improved']}/{summary['hyde_only_unchanged']}/{summary['hyde_only_degraded']}",
        f"- Fused improved/unchanged/degraded: {summary['fused_improved']}/{summary['fused_unchanged']}/{summary['fused_degraded']}",
        f"- HyDE-only rescued/lost top-five evidence: {summary['hyde_only_not_found_to_found']}/{summary['hyde_only_found_to_not_found']}",
        f"- Fused rescued/lost top-five evidence: {summary['fused_not_found_to_found']}/{summary['fused_found_to_not_found']}",
        "", "## Probe and latency", "",
        f"- Average hypothetical length: {summary['average_hypothetical_words']:.1f} words / {summary['average_hypothetical_characters']:.1f} characters",
        f"- Fallbacks/errors/truncations: {summary['generation_fallbacks']}/{summary['generation_or_branch_errors']}/{summary['truncated_hypothetical_documents']}",
        f"- Missing-protected-term / numeric-detail / citation-like output questions: {summary['missing_protected_term_questions']}/{summary['introduced_numeric_detail_questions']}/{summary['citation_like_output_questions']}",
        f"- Introduced-entity / any-warning questions: {summary['introduced_named_entity_questions']}/{summary['questions_with_drift_warnings']}",
        f"- Mean baseline retrieval: {summary['average_baseline_retrieval_seconds']:.3f}s",
        f"- Mean generation / embedding / HyDE dense retrieval: {summary['average_hyde_generation_seconds']:.3f}s / {summary['average_hyde_embedding_seconds']:.3f}s / {summary['average_hyde_dense_retrieval_seconds']:.3f}s",
        f"- Mean HyDE-only / fused total: {summary['average_hyde_only_total_seconds']:.3f}s / {summary['average_fused_total_seconds']:.3f}s",
        f"- Unanswerable top-result changes, HyDE-only/fused: {summary['unanswerable_hyde_top_result_changed']}/{summary['unanswerable_fused_top_result_changed']}",
        f"- Mean unanswerable fused overlap with baseline top five: {summary['average_unanswerable_fused_overlap_at_5']:.2f}/5",
        "", "## Interpretation boundary", "",
        "Do not infer a recommendation from this generated table alone. Inspect rescues, regressions, numeric/entity drift, unanswerable probes, and fusion safeguards before deciding whether HyDE merits retention as an optional experiment.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_results(label: str, values: Sequence[SearchResult]) -> None:
    print(f"\n{label}")
    for rank, value in enumerate(values, 1):
        chunk = value.chunk
        print(
            f"{rank}. {chunk.document} p{chunk.start_page}-{chunk.end_page} chunk={chunk.chunk_id} score={value.score:.4f}"
        )


def run_smoke(
    example: EvaluationExample,
    generator: HyDEDocumentGenerator,
    retriever: HyDEExperimentalRetriever,
) -> None:
    hypothetical = generator.generate(example.question)
    result = retriever.search(example.question, hypothetical, top_k=5)
    baseline_eval = evaluate_retrieval_results(example, list(result.baseline_results))
    hyde_eval = evaluate_retrieval_results(example, list(result.hyde_only_results))
    fused_eval = evaluate_retrieval_results(example, list(result.fused_results))
    print("QUESTION\n" + example.question)
    print("\nHYPOTHETICAL DOCUMENT\n" + (hypothetical.text or "<fallback: no document>"))
    print(
        f"\nwords={hypothetical.word_count} chars={hypothetical.character_count} generation={hypothetical.generation_seconds:.3f}s "
        f"truncated={hypothetical.truncated} fallback={result.fallback} error={result.error!r}"
    )
    print("DRIFT DIAGNOSTICS\n" + json.dumps(
        asdict(hypothetical.diagnostics) if hypothetical.diagnostics else None,
        indent=2,
        ensure_ascii=False,
    ))
    _print_results("BASELINE", result.baseline_results)
    _print_results("HYDE DENSE", result.hyde_dense_results)
    _print_results("HYDE-ONLY RERANKED", result.hyde_only_results)
    _print_results("FUSED FINAL", result.fused_results)
    print(
        "\nfirst_relevant_rank "
        f"baseline={baseline_eval.first_correct_rank} hyde_only={hyde_eval.first_correct_rank} fused={fused_eval.first_correct_rank}"
    )
    print(
        "latency_seconds "
        f"baseline={result.baseline_seconds:.3f} generation={hypothetical.generation_seconds:.3f} "
        f"embedding={result.embedding_seconds:.3f} dense={result.dense_seconds:.3f} "
        f"hyde_rerank={result.hyde_reranking_seconds:.3f} fusion={result.fusion_seconds:.6f} "
        f"fused_rerank={result.fused_reranking_seconds:.3f} "
        f"hyde_total={result.hyde_only_total_seconds:.3f} fused_total={result.fused_total_seconds:.3f}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--diagnostic", type=int, metavar="COUNT")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def _experimental_retriever(loaded) -> HyDEExperimentalRetriever:
    if not isinstance(loaded.retriever, RerankingRetriever):
        raise ValueError("frozen retriever must expose the existing reranker")
    if not isinstance(loaded.dense_retriever, QdrantDenseRetriever):
        raise ValueError("HyDE requires the frozen Qdrant dense retriever")
    reranker = loaded.retriever.reranker
    if not isinstance(reranker, CrossEncoderReranker):
        raise ValueError("frozen retriever does not expose the expected cross-encoder")
    return HyDEExperimentalRetriever(
        loaded.retriever, loaded.dense_retriever, reranker,
        candidate_depth=20, rrf_k=60,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.diagnostic is not None and args.diagnostic <= 0:
        print("Error: --diagnostic must be positive", file=sys.stderr)
        return 2
    loaded = None
    try:
        examples = load_evaluation_dataset(args.dataset)
        question_metadata = load_question_metadata(args.dataset)
        settings = ApplicationSettings.from_environment()
        loaded = load_retriever(default_retrieval_config(settings))
        if loaded.metadata.chunk_count != EXPECTED_LEGACY_CHUNKS:
            raise ValueError(
                f"frozen legacy corpus requires {EXPECTED_LEGACY_CHUNKS} chunks; index reports {loaded.metadata.chunk_count}"
            )
        ollama = OllamaGenerator(
            model_name=settings.ollama_model,
            base_url=settings.ollama_url,
            timeout=settings.ollama_timeout,
        )
        ollama.ensure_model_available()
        generator = HyDEDocumentGenerator(ollama, temperature=0.0, max_words=150)
        retriever = _experimental_retriever(loaded)
        if args.smoke:
            run_smoke(examples[0], generator, retriever)
            return 0
        if args.diagnostic is not None:
            examples = examples[: args.diagnostic]
            json_output = args.json_output or DEFAULT_DIAGNOSTIC_JSON
            markdown_output = args.markdown_output or DEFAULT_DIAGNOSTIC_MARKDOWN
        else:
            json_output = args.json_output or DEFAULT_JSON_OUTPUT
            markdown_output = args.markdown_output or DEFAULT_MARKDOWN_OUTPUT
        report = evaluate_experiment(
            examples, generator, retriever,
            question_metadata=question_metadata,
            dataset_path=str(args.dataset),
            configuration={
                "retrieval_stack": "frozen legacy Qdrant dense + BM25 + RRF + cross-encoder",
                "conditions": ["baseline", "hyde_only", "baseline_plus_hyde_rrf"],
                "candidate_depth": 20,
                "top_k": 5,
                "rrf_k": 60,
                "fusion_weights": "equal/unweighted",
                "embedding_model": loaded.metadata.embedding_model,
                "reranker_model": loaded.retriever.reranker_model,
                "generator_model": ollama.model_name,
                "generation_temperature": 0.0,
                "maximum_hypothetical_words": 150,
                "final_reranker_query": "original question",
                "hypothetical_document_is_evidence": False,
            },
            on_progress=lambda current, total, value: print(
                f"[{current}/{total}] {value['question_id']} hyde={value['hyde_only_vs_baseline']} fused={value['fused_vs_baseline']}",
                flush=True,
            ),
        )
        write_json(report, json_output)
        write_markdown(report, markdown_output)
        print(f"JSON report: {json_output}\nMarkdown report: {markdown_output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Phase 10F experiment failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if loaded is not None:
            loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
