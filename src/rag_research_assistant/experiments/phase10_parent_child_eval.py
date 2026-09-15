"""Evaluate semantic-child retrieval with page and structural parent expansion."""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

from ..application import (
    ApplicationSettings,
    RetrievalConfig,
    default_retrieval_config,
    load_retriever,
)
from ..evaluation import (
    EvaluationExample,
    QuestionEvaluation,
    evaluate_retrieval_results,
    load_evaluation_dataset,
)
from ..experiments.phase10_rewrite_eval import classify_rank_change
from ..models import SearchResult
from ..parent_child import ParentChildRetriever, build_parent_index
from ..parent_child.evaluation import ParentEvaluation, evaluate_parent_results
from ..parent_child.models import ParentRetrievalResult
from ..pdf import extract_directory
from ..pipeline import read_jsonl


DEFAULT_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_PDF_DIR = Path("data/corpus/pdfs")
DEFAULT_CHILD_CHUNKS = Path("data/processed/corpus-52/chunks-semantic.jsonl")
DEFAULT_STRUCTURAL_CHUNKS = Path("data/processed/corpus-52/chunks-structural.jsonl")
DEFAULT_CHILD_QDRANT = Path("data/processed/corpus-52/qdrant-semantic")
DEFAULT_CHILD_COLLECTION = "phase75_semantic"
EXPECTED_BASELINE_CHUNKS = 3_793
EXPECTED_CHILD_CHUNKS = 5_124
EXPECTED_STRUCTURAL_CHUNKS = 2_939
DEFAULT_JSON_OUTPUT = Path("data/evaluation/benchmarks/phase-10d-parent-child-results.json")
DEFAULT_MARKDOWN_OUTPUT = Path("docs/phase-10d-parent-child-results.md")
DEFAULT_DIAGNOSTIC_JSON = Path("data/evaluation/benchmarks/phase-10d-parent-child-diagnostic.json")
DEFAULT_DIAGNOSTIC_MARKDOWN = Path("docs/phase-10d-parent-child-diagnostic.md")


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_1: Optional[float]
    hit_at_3: Optional[float]
    hit_at_5: Optional[float]
    recall_at_1: Optional[float]
    recall_at_3: Optional[float]
    recall_at_5: Optional[float]
    mean_first_relevant_rank: Optional[float]


@dataclass(frozen=True)
class ParentMetrics:
    hit_at_1: Optional[float]
    hit_at_3: Optional[float]
    hit_at_5: Optional[float]
    recall_at_1: Optional[float]
    recall_at_3: Optional[float]
    recall_at_5: Optional[float]
    mean_first_relevant_rank: Optional[float]
    average_parent_characters: float
    median_parent_characters: float
    p95_parent_characters: float
    average_parent_tokens_approx: float
    average_expansion_ratio: float
    average_contributing_children: float
    parent_deduplication_rate: float
    relevant_child_mapping_rate: Optional[float]
    relevant_child_containment_rate: Optional[float]


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


def _retrieval_metrics(values: Sequence[QuestionEvaluation]) -> RetrievalMetrics:
    scored = [value for value in values if value.answerability != "unanswerable"]
    def mean(field: str) -> Optional[float]:
        numbers = [float(getattr(value, field)) for value in scored if getattr(value, field) is not None]
        return _average(numbers) if numbers else None
    ranks = [float(value.first_correct_rank) for value in scored if value.first_correct_rank is not None]
    return RetrievalMetrics(
        mean("hit_at_1"), mean("hit_at_3"), mean("hit_at_5"),
        mean("expected_source_recall_at_1"), mean("expected_source_recall_at_3"),
        mean("expected_source_recall_at_5"), _average(ranks) if ranks else None,
    )


def _parent_metrics(
    evaluations: Sequence[ParentEvaluation],
    results: Sequence[Sequence[ParentRetrievalResult]],
    child_result_counts: Sequence[int],
    unique_parent_counts: Sequence[int],
) -> ParentMetrics:
    scored = [value for value in evaluations if value.hit_at_1 is not None]
    def mean(field: str) -> Optional[float]:
        numbers = [float(getattr(value, field)) for value in scored if getattr(value, field) is not None]
        return _average(numbers) if numbers else None
    flattened = [result for ranking in results for result in ranking]
    sizes = [float(result.parent.character_count) for result in flattened]
    expansion = [result.expansion_ratio for result in flattened]
    contributions = [float(len(result.contributing_child_ids)) for result in flattened]
    input_children = sum(child_result_counts)
    unique_parents = sum(unique_parent_counts)
    relevant_children = sum(value.relevant_retrieved_child_count for value in scored)
    mapped = sum(value.relevant_children_mapped_into_returned_parents for value in scored)
    contained = sum(value.relevant_children_contained_by_returned_parents for value in scored)
    ranks = [float(value.first_relevant_rank) for value in scored if value.first_relevant_rank is not None]
    return ParentMetrics(
        mean("hit_at_1"), mean("hit_at_3"), mean("hit_at_5"),
        mean("recall_at_1"), mean("recall_at_3"), mean("recall_at_5"),
        _average(ranks) if ranks else None,
        _average(sizes), statistics.median(sizes) if sizes else 0.0,
        _percentile(sizes, 0.95),
        _average([float(len(result.parent.text.split())) for result in flattened]),
        _average(expansion), _average(contributions),
        (input_children - unique_parents) / input_children if input_children else 0.0,
        mapped / relevant_children if relevant_children else None,
        contained / relevant_children if relevant_children else None,
    )


def _search_summary(result: SearchResult, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank, "score": result.score, "chunk_id": result.chunk.chunk_id,
        "document": result.chunk.document, "start_page": result.chunk.start_page,
        "end_page": result.chunk.end_page, "character_count": len(result.chunk.text),
    }


def _parent_summary(result: ParentRetrievalResult, rank: int) -> Dict[str, Any]:
    return {
        "rank": rank, "score": result.score, "parent_id": result.parent.parent_id,
        "parent_type": result.parent.parent_type, "document": result.parent.document,
        "start_page": result.parent.start_page, "end_page": result.parent.end_page,
        "section_title": result.parent.section_title,
        "parent_character_count": result.parent.character_count,
        "parent_token_count_approx": len(result.parent.text.split()),
        "contributing_child_ids": list(result.contributing_child_ids),
        "contributing_child_ranks": list(result.contributing_child_ranks),
        "best_child_score": result.best_child_score,
        "best_child_rank": result.best_child_rank,
        "best_child_character_count": result.best_child_character_count,
        "expansion_ratio": result.expansion_ratio,
        "best_child_start_character_in_parent": result.best_child_start_character_in_parent,
        "characters_before_best_child": result.characters_before_best_child,
        "characters_after_best_child": result.characters_after_best_child,
        "parent_contains_all_contributing_children": result.parent_contains_all_contributing_children,
        "mapping_warnings": list(result.mapping_warnings),
        "text_preview": result.parent.text[:500],
    }


def evaluate_experiment(
    examples: Sequence[EvaluationExample],
    baseline_retriever,
    parent_child: ParentChildRetriever,
    *,
    parent_top_k: int = 5,
    dataset_path: str = "",
    configuration: Optional[Dict[str, Any]] = None,
    on_progress=None,
) -> Dict[str, Any]:
    if not examples:
        raise ValueError("at least one evaluation example is required")
    baseline_evals: List[QuestionEvaluation] = []
    child_evals: List[QuestionEvaluation] = []
    page_evals: List[ParentEvaluation] = []
    structural_evals: List[ParentEvaluation] = []
    page_rankings: List[Sequence[ParentRetrievalResult]] = []
    structural_rankings: List[Sequence[ParentRetrievalResult]] = []
    child_counts: List[int] = []
    page_unique_counts: List[int] = []
    structural_unique_counts: List[int] = []
    records = []
    for example in examples:
        baseline_started = perf_counter()
        baseline = baseline_retriever.search(example.question, top_k=5)
        baseline_seconds = perf_counter() - baseline_started
        expanded = parent_child.search(example.question, parent_top_k=parent_top_k)
        child_top5 = list(expanded.child_results[:5])
        baseline_eval = evaluate_retrieval_results(example, baseline)
        child_eval = evaluate_retrieval_results(example, child_top5)
        page_eval = evaluate_parent_results(
            example.expected_sources,
            expanded.child_results,
            expanded.page_parents,
            parent_child.parent_index,
            parent_type="page",
            answerability=example.answerability,
        )
        structural_eval = evaluate_parent_results(
            example.expected_sources,
            expanded.child_results,
            expanded.structural_parents,
            parent_child.parent_index,
            parent_type="structural",
            answerability=example.answerability,
        )
        baseline_evals.append(baseline_eval); child_evals.append(child_eval)
        page_evals.append(page_eval); structural_evals.append(structural_eval)
        page_rankings.append(expanded.page_parents); structural_rankings.append(expanded.structural_parents)
        child_counts.append(len(expanded.child_results))
        page_unique_counts.append(expanded.page_unique_parents_before_top_k)
        structural_unique_counts.append(expanded.structural_unique_parents_before_top_k)
        record = {
            "question_id": example.id,
            "original_question": example.question,
            "answerability": example.answerability,
            "expected_sources": [asdict(source) for source in example.expected_sources],
            "baseline_first_relevant_rank": baseline_eval.first_correct_rank,
            "child_first_relevant_rank": child_eval.first_correct_rank,
            "page_parent_first_relevant_rank": page_eval.first_relevant_rank,
            "structural_parent_first_relevant_rank": structural_eval.first_relevant_rank,
            "child_vs_baseline_outcome": classify_rank_change(
                example.answerability, baseline_eval.first_correct_rank, child_eval.first_correct_rank
            ),
            "page_parent_vs_baseline_outcome": classify_rank_change(
                example.answerability, baseline_eval.first_correct_rank, page_eval.first_relevant_rank
            ),
            "structural_parent_vs_baseline_outcome": classify_rank_change(
                example.answerability, baseline_eval.first_correct_rank, structural_eval.first_relevant_rank
            ),
            "page_parent_vs_child_outcome": classify_rank_change(
                example.answerability, child_eval.first_correct_rank, page_eval.first_relevant_rank
            ),
            "structural_parent_vs_child_outcome": classify_rank_change(
                example.answerability, child_eval.first_correct_rank, structural_eval.first_relevant_rank
            ),
            "baseline_results": [_search_summary(value, rank) for rank, value in enumerate(baseline, 1)],
            "child_results": [_search_summary(value, rank) for rank, value in enumerate(expanded.child_results, 1)],
            "page_parent_results": [_parent_summary(value, rank) for rank, value in enumerate(expanded.page_parents, 1)],
            "structural_parent_results": [_parent_summary(value, rank) for rank, value in enumerate(expanded.structural_parents, 1)],
            "page_evidence_preservation": asdict(page_eval),
            "structural_evidence_preservation": asdict(structural_eval),
            "baseline_retrieval_seconds": baseline_seconds,
            "child_retrieval_seconds": expanded.child_retrieval_seconds,
            "parent_mapping_seconds": expanded.parent_mapping_seconds,
            "page_aggregation_seconds": expanded.page_aggregation_seconds,
            "structural_aggregation_seconds": expanded.structural_aggregation_seconds,
            "page_unique_parents_before_top_k": expanded.page_unique_parents_before_top_k,
            "structural_unique_parents_before_top_k": expanded.structural_unique_parents_before_top_k,
            "parent_child_total_seconds": expanded.total_seconds,
            "page_parent_fallback_warnings": sum(bool(value.mapping_warnings) for value in expanded.page_parents),
            "structural_parent_fallback_warnings": sum(bool(value.mapping_warnings) for value in expanded.structural_parents),
        }
        records.append(record)
        if on_progress:
            on_progress(len(records), len(examples), record)

    index = parent_child.parent_index
    baseline_times = [value["baseline_retrieval_seconds"] for value in records]
    child_times = [value["child_retrieval_seconds"] for value in records]
    mapping_times = [value["parent_mapping_seconds"] for value in records]
    page_times = [value["page_aggregation_seconds"] for value in records]
    structural_times = [value["structural_aggregation_seconds"] for value in records]
    return {
        "schema_version": 1,
        "experiment": "phase-10d-semantic-child-parent-expansion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest() if dataset_path and Path(dataset_path).is_file() else None,
        "configuration": configuration or {},
        "structural_coverage": {
            "total_child_chunks": index.total_children,
            "children_with_section_metadata": index.children_with_section_metadata,
            "children_without_section_metadata": index.total_children - index.children_with_section_metadata,
            "structural_exact_mappings": index.structural_exact_mappings,
            "structural_page_fallback_mappings": index.structural_fallback_mappings,
            "structural_exact_coverage": index.structural_coverage,
            "unique_page_parents": sum(parent.parent_type == "page" for parent in index.parents.values()),
            "unique_structural_parents": sum(parent.parent_type == "structural" for parent in index.parents.values()),
            "fallback_rule": "page parent when no unique same-section structural chunk contains the full child text",
        },
        "summary": {
            "total_questions": len(records),
            "retrieval_scored_questions": sum(value.answerability != "unanswerable" for value in examples),
            "unanswerable_questions": sum(value.answerability == "unanswerable" for value in examples),
            "baseline": asdict(_retrieval_metrics(baseline_evals)),
            "child": asdict(_retrieval_metrics(child_evals)),
            "page_parent": asdict(_parent_metrics(page_evals, page_rankings, child_counts, page_unique_counts)),
            "structural_parent": asdict(_parent_metrics(structural_evals, structural_rankings, child_counts, structural_unique_counts)),
            "child_improved": sum(value["child_vs_baseline_outcome"] == "improved" for value in records),
            "child_degraded": sum(value["child_vs_baseline_outcome"] == "degraded" for value in records),
            "child_unchanged": sum(value["child_vs_baseline_outcome"] == "unchanged" for value in records),
            "page_parent_harms_child_hit": sum(
                value["child_first_relevant_rank"] is not None and value["page_parent_first_relevant_rank"] is None
                for value in records if value["answerability"] != "unanswerable"
            ),
            "structural_parent_harms_child_hit": sum(
                value["child_first_relevant_rank"] is not None and value["structural_parent_first_relevant_rank"] is None
                for value in records if value["answerability"] != "unanswerable"
            ),
            "average_baseline_retrieval_seconds": _average(baseline_times),
            "average_child_retrieval_seconds": _average(child_times),
            "average_parent_mapping_seconds": _average(mapping_times),
            "average_page_aggregation_seconds": _average(page_times),
            "average_structural_aggregation_seconds": _average(structural_times),
            "average_parent_child_total_seconds": _average([value["parent_child_total_seconds"] for value in records]),
        },
        "questions": records,
    }


def write_json(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    s = report["summary"]
    lines = [
        "# Phase 10D: parent-child retrieval results", "",
        "Semantic child chunks are retrieved through the frozen dense+BM25+RRF+cross-encoder stack, then expanded to physical-page or conservatively matched structural parents. Parent relevance is provenance-based: the parent document/page span must cover an expected source.", "",
        "## Child retrieval", "",
        "| Metric | Legacy baseline | Semantic child |", "|---|---:|---:|",
    ]
    for label, field in [("Hit@1", "hit_at_1"), ("Hit@3", "hit_at_3"), ("Hit@5", "hit_at_5"), ("Recall@1", "recall_at_1"), ("Recall@3", "recall_at_3"), ("Recall@5", "recall_at_5"), ("Mean first relevant rank", "mean_first_relevant_rank")]:
        lines.append(f"| {label} | {_metric(s['baseline'][field])} | {_metric(s['child'][field])} |")
    lines.extend(["", "## Parent retrieval", "", "| Metric | Page parent | Structural parent |", "|---|---:|---:|"])
    for label, field in [("Hit@1", "hit_at_1"), ("Hit@3", "hit_at_3"), ("Hit@5", "hit_at_5"), ("Recall@5", "recall_at_5"), ("Mean first relevant rank", "mean_first_relevant_rank"), ("Average characters", "average_parent_characters"), ("Median characters", "median_parent_characters"), ("p95 characters", "p95_parent_characters"), ("Average expansion ratio", "average_expansion_ratio"), ("Deduplication rate", "parent_deduplication_rate")]:
        lines.append(f"| {label} | {_metric(s['page_parent'][field])} | {_metric(s['structural_parent'][field])} |")
    coverage = report["structural_coverage"]
    lines.extend([
        "", "## Coverage and latency", "",
        f"- Structural exact mappings: {coverage['structural_exact_mappings']}/{coverage['total_child_chunks']} ({100*coverage['structural_exact_coverage']:.1f}%); page fallbacks: {coverage['structural_page_fallback_mappings']}",
        f"- Average baseline/child retrieval: {s['average_baseline_retrieval_seconds']:.3f}s / {s['average_child_retrieval_seconds']:.3f}s",
        f"- Average parent mapping: {s['average_parent_mapping_seconds']:.6f}s",
        f"- Average page/structural aggregation: {s['average_page_aggregation_seconds']:.6f}s / {s['average_structural_aggregation_seconds']:.6f}s",
        "", "## Interpretation", "",
        f"- Child improved/degraded/unchanged: {s['child_improved']}/{s['child_degraded']}/{s['child_unchanged']}",
        f"- Page/structural parents losing a child top-five hit: {s['page_parent_harms_child_hit']}/{s['structural_parent_harms_child_hit']}",
        f"- Relevant-child return rate, page/structural: {_metric(s['page_parent']['relevant_child_mapping_rate'])}/{_metric(s['structural_parent']['relevant_child_mapping_rate'])}",
        f"- Strict relevant-child containment, page/structural: {_metric(s['page_parent']['relevant_child_containment_rate'])}/{_metric(s['structural_parent']['relevant_child_containment_rate'])}",
        "",
        "Legacy remains the production default. Page parents are broad and weaken early ranking. Conservative structural parents are the more promising experimental branch because they preserve top-five retrieval with smaller exact contexts, but incomplete structural coverage and page fallbacks prevent integration from this result alone.",
        "",
        "Parent Hit measures expected document/page coverage, not answer-generation quality. Inspect per-question mappings and strict containment before treating broader provenance as better context.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_search(label: str, values: Sequence[SearchResult]) -> None:
    print(f"\n{label}")
    for rank, value in enumerate(values, 1):
        print(f"{rank}. {value.chunk.document} p{value.chunk.start_page}-{value.chunk.end_page} child={value.chunk.chunk_id} chars={len(value.chunk.text)} score={value.score:.4f}")


def _print_parents(label: str, values: Sequence[ParentRetrievalResult]) -> None:
    print(f"\n{label}")
    for rank, value in enumerate(values, 1):
        p = value.parent
        print(f"{rank}. {p.document} p{p.start_page}-{p.end_page} parent={p.parent_id} type={p.parent_type} chars={len(p.text)} ratio={value.expansion_ratio:.2f} children={list(value.contributing_child_ids)} warnings={list(value.mapping_warnings)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--diagnostic", type=int, metavar="COUNT")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--question", default="What learning signal allows REALM to pretrain its retriever without supervised retrieval labels?")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--child-chunks", type=Path, default=DEFAULT_CHILD_CHUNKS)
    parser.add_argument("--structural-chunks", type=Path, default=DEFAULT_STRUCTURAL_CHUNKS)
    parser.add_argument("--child-qdrant", type=Path, default=DEFAULT_CHILD_QDRANT)
    parser.add_argument("--child-collection", default=DEFAULT_CHILD_COLLECTION)
    parser.add_argument("--child-depth", type=int, default=20)
    parser.add_argument("--parent-top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.diagnostic is not None and args.diagnostic <= 0:
        print("Error: --diagnostic must be positive", file=sys.stderr); return 2
    settings = ApplicationSettings.from_environment()
    baseline = child = None
    try:
        children = read_jsonl(args.child_chunks)
        structural = read_jsonl(args.structural_chunks)
        if len(children) != EXPECTED_CHILD_CHUNKS:
            raise ValueError(
                f"frozen semantic child corpus requires {EXPECTED_CHILD_CHUNKS} chunks; "
                f"found {len(children)}"
            )
        if len(structural) != EXPECTED_STRUCTURAL_CHUNKS:
            raise ValueError(
                f"frozen structural corpus requires {EXPECTED_STRUCTURAL_CHUNKS} chunks; "
                f"found {len(structural)}"
            )
        parent_build_started = perf_counter()
        pages = list(extract_directory(args.pdf_dir))
        parent_index = build_parent_index(children, pages, structural)
        parent_build_seconds = perf_counter() - parent_build_started
        baseline = load_retriever(default_retrieval_config(settings))
        child = load_retriever(RetrievalConfig(
            chunks_path=args.child_chunks,
            index_path=Path("data/processed/corpus-52/index-semantic"),
            dense_backend="qdrant", qdrant_path=args.child_qdrant,
            collection_name=args.child_collection, retriever_name="hybrid", rerank=True,
            candidate_depth=20, device=settings.device,
        ))
        if baseline.metadata.chunk_count != EXPECTED_BASELINE_CHUNKS:
            raise ValueError(
                f"frozen baseline requires {EXPECTED_BASELINE_CHUNKS} chunks; "
                f"index reports {baseline.metadata.chunk_count}"
            )
        if child.metadata.chunk_count != EXPECTED_CHILD_CHUNKS:
            raise ValueError(
                f"semantic child index requires {EXPECTED_CHILD_CHUNKS} chunks; "
                f"index reports {child.metadata.chunk_count}"
            )
        parent_child = ParentChildRetriever(child.retriever, parent_index, child_depth=args.child_depth)
        if args.smoke:
            print("QUESTION\n" + args.question)
            base = baseline.retriever.search(args.question, top_k=5)
            expanded = parent_child.search(args.question, parent_top_k=args.parent_top_k)
            _print_search("BASELINE LEGACY RESULTS", base)
            _print_search("SEMANTIC CHILD RESULTS", expanded.child_results[:5])
            _print_parents("PAGE PARENT RESULTS", expanded.page_parents)
            _print_parents("STRUCTURAL PARENT RESULTS", expanded.structural_parents)
            print(f"\nstructural_coverage={parent_index.structural_coverage:.1%} parent_index_build={parent_build_seconds:.3f}s total_query={expanded.total_seconds:.3f}s")
            return 0
        examples = load_evaluation_dataset(args.dataset)
        if args.diagnostic is not None:
            examples = examples[:args.diagnostic]
            json_output = args.json_output or DEFAULT_DIAGNOSTIC_JSON
            markdown_output = args.markdown_output or DEFAULT_DIAGNOSTIC_MARKDOWN
        else:
            json_output = args.json_output or DEFAULT_JSON_OUTPUT
            markdown_output = args.markdown_output or DEFAULT_MARKDOWN_OUTPUT
        report = evaluate_experiment(
            examples, baseline.retriever, parent_child,
            parent_top_k=args.parent_top_k, dataset_path=str(args.dataset),
            configuration={
                "baseline_chunks": str(settings.chunks_path), "child_strategy": "semantic",
                "child_chunks": str(args.child_chunks), "child_chunk_count": len(children),
                "structural_chunks": str(args.structural_chunks), "parent_index_build_seconds": parent_build_seconds,
                "extracted_pages": len(pages), "corpus_documents": len({page.document for page in pages}),
                "child_depth": args.child_depth, "parent_top_k": args.parent_top_k,
                "child_retrieval_strategy": child.retriever.name,
                "embedding_model": child.metadata.embedding_model,
                "reranker_model": getattr(child.retriever, "reranker_model", None),
                "parent_aggregation": "sum 1/(60+child_rank) after child reranking",
            },
            on_progress=lambda current, total, value: print(f"[{current}/{total}] {value['question_id']} child={value['child_vs_baseline_outcome']}", flush=True),
        )
        write_json(report, json_output); write_markdown(report, markdown_output)
        print(f"JSON report: {json_output}\nMarkdown report: {markdown_output}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: Phase 10D experiment failed: {exc}", file=sys.stderr); return 2
    finally:
        if child is not None: child.close()
        if baseline is not None: baseline.close()


if __name__ == "__main__":
    raise SystemExit(main())
