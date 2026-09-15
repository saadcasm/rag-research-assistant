"""Evaluate experimental query decomposition on auditable multi-hop evidence."""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..application import ApplicationSettings, default_retrieval_config, load_retriever
from ..evaluation import evaluate_retrieval_results, load_evaluation_dataset
from ..generation import OllamaGenerator
from ..models import SearchResult
from ..multihop import (
    FusedCondition,
    MultiHopExample,
    MultiHopRetriever,
    aggregate_hop_coverage,
    evaluate_hop_coverage,
    load_multihop_dataset,
    verify_evidence_passages,
)
from ..pipeline import read_jsonl
from ..query_transform.decomposition import QueryDecomposer
from ..retrievers import RerankingRetriever


DEFAULT_DATASET = Path("data/evaluation/questions-phase-10g-multihop-draft.jsonl")
REGRESSION_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_JSON = Path("data/evaluation/benchmarks/phase-10g-multihop-results.json")
DEFAULT_MARKDOWN = Path("docs/phase-10g-multihop-results.md")
DIAGNOSTIC_JSON = Path("data/evaluation/benchmarks/phase-10g-multihop-diagnostic.json")
DIAGNOSTIC_MARKDOWN = Path("docs/phase-10g-multihop-diagnostic.md")
REGRESSION_JSON = Path("data/evaluation/benchmarks/phase-10g-multihop-regression.json")
REGRESSION_MARKDOWN = Path("docs/phase-10g-multihop-regression.md")
EXPECTED_LEGACY_CHUNKS = 3_793
CONDITIONS = ("baseline", "always_decompose", "original_plus_decomposed")


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_rows(condition: FusedCondition) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "score": result.score,
            "chunk_id": result.chunk.chunk_id,
            "document": result.chunk.document,
            "start_page": result.chunk.start_page,
            "end_page": result.chunk.end_page,
            "text_preview": " ".join(result.chunk.text.split())[:500],
            "contributing_queries": list(condition.contributions.get(result.chunk.chunk_id, ())),
        }
        for rank, result in enumerate(condition.results, 1)
    ]


def _condition_record(
    example: MultiHopExample, condition: FusedCondition
) -> dict[str, Any]:
    coverage = {
        str(k): asdict(evaluate_hop_coverage(example, condition.results, top_k=k))
        for k in (3, 5, 10)
    }
    final_ids = {result.chunk.chunk_id for result in condition.results}
    contribution_counts = {
        branch.query: sum(
            result.chunk.chunk_id in final_ids for result in branch.results
        )
        for branch in condition.branch_rankings
    }
    top_count = len(condition.results)
    dominant_share = max(contribution_counts.values(), default=0) / top_count if top_count else 0.0
    return {
        "branch_rankings": [
            {
                "query": branch.query,
                "latency_seconds": branch.latency_seconds,
                "results": [
                    {
                        "rank": rank,
                        "score": result.score,
                        "chunk_id": result.chunk.chunk_id,
                        "document": result.chunk.document,
                        "start_page": result.chunk.start_page,
                        "end_page": result.chunk.end_page,
                    }
                    for rank, result in enumerate(branch.results, 1)
                ],
            }
            for branch in condition.branch_rankings
        ],
        "fused_candidates": [
            {
                "rank": rank,
                "score": result.score,
                "chunk_id": result.chunk.chunk_id,
                "document": result.chunk.document,
                "start_page": result.chunk.start_page,
                "end_page": result.chunk.end_page,
                "contributing_queries": list(condition.contributions.get(result.chunk.chunk_id, ())),
            }
            for rank, result in enumerate(condition.candidates, 1)
        ],
        "final_results": _result_rows(condition),
        "hop_coverage": coverage,
        "candidate_contribution_to_final": contribution_counts,
        "dominant_query_share_of_final": dominant_share,
        "deduplicated_occurrences": condition.deduplicated_occurrences,
        "source_diversity": len({result.chunk.document for result in condition.results}),
        "retrieval_seconds": sum(branch.latency_seconds for branch in condition.branch_rankings),
        "fusion_seconds": condition.fusion_seconds,
        "reranking_seconds": condition.reranking_seconds,
    }


def evaluate_multihop(
    examples: Sequence[MultiHopExample],
    decomposer: QueryDecomposer,
    retriever: MultiHopRetriever,
    *,
    dataset_path: Path,
    settings: Mapping[str, Any],
    on_progress=None,
) -> dict[str, Any]:
    """Run all three conditions and retain enough state to reconstruct every rank."""

    if not examples:
        raise ValueError("at least one multi-hop example is required")
    records = []
    rankings: dict[str, list[Sequence[SearchResult]]] = {name: [] for name in CONDITIONS}
    for index, example in enumerate(examples, 1):
        decomposition = decomposer.decompose(example.question)
        retrieved = retriever.search(example.question, decomposition, top_k=10)
        conditions = {
            "baseline": retrieved.baseline,
            "always_decompose": retrieved.always_decompose,
            "original_plus_decomposed": retrieved.original_plus_decomposed,
        }
        for name, condition in conditions.items():
            rankings[name].append(condition.results)
        record = {
            "question_id": example.id,
            "question": example.question,
            "answerability": example.answerability,
            "category": example.category,
            "expected_hops": [asdict(hop) for hop in example.hops],
            "decomposition": asdict(decomposition),
            "fallback": retrieved.fallback,
            "error": retrieved.error,
            "conditions": {
                name: _condition_record(example, condition)
                for name, condition in conditions.items()
            },
            "total_seconds": decomposition.latency_seconds + sum(
                conditions["original_plus_decomposed"].branch_rankings[i].latency_seconds
                for i in range(len(conditions["original_plus_decomposed"].branch_rankings))
            ) + conditions["original_plus_decomposed"].fusion_seconds + conditions["original_plus_decomposed"].reranking_seconds,
        }
        records.append(record)
        if on_progress:
            on_progress(index, len(examples), record)

    summary: dict[str, Any] = {
        "questions": len(examples),
        "answerable": sum(example.answerability == "answerable" for example in examples),
        "partially_answerable": sum(example.answerability == "partially_answerable" for example in examples),
        "decomposition_requested": sum(record["decomposition"]["needs_decomposition"] for record in records),
        "fallbacks": sum(record["fallback"] for record in records),
        "average_decomposition_seconds": _mean([record["decomposition"]["latency_seconds"] for record in records]),
        "average_total_seconds": _mean([record["total_seconds"] for record in records]),
        "conditions": {},
    }
    for name in CONDITIONS:
        condition_records = [record["conditions"][name] for record in records]
        summary["conditions"][name] = {
            f"hop_coverage_at_{k}": aggregate_hop_coverage(examples, rankings[name], top_k=k)
            for k in (3, 5, 10)
        }
        summary["conditions"][name].update({
            "average_source_diversity_at_10": _mean([row["source_diversity"] for row in condition_records]),
            "average_deduplicated_occurrences": _mean([row["deduplicated_occurrences"] for row in condition_records]),
            "average_dominant_query_share": _mean([row["dominant_query_share_of_final"] for row in condition_records]),
            "average_retrieval_seconds": _mean([row["retrieval_seconds"] for row in condition_records]),
            "average_reranking_seconds": _mean([row["reranking_seconds"] for row in condition_records]),
        })
    return {
        "schema_version": 1,
        "experiment": "phase-10g-query-decomposition-multihop-retrieval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "settings": dict(settings),
        "summary": summary,
        "questions": records,
    }


def evaluate_regression(examples, decomposer, retriever, *, dataset_path: Path, settings: Mapping[str, Any], on_progress=None):
    records = []
    evaluations = {name: [] for name in CONDITIONS}
    for index, example in enumerate(examples, 1):
        decomposition = decomposer.decompose(example.question)
        retrieved = retriever.search(example.question, decomposition, top_k=5)
        conditions = {
            "baseline": retrieved.baseline.results,
            "always_decompose": retrieved.always_decompose.results,
            "original_plus_decomposed": retrieved.original_plus_decomposed.results,
        }
        question_evals = {}
        for name, results in conditions.items():
            evaluated = evaluate_retrieval_results(example, list(results))
            evaluations[name].append(evaluated)
            question_evals[name] = {
                "first_correct_rank": evaluated.first_correct_rank,
                "results": [
                    {"rank": rank, "chunk_id": item.chunk.chunk_id, "document": item.chunk.document,
                     "start_page": item.chunk.start_page, "end_page": item.chunk.end_page, "score": item.score}
                    for rank, item in enumerate(results, 1)
                ],
            }
        records.append({
            "question_id": example.id, "question": example.question,
            "answerability": example.answerability, "decomposition": asdict(decomposition),
            "fallback": retrieved.fallback, "error": retrieved.error, "conditions": question_evals,
        })
        if on_progress:
            on_progress(index, len(examples), records[-1])

    def metrics(values):
        scored = [item for item in values if item.answerability != "unanswerable"]
        def average(field):
            found = [float(getattr(item, field)) for item in scored if getattr(item, field) is not None]
            return _mean(found) if found else None
        ranks = [float(item.first_correct_rank) for item in scored if item.first_correct_rank is not None]
        return {
            "hit_at_1": average("hit_at_1"), "hit_at_3": average("hit_at_3"), "hit_at_5": average("hit_at_5"),
            "recall_at_1": average("expected_source_recall_at_1"),
            "recall_at_3": average("expected_source_recall_at_3"),
            "recall_at_5": average("expected_source_recall_at_5"),
            "mean_first_correct_rank": _mean(ranks) if ranks else None,
        }
    return {
        "schema_version": 1, "experiment": "phase-10g-regression",
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path), "settings": dict(settings),
        "summary": {
            "questions": len(examples),
            "decomposition_requested": sum(r["decomposition"]["needs_decomposition"] for r in records),
            "fallbacks": sum(r["fallback"] for r in records),
            "conditions": {name: metrics(values) for name, values in evaluations.items()},
        },
        "questions": records,
    }


def write_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Phase 10G: Query Decomposition and Multi-Hop Retrieval", "",
        "> Experimental only. The production retrieval path is unchanged.", "",
        f"Dataset: `{report['dataset_path']}`", "",
        f"Questions: **{summary['questions']}**", "",
    ]
    if report["experiment"].endswith("multihop-retrieval"):
        lines.extend([
            f"Answerable: **{summary['answerable']}**; partially answerable: **{summary['partially_answerable']}**", "",
            f"Successful decompositions: **{summary['decomposition_requested']}**; fallbacks: **{summary['fallbacks']}**", "",
            "## Hop coverage", "",
            "| Condition | Full@3 | Full@5 | Full@10 | Partial@3 | Partial@5 | Partial@10 |", "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name in CONDITIONS:
            values = summary["conditions"][name]
            lines.append("| " + name + " | " + " | ".join(
                f"{values[f'hop_coverage_at_{k}'][field]:.4f}"
                for field in ("full_hop_coverage", "partial_hop_coverage")
                for k in (3, 5, 10)
            ) + " |")
    else:
        lines.extend(["## Regression metrics", "", "```json", json.dumps(summary, indent=2), "```"])
    lines.extend(["", "## Interpretation status", "", "No production decision is encoded here. Review the per-question artifact before drawing conclusions."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(example: MultiHopExample, decomposer: QueryDecomposer, retriever: MultiHopRetriever) -> None:
    decomposition = decomposer.decompose(example.question)
    result = retriever.search(example.question, decomposition, top_k=10)
    print(f"Question: {example.question}")
    print("Expected hops:")
    for index, hop in enumerate(example.hops, 1):
        print(f"  {index}. {hop.description} (available={hop.evidence_available})")
    print(f"Decision: needs_decomposition={decomposition.needs_decomposition} fallback={result.fallback}")
    print(f"Sub-questions: {list(decomposition.subquestions)}")
    print(f"Rejected: {[asdict(item) for item in decomposition.rejected_subquestions]}")
    for name, condition in (
        ("baseline", result.baseline), ("always_decompose", result.always_decompose),
        ("original_plus_decomposed", result.original_plus_decomposed),
    ):
        print(f"\n{name}:")
        for branch in condition.branch_rankings:
            print(f"  branch={branch.query!r} candidates={len(branch.results)} seconds={branch.latency_seconds:.3f}")
        for row in _result_rows(condition):
            print(f"  {row['rank']}. {row['document']} p{row['start_page']}-{row['end_page']} {row['chunk_id']} via={row['contributing_queries']}")
        print(f"  coverage@10={asdict(evaluate_hop_coverage(example, condition.results, top_k=10))}")
    print(f"\nlatency decomposition={decomposition.latency_seconds:.3f}s fallback={result.fallback} error={result.error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--diagnostic", type=int, metavar="COUNT")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--regression", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.diagnostic is not None and args.diagnostic <= 0:
        print("Error: --diagnostic must be positive", file=sys.stderr)
        return 2
    loaded = None
    try:
        settings = ApplicationSettings.from_environment()
        loaded = load_retriever(default_retrieval_config(settings))
        if loaded.metadata.chunk_count != EXPECTED_LEGACY_CHUNKS:
            raise ValueError(f"frozen legacy corpus requires {EXPECTED_LEGACY_CHUNKS} chunks; index reports {loaded.metadata.chunk_count}")
        if not isinstance(loaded.retriever, RerankingRetriever):
            raise ValueError("frozen retriever must expose the existing cross-encoder reranker")
        ollama = OllamaGenerator(settings.ollama_model, settings.ollama_url, settings.ollama_timeout)
        ollama.ensure_model_available()
        decomposer = QueryDecomposer(ollama, temperature=0.0)
        retriever = MultiHopRetriever(loaded.retriever, rrf_k=60)
        config = {
            "production_path_changed": False,
            "retrieval_stack": "legacy Qdrant dense + BM25 + RRF + one cross-encoder rerank",
            "conditions": list(CONDITIONS), "candidate_depth": 20, "top_k": 10,
            "rrf_k": 60, "generator_model": ollama.model_name, "temperature": 0.0,
            "maximum_subquestions": 3, "final_reranker_query": "original question",
        }
        if args.regression:
            dataset = REGRESSION_DATASET if args.dataset == DEFAULT_DATASET else args.dataset
            examples = load_evaluation_dataset(dataset)
            report = evaluate_regression(
                examples, decomposer, retriever, dataset_path=dataset, settings=config,
                on_progress=lambda i, n, row: print(f"[{i}/{n}] {row['question_id']}", flush=True),
            )
            json_output = args.json_output or REGRESSION_JSON
            markdown_output = args.markdown_output or REGRESSION_MARKDOWN
        else:
            examples = load_multihop_dataset(args.dataset)
            issues = verify_evidence_passages(examples, read_jsonl(settings.chunks_path))
            if issues:
                raise ValueError(f"dataset evidence validation failed: {issues[:3]}")
            if args.smoke:
                run_smoke(examples[0], decomposer, retriever)
                return 0
            if args.diagnostic is not None:
                examples = examples[: args.diagnostic]
                json_output = args.json_output or DIAGNOSTIC_JSON
                markdown_output = args.markdown_output or DIAGNOSTIC_MARKDOWN
            else:
                json_output = args.json_output or DEFAULT_JSON
                markdown_output = args.markdown_output or DEFAULT_MARKDOWN
            report = evaluate_multihop(
                examples, decomposer, retriever, dataset_path=args.dataset, settings=config,
                on_progress=lambda i, n, row: print(f"[{i}/{n}] {row['question_id']} fallback={row['fallback']}", flush=True),
            )
        write_json(report, json_output)
        write_markdown(report, markdown_output)
        print(f"JSON report: {json_output}\nMarkdown report: {markdown_output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Phase 10G experiment failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if loaded is not None:
            loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
