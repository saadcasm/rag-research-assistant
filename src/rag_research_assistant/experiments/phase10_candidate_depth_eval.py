"""Compare true branch/fusion/reranker candidate depths for Phase 10I-2."""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..application import ApplicationSettings, default_retrieval_config, load_retriever
from ..candidate_generation.depth import (
    DEPTHS,
    FINAL_TOP_K,
    RRF_K,
    aggregate_multihop,
    classify_movement,
    evaluate_depth_run,
    run_candidate_depth,
)
from ..evidence_selection.candidate_coverage import sha256_file
from ..evaluation import evaluate_retrieval_results, load_evaluation_dataset
from ..multihop import load_multihop_dataset, verify_evidence_passages
from ..pipeline import read_jsonl
from ..retrievers import HybridRetriever, RerankingRetriever


MULTIHOP_DATASET = Path("data/evaluation/questions-phase-10g-multihop-draft.jsonl")
REGRESSION_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_JSON = Path("data/evaluation/benchmarks/phase-10i-candidate-depth-results.json")
DEFAULT_MARKDOWN = Path("docs/phase-10i-candidate-depth-results.md")
EXPECTED_LEGACY_CHUNKS = 3_793


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _latency_summary(records, depth: int) -> dict[str, Any]:
    rows = [record["conditions"][str(depth)]["latency_seconds"] for record in records]
    fields = ("dense", "bm25", "rrf", "reranker", "total")
    return {
        field: {
            "mean": statistics.fmean(row[field] for row in rows) if rows else 0.0,
            "p50": _percentile([row[field] for row in rows], 0.50),
            "p95": _percentile([row[field] for row in rows], 0.95),
        }
        for field in fields
    }


def evaluate_multihop_depths(examples, runs_by_question) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for example in examples:
        conditions = {
            str(depth): evaluate_depth_run(example, runs_by_question[example.id][depth])
            for depth in DEPTHS
        }
        movements = {
            str(depth): classify_movement(conditions["20"], conditions[str(depth)])
            for depth in (50, 100)
        }
        records.append({
            "question_id": example.id,
            "question": example.question,
            "answerability": example.answerability,
            "category": example.category,
            "conditions": conditions,
            "movements_from_depth_20": movements,
        })
    summary = {"conditions": {}, "movements_from_depth_20": {}}
    for depth in DEPTHS:
        summary["conditions"][str(depth)] = {
            "final_coverage": {
                str(k): aggregate_multihop(records, depth, k) for k in (3, 5, 10)
            },
            "candidate_pool_oracle": _aggregate_oracle(records, depth),
            "latency_seconds": _latency_summary(records, depth),
            "average_candidates_reranked": statistics.fmean(
                record["conditions"][str(depth)]["reranked_candidate_count"]
                for record in records
            ),
        }
    for depth in (50, 100):
        movements = [record["movements_from_depth_20"][str(depth)] for record in records]
        summary["movements_from_depth_20"][str(depth)] = {
            outcome: [
                record["question_id"]
                for record in records
                if record["movements_from_depth_20"][str(depth)]["outcome"] == outcome
            ]
            for outcome in ("improved", "unchanged", "degraded")
        }
        summary["movements_from_depth_20"][str(depth)]["transitions"] = {
            transition: sum(item["transition"] == transition for item in movements)
            for transition in sorted({item["transition"] for item in movements})
        }
    return records, summary


def _aggregate_oracle(records, depth: int) -> dict[str, Any]:
    values = [record["conditions"][str(depth)]["oracle_candidate_coverage"] for record in records]
    total = len(values)
    return {
        "questions": total,
        "full_hop_availability": sum(value["full"] for value in values) / total if total else 0.0,
        "partial_hop_availability": sum(value["partial"] for value in values) / total if total else 0.0,
        "average_recovered_hops": sum(value["recovered_hops"] for value in values) / total if total else 0.0,
        "zero_hop_questions": sum(value["recovered_hops"] == 0 for value in values),
        "one_hop_questions": sum(value["recovered_hops"] == 1 for value in values),
        "all_hop_questions": sum(value["full"] for value in values),
    }


def evaluate_regression_depths(examples, runs_by_question) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    evaluations = {depth: [] for depth in DEPTHS}
    for example in examples:
        conditions = {}
        for depth in DEPTHS:
            run = runs_by_question[example.id][depth]
            evaluated = evaluate_retrieval_results(example, list(run.final))
            evaluations[depth].append(evaluated)
            conditions[str(depth)] = {
                "first_correct_rank": evaluated.first_correct_rank,
                "results": [
                    {"rank": rank, "chunk_id": item.chunk.chunk_id, "document": item.chunk.document,
                     "start_page": item.chunk.start_page, "end_page": item.chunk.end_page, "score": item.score}
                    for rank, item in enumerate(run.final, 1)
                ],
                "latency_seconds": dict(run.latency_seconds),
                "reranked_candidate_count": run.reranked_candidate_count,
            }
        records.append({"question_id": example.id, "question": example.question,
                        "answerability": example.answerability, "conditions": conditions})
    return records, {
        "conditions": {
            str(depth): {**_regression_metrics(evaluations[depth]),
                         "latency_seconds": _regression_latency(records, depth)}
            for depth in DEPTHS
        }
    }


def _regression_metrics(values) -> dict[str, Any]:
    scored = [value for value in values if value.answerability != "unanswerable"]
    def average(field):
        rows = [float(getattr(value, field)) for value in scored if getattr(value, field) is not None]
        return statistics.fmean(rows) if rows else None
    ranks = [float(value.first_correct_rank) for value in scored if value.first_correct_rank is not None]
    return {
        "retrieval_scored_questions": len(scored),
        "unanswerable_questions": sum(value.answerability == "unanswerable" for value in values),
        "hit_at_1": average("hit_at_1"), "hit_at_3": average("hit_at_3"),
        "hit_at_5": average("hit_at_5"),
        "recall_at_1": average("expected_source_recall_at_1"),
        "recall_at_3": average("expected_source_recall_at_3"),
        "recall_at_5": average("expected_source_recall_at_5"),
        "mean_first_correct_rank": statistics.fmean(ranks) if ranks else None,
    }


def _regression_latency(records, depth):
    rows = [record["conditions"][str(depth)]["latency_seconds"] for record in records]
    return {field: statistics.fmean(row[field] for row in rows)
            for field in ("dense", "bm25", "rrf", "reranker", "total")}


def build_report(multihop_records, multihop_summary, regression_records, regression_summary,
                 *, settings, multihop_path, regression_path, chunks_path):
    return {
        "schema_version": 1,
        "experiment": "phase-10i-candidate-depth-comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": dict(settings),
        "provenance": {
            "multihop_dataset": str(multihop_path), "multihop_sha256": sha256_file(multihop_path),
            "regression_dataset": str(regression_path), "regression_sha256": sha256_file(regression_path),
            "chunks": str(chunks_path), "chunks_sha256": sha256_file(chunks_path),
        },
        "summary": {"multihop": multihop_summary, "regression": regression_summary},
        "multihop_questions": multihop_records,
        "regression_questions": regression_records,
    }


def write_outputs(report, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Phase 10I-2: Controlled Candidate-Depth Comparison", "",
             "> Experimental only. Production candidate depth remains unchanged.", "",
             "## Multi-hop final reranked coverage", "",
             "| Depth | Candidates reranked | Full@3 | Full@5 | Full@10 | Partial@3 | Partial@5 | Partial@10 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for depth in DEPTHS:
        row = report["summary"]["multihop"]["conditions"][str(depth)]
        coverage = row["final_coverage"]
        lines.append(f"| {depth} | {row['average_candidates_reranked']:.1f} | " + " | ".join(
            f"{coverage[str(k)][field]:.4f}" for field in ("full_hop_coverage", "partial_hop_coverage") for k in (3, 5, 10)
        ) + " |")
    lines.extend(["", "## Candidate-pool oracle", "", "| Depth | Full | Partial |", "|---:|---:|---:|"])
    for depth in DEPTHS:
        row = report["summary"]["multihop"]["conditions"][str(depth)]["candidate_pool_oracle"]
        lines.append(f"| {depth} | {row['full_hop_availability']:.4f} | {row['partial_hop_availability']:.4f} |")
    lines.extend(["", "## Frozen 100-question regression", "",
                  "| Depth | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MFR |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for depth in DEPTHS:
        row = report["summary"]["regression"]["conditions"][str(depth)]
        lines.append(f"| {depth} | " + " | ".join(
            "n/a" if row[field] is None else f"{row[field]:.4f}"
            for field in ("hit_at_1", "hit_at_3", "hit_at_5", "recall_at_1", "recall_at_3", "recall_at_5", "mean_first_correct_rank")
        ) + " |")
    lines.extend(["", "## Movements and latency", "", "```json",
                  json.dumps({"movements": report["summary"]["multihop"]["movements_from_depth_20"],
                              "multihop_latency": {str(d): report["summary"]["multihop"]["conditions"][str(d)]["latency_seconds"] for d in DEPTHS}}, indent=2),
                  "```", "", "## Interpretation status", "",
                  "No production depth decision is encoded. Review oracle, reranked coverage, regression, question movement, and latency together."])
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_all(examples, hybrid, reranker, progress_label):
    runs = {}
    for index, example in enumerate(examples, 1):
        runs[example.id] = {
            depth: run_candidate_depth(example.question, hybrid.dense, hybrid.lexical, reranker,
                                       depth=depth, final_top_k=FINAL_TOP_K, rrf_k=RRF_K)
            for depth in DEPTHS
        }
        print(f"[{progress_label} {index}/{len(examples)}] {example.id}", flush=True)
    return runs


def run_smoke(example, hybrid, reranker):
    runs = _run_all([example], hybrid, reranker, "smoke")[example.id]
    print(f"Question: {example.question}")
    for depth in DEPTHS:
        row = evaluate_depth_run(example, runs[depth])
        print(f"\ndepth={depth} dense={row['dense_candidate_count']} bm25={row['bm25_candidate_count']} fused={row['fused_candidate_count']} reranked={row['reranked_candidate_count']}")
        print(f"  hops={row['hops']}")
        print(f"  latency={row['latency_seconds']}")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    loaded = None
    try:
        app_settings = ApplicationSettings.from_environment()
        loaded = load_retriever(default_retrieval_config(app_settings))
        if loaded.metadata.chunk_count != EXPECTED_LEGACY_CHUNKS:
            raise ValueError("frozen legacy chunk count mismatch")
        if not isinstance(loaded.retriever, RerankingRetriever) or not isinstance(loaded.retriever.base, HybridRetriever):
            raise ValueError("frozen hybrid+cross-encoder stack is required")
        multihop = load_multihop_dataset(MULTIHOP_DATASET)
        issues = verify_evidence_passages(multihop, read_jsonl(app_settings.chunks_path))
        if issues:
            raise ValueError(f"dataset evidence validation failed: {issues[:3]}")
        hybrid, reranker = loaded.retriever.base, loaded.retriever.reranker
        if args.smoke:
            run_smoke(multihop[0], hybrid, reranker)
            return 0
        multi_runs = _run_all(multihop, hybrid, reranker, "multihop")
        multi_records, multi_summary = evaluate_multihop_depths(multihop, multi_runs)
        regression = load_evaluation_dataset(REGRESSION_DATASET)
        regression_runs = _run_all(regression, hybrid, reranker, "regression")
        regression_records, regression_summary = evaluate_regression_depths(regression, regression_runs)
        settings = {"production_path_changed": False, "depths": list(DEPTHS),
                    "final_top_k": FINAL_TOP_K, "rrf_k": RRF_K,
                    "query_source": "original_questions_only", "ollama_used": False,
                    "embedding_model": loaded.metadata.embedding_model,
                    "reranker_model": loaded.retriever.reranker_model,
                    "chunk_count": loaded.metadata.chunk_count}
        report = build_report(multi_records, multi_summary, regression_records, regression_summary,
                              settings=settings, multihop_path=MULTIHOP_DATASET,
                              regression_path=REGRESSION_DATASET, chunks_path=app_settings.chunks_path)
        write_outputs(report, args.json_output, args.markdown_output)
        print(f"JSON report: {args.json_output}\nMarkdown report: {args.markdown_output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Phase 10I-2 experiment failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if loaded is not None:
            loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
