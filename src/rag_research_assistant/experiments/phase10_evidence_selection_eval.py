"""Phase 10H candidate-pool oracle analysis over saved Phase 10G retrievals."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional, Sequence

from ..evidence_selection.candidate_coverage import (
    CandidatePoolEvaluation,
    aggregate_candidate_coverage,
    evaluate_candidate_pool,
    load_json,
    load_phase10g_candidate_pools,
    sha256_file,
)
from ..models import Chunk, SearchResult
from ..multihop import (
    MultiHopExample,
    aggregate_hop_coverage,
    evaluate_hop_coverage,
    load_multihop_dataset,
    verify_evidence_passages,
)
from ..pipeline import read_jsonl


DEFAULT_PHASE10G_ARTIFACT = Path(
    "data/evaluation/benchmarks/phase-10g-multihop-results.json"
)
DEFAULT_DATASET = Path("data/evaluation/questions-phase-10g-multihop-draft.jsonl")
DEFAULT_CHUNKS = Path("data/processed/corpus-52/chunks-legacy.jsonl")
DEFAULT_JSON = Path(
    "data/evaluation/benchmarks/phase-10h-candidate-coverage-results.json"
)
DEFAULT_MARKDOWN = Path("docs/phase-10h-candidate-coverage-results.md")
REQUESTED_DEPTHS = (5, 10, 20, 50)


def _rows_to_results(
    question_id: str, rows: Sequence[Mapping[str, Any]], chunk_by_id: Mapping[str, Chunk]
) -> list[SearchResult]:
    results = []
    for expected_rank, row in enumerate(rows, 1):
        if row.get("rank") != expected_rank:
            raise ValueError(f"{question_id}: final result ranks are not contiguous")
        chunk_id = row.get("chunk_id")
        if chunk_id not in chunk_by_id:
            raise ValueError(f"{question_id}: unknown final-result chunk {chunk_id!r}")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{question_id}: final-result score must be numeric")
        results.append(SearchResult(float(score), chunk_by_id[chunk_id]))
    return results


def _cohort_summary(
    evaluations: Sequence[CandidatePoolEvaluation], depths: Sequence[int]
) -> dict[str, Any]:
    return {
        str(depth): aggregate_candidate_coverage(evaluations, depth)
        for depth in depths
    }


def build_candidate_coverage_report(
    artifact: Mapping[str, Any],
    examples: Sequence[MultiHopExample],
    chunks: Sequence[Chunk],
    *,
    artifact_path: Path,
    dataset_path: Path,
    chunks_path: Path,
    requested_depths: Sequence[int] = REQUESTED_DEPTHS,
) -> dict[str, Any]:
    """Analyze saved pools; no retriever, model, or selector is invoked."""

    started = perf_counter()
    if artifact.get("experiment") != "phase-10g-query-decomposition-multihop-retrieval":
        raise ValueError("input is not the Phase 10G multi-hop artifact")
    if artifact.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Phase 10G artifact dataset checksum does not match current dataset")
    issues = verify_evidence_passages(examples, chunks)
    if issues:
        raise ValueError(f"dataset evidence validation failed: {issues[:3]}")
    pools = load_phase10g_candidate_pools(artifact, chunks)
    examples_by_id = {example.id: example for example in examples}
    if set(examples_by_id) != {pool.question_id for pool in pools}:
        raise ValueError("artifact and dataset question IDs do not match")
    evaluations = [
        evaluate_candidate_pool(
            examples_by_id[pool.question_id], pool, requested_depths=requested_depths
        )
        for pool in pools
    ]

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    artifact_questions = {
        item["question_id"]: item for item in artifact.get("questions", [])
    }
    final_rankings = []
    final_by_id: dict[str, list[SearchResult]] = {}
    for example in examples:
        try:
            rows = artifact_questions[example.id]["conditions"]["baseline"]["final_results"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{example.id}: baseline final ranking is missing") from exc
        final = _rows_to_results(example.id, rows, chunks_by_id)
        final_rankings.append(final)
        final_by_id[example.id] = final

    answerable = [item for item in evaluations if item.answerability == "answerable"]
    partial = [item for item in evaluations if item.answerability == "partially_answerable"]
    categories = sorted({item.category for item in evaluations})
    candidate_counts = sorted({item.candidate_count for item in evaluations})
    available_depths = [
        depth for depth in requested_depths
        if all(item.depths[depth].depth_available for item in evaluations)
    ]
    unavailable_depths = [depth for depth in requested_depths if depth not in available_depths]
    final_summary = aggregate_hop_coverage(examples, final_rankings, top_k=10)
    answerable_examples = [example for example in examples if example.answerability == "answerable"]
    answerable_final = [final_by_id[example.id] for example in answerable_examples]
    answerable_final_summary = aggregate_hop_coverage(
        answerable_examples, answerable_final, top_k=10
    )

    def headroom(cohort: Sequence[CandidatePoolEvaluation], final_full: float) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for depth in requested_depths:
            aggregate = aggregate_candidate_coverage(cohort, depth)
            values[str(depth)] = (
                None
                if aggregate is None
                else aggregate["oracle_full_hop_availability"] - final_full
            )
        return values

    records = []
    for item in evaluations:
        example = examples_by_id[item.question_id]
        final_coverage = evaluate_hop_coverage(
            example, final_by_id[item.question_id], top_k=10
        )
        row = item.to_dict()
        row["final_baseline_top10"] = {
            "chunk_ids": [result.chunk.chunk_id for result in final_by_id[item.question_id]],
            "hop_coverage": asdict(final_coverage),
        }
        records.append(row)

    return {
        "schema_version": 1,
        "experiment": "phase-10h-candidate-pool-coverage",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_phase10g_artifact": str(artifact_path),
        "source_phase10g_artifact_sha256": sha256_file(artifact_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "chunks_path": str(chunks_path),
        "chunks_sha256": sha256_file(chunks_path),
        "settings": {
            "source_ranking": "saved Phase 10G baseline pre-rerank hybrid candidates",
            "retrieval_or_model_rerun": False,
            "gold_labels_used_for": "evaluation_only",
            "requested_depths": list(requested_depths),
            "available_depths": available_depths,
            "unavailable_depths": unavailable_depths,
            "candidate_counts": candidate_counts,
        },
        "summary": {
            "questions": len(evaluations),
            "answerable_questions": len(answerable),
            "partially_answerable_questions": len(partial),
            "current_final_top10": final_summary,
            "current_final_top10_answerable": answerable_final_summary,
            "oracle_candidate_pool": _cohort_summary(evaluations, requested_depths),
            "oracle_answerable": _cohort_summary(answerable, requested_depths),
            "oracle_partially_answerable": _cohort_summary(partial, requested_depths),
            "category_breakdown": {
                category: _cohort_summary(
                    [item for item in evaluations if item.category == category],
                    requested_depths,
                )
                for category in categories
            },
            "full_hop_headroom_vs_final_top10": headroom(
                evaluations, final_summary["full_hop_coverage"]
            ),
            "answerable_full_hop_headroom_vs_final_top10": headroom(
                answerable, answerable_final_summary["full_hop_coverage"]
            ),
            "duplicate_candidate_questions": sum(
                bool(item.duplicate_chunk_ids) for item in evaluations
            ),
            "analysis_seconds": perf_counter() - started,
        },
        "questions": records,
    }


def write_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_metric(value: Any) -> str:
    return "unavailable" if value is None else f"{float(value):.4f}"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    oracle = summary["oracle_candidate_pool"]
    answerable = summary["oracle_answerable"]
    partial = summary["oracle_partially_answerable"]
    lines = [
        "# Phase 10H-1: Candidate-Pool Coverage", "",
        "> Oracle labels are used only after retrieval to measure availability.", "",
        f"Source artifact: `{report['source_phase10g_artifact']}`", "",
        f"Questions: **{summary['questions']}** ({summary['answerable_questions']} answerable, {summary['partially_answerable_questions']} partially answerable)", "",
        "No retrieval or model inference was rerun. Candidate order is the saved Phase 10G baseline hybrid order.", "",
        "## All questions", "",
        "| Depth | Oracle full | Partial | Available-hop full | Available-hop partial |", "|---:|---:|---:|---:|---:|",
    ]
    for depth in report["settings"]["requested_depths"]:
        row = oracle[str(depth)]
        lines.append(
            f"| {depth} | " + " | ".join(
                _format_metric(None if row is None else row[field])
                for field in (
                    "oracle_full_hop_availability", "partial_hop_availability",
                    "oracle_available_hop_full_availability", "available_hop_partial_availability",
                )
            ) + " |"
        )
    lines.extend([
        "", "## Cohorts", "",
        "| Cohort/depth | Oracle full | Partial | Available-hop full | Available-hop partial |", "|---|---:|---:|---:|---:|",
    ])
    for label, values in (("Answerable", answerable), ("Partially answerable", partial)):
        for depth in report["settings"]["requested_depths"]:
            row = values[str(depth)]
            lines.append(
                f"| {label} @{depth} | " + " | ".join(
                    _format_metric(None if row is None else row[field])
                    for field in (
                        "oracle_full_hop_availability", "partial_hop_availability",
                        "oracle_available_hop_full_availability", "available_hop_partial_availability",
                    )
                ) + " |"
            )
    lines.extend([
        "", "## Headroom reference", "",
        f"Current final Full Hop Coverage@10: **{summary['current_final_top10']['full_hop_coverage']:.4f}**", "",
        f"Current answerable-only final Full Hop Coverage@10: **{summary['current_final_top10_answerable']['full_hop_coverage']:.4f}**", "",
        "```json", json.dumps({
            "all_questions": summary["full_hop_headroom_vs_final_top10"],
            "answerable_only": summary["answerable_full_hop_headroom_vs_final_top10"],
        }, indent=2), "```", "",
        "Depth 50 is explicitly unavailable when the saved candidate pool contains only 20 rows.", "",
        "## Decision status", "",
        "This generated report does not use gold labels for ranking and does not automatically encode a Phase 10H-2 decision. See `docs/phase-10h-candidate-coverage.md` for the reviewed project decision.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-coverage", action="store_true", required=True)
    parser.add_argument("--phase10g-artifact", type=Path, default=DEFAULT_PHASE10G_ARTIFACT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact = load_json(args.phase10g_artifact)
        examples = load_multihop_dataset(args.dataset)
        chunks = read_jsonl(args.chunks)
        report = build_candidate_coverage_report(
            artifact, examples, chunks,
            artifact_path=args.phase10g_artifact,
            dataset_path=args.dataset,
            chunks_path=args.chunks,
        )
        write_json(report, args.json_output)
        write_markdown(report, args.markdown_output)
        print(f"JSON report: {args.json_output}\nMarkdown report: {args.markdown_output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Phase 10H-1 analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
