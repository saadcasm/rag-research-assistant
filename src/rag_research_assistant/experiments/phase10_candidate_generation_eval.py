"""Diagnose dense, BM25, and RRF candidate recall for Phase 10I-1."""

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..application import (
    ApplicationSettings,
    default_retrieval_config,
    load_retriever,
)
from ..candidate_generation.diagnostics import (
    BRANCHES,
    DEFAULT_DEPTHS,
    ComponentRankings,
    aggregate_component_diagnostics,
    evaluate_component_rankings,
    retrieve_component_rankings,
)
from ..evidence_selection.candidate_coverage import sha256_file
from ..multihop import (
    MultiHopExample,
    load_multihop_dataset,
    verify_evidence_passages,
)
from ..pipeline import read_jsonl
from ..retrievers import HybridRetriever


DEFAULT_DATASET = Path("data/evaluation/questions-phase-10g-multihop-draft.jsonl")
DEFAULT_JSON = Path(
    "data/evaluation/benchmarks/phase-10i-candidate-generation-analysis.json"
)
DEFAULT_MARKDOWN = Path("docs/phase-10i-candidate-generation-analysis.md")
EXPECTED_LEGACY_CHUNKS = 3_793
DEPTHS = DEFAULT_DEPTHS
MAX_DEPTH = max(DEPTHS)
RRF_K = 60


def _result_rows(ranking: Sequence[Any], limit: int | None = None) -> list[dict[str, Any]]:
    values = ranking if limit is None else ranking[:limit]
    return [
        {
            "rank": rank,
            "score": result.score,
            "chunk_id": result.chunk.chunk_id,
            "document": result.chunk.document,
            "start_page": result.chunk.start_page,
            "end_page": result.chunk.end_page,
        }
        for rank, result in enumerate(values, 1)
    ]


def build_candidate_generation_report(
    examples: Sequence[MultiHopExample],
    rankings: Sequence[ComponentRankings],
    *,
    dataset_path: Path,
    chunks_path: Path,
    settings: Mapping[str, Any],
    depths: Sequence[int] = DEPTHS,
) -> dict[str, Any]:
    """Evaluate already-frozen rankings; retrieval remains label-free."""

    if not examples:
        raise ValueError("at least one multi-hop example is required")
    rankings_by_id = {item.question_id: item for item in rankings}
    if len(rankings_by_id) != len(rankings):
        raise ValueError("component rankings contain duplicate question IDs")
    if set(rankings_by_id) != {example.id for example in examples}:
        raise ValueError("component rankings and dataset question IDs do not match")
    records = [
        evaluate_component_rankings(
            example, rankings_by_id[example.id], depths=depths
        )
        for example in examples
    ]
    categories = sorted({record["category"] for record in records})
    answerable = [record for record in records if record["answerability"] == "answerable"]
    partial = [
        record for record in records
        if record["answerability"] == "partially_answerable"
    ]
    return {
        "schema_version": 1,
        "experiment": "phase-10i-component-level-candidate-generation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "chunks_path": str(chunks_path),
        "chunks_sha256": sha256_file(chunks_path),
        "settings": dict(settings),
        "summary": {
            "all_questions": aggregate_component_diagnostics(records, depths=depths),
            "answerable": aggregate_component_diagnostics(answerable, depths=depths),
            "partially_answerable": aggregate_component_diagnostics(
                partial, depths=depths
            ),
            "categories": {
                category: aggregate_component_diagnostics(
                    [record for record in records if record["category"] == category],
                    depths=depths,
                )
                for category in categories
            },
            "latency_seconds": {
                branch: sum(record["latency_seconds"][branch] for record in records)
                for branch in ("dense", "bm25", "rrf", "total")
            },
        },
        "questions": records,
    }


def write_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _metric(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]["all_questions"]
    settings = report["settings"]
    lines = [
        "# Phase 10I-1: Component-Level Candidate-Generation Analysis",
        "",
        "> Gold passages are applied only after dense, BM25, and RRF rankings are frozen.",
        "",
        f"Dataset: `{report['dataset_path']}`",
        "",
        f"Questions: **{summary['questions']}**; available evidence hops: **{summary['available_hops']}**",
        "",
        f"Original questions only; depths: **{', '.join(map(str, settings['depths']))}**; RRF k: **{settings['rrf_k']}**.",
        "",
        "## Passage recall",
        "",
        "| Branch | " + " | ".join(f"R@{depth}" for depth in settings["depths"]) + " |",
        "|---|" + "---:|" * len(settings["depths"]),
    ]
    for branch in BRANCHES:
        values = summary["branches"][branch]["passage_recall"]
        lines.append(
            f"| {branch} | "
            + " | ".join(_metric(values[str(depth)]) for depth in settings["depths"])
            + " |"
        )
    lines.extend([
        "",
        "## Document recall",
        "",
        "| Branch | " + " | ".join(f"R@{depth}" for depth in settings["depths"]) + " |",
        "|---|" + "---:|" * len(settings["depths"]),
    ])
    for branch in BRANCHES:
        values = summary["branches"][branch]["document_recall"]
        lines.append(
            f"| {branch} | "
            + " | ".join(_metric(values[str(depth)]) for depth in settings["depths"])
            + " |"
        )
    lines.extend([
        "",
        "## Correct document found, exact passage missing",
        "",
        "| Branch | " + " | ".join(f"@{depth}" for depth in settings["depths"]) + " |",
        "|---|" + "---:|" * len(settings["depths"]),
    ])
    for branch in BRANCHES:
        values = summary["branches"][branch]["document_found_passage_missing_rate"]
        lines.append(
            f"| {branch} | "
            + " | ".join(_metric(values[str(depth)]) for depth in settings["depths"])
            + " |"
        )
    lines.extend([
        "",
        "## Available-hop question coverage",
        "",
        "Partially answerable questions are scored only against their available hop.",
        "",
        "| Branch/depth | Full | Partial | Zero-hop | One-hop | All-hop |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for branch in BRANCHES:
        for depth in settings["depths"]:
            values = summary["branches"][branch]["question_coverage"][str(depth)]
            lines.append(
                f"| {branch} @{depth} | "
                f"{values['full_hop_availability']:.4f} | "
                f"{values['partial_hop_availability']:.4f} | "
                f"{values['zero_hop_questions']} | {values['one_hop_questions']} | "
                f"{values['all_hop_questions']} |"
            )
    lines.extend([
        "",
        "## Category-level hybrid coverage",
        "",
        "Small categories are descriptive only.",
        "",
        "| Category | Questions | "
        + " | ".join(f"Full@{depth}" for depth in settings["depths"])
        + " |",
        "|---|---:|" + "---:|" * len(settings["depths"]),
    ])
    for category, values in report["summary"]["categories"].items():
        coverage = values["branches"]["hybrid"]["question_coverage"]
        lines.append(
            f"| {category} | {values['questions']} | "
            + " | ".join(
                f"{coverage[str(depth)]['full_hop_availability']:.4f}"
                for depth in settings["depths"]
            )
            + " |"
        )
    lines.extend([
        "",
        "## Diagnostic counts",
        "",
        "```json",
        json.dumps(
            {
                "classification_counts": summary["classification_counts"],
                "flag_counts": summary["flag_counts"],
                "first_passage_rank": {
                    branch: summary["branches"][branch]["first_passage_rank"]
                    for branch in BRANCHES
                },
            },
            indent=2,
        ),
        "```",
        "",
        "Retrieval-only latency totals:",
        "",
        "```json",
        json.dumps(report["summary"]["latency_seconds"], indent=2),
        "```",
        "",
        "## Interpretation status",
        "",
        "This generated report records diagnostics but does not select or implement a retrieval intervention. Review the per-hop records before deciding Phase 10I-2.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_smoke(
    example: MultiHopExample,
    rankings: ComponentRankings,
    *,
    depths: Sequence[int] = DEPTHS,
) -> None:
    record = evaluate_component_rankings(example, rankings, depths=depths)
    print(f"Question: {record['question']}")
    print("Expected hops (evaluation only):")
    for hop in record["hops"]:
        print(
            f"  {hop['hop_index']}. {hop['hop_description']} "
            f"available={hop['evidence_available']}"
        )
        if hop["evidence_available"]:
            print(
                "     passage ranks="
                f"{hop['first_passage_rank']} document ranks={hop['first_document_rank']}"
            )
            print(
                f"     class={hop['diagnostic_classification']} "
                f"flags={hop['diagnostic_flags']}"
            )
    for branch in BRANCHES:
        print(f"\n{branch.upper()} TOP 10:")
        for row in _result_rows(rankings.branch(branch), 10):
            print(
                f"  {row['rank']:>2}. {row['document']} "
                f"p{row['start_page']}-{row['end_page']} {row['chunk_id']} "
                f"score={row['score']:.6f}"
            )
    print(f"\nLatency seconds: {dict(rankings.latency_seconds)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    loaded = None
    try:
        application_settings = ApplicationSettings.from_environment()
        retrieval_config = replace(
            default_retrieval_config(application_settings),
            rerank=False,
            candidate_depth=MAX_DEPTH,
        )
        loaded = load_retriever(retrieval_config)
        if loaded.metadata.chunk_count != EXPECTED_LEGACY_CHUNKS:
            raise ValueError(
                f"frozen legacy corpus requires {EXPECTED_LEGACY_CHUNKS} chunks; "
                f"index reports {loaded.metadata.chunk_count}"
            )
        if not isinstance(loaded.retriever, HybridRetriever):
            raise ValueError("diagnostic requires the existing dense+BM25 hybrid retriever")
        examples = load_multihop_dataset(args.dataset)
        chunks = read_jsonl(application_settings.chunks_path)
        issues = verify_evidence_passages(examples, chunks)
        if issues:
            raise ValueError(f"dataset evidence validation failed: {issues[:3]}")
        selected_examples = examples[:1] if args.smoke else examples
        rankings = []
        for index, example in enumerate(selected_examples, 1):
            ranking = retrieve_component_rankings(
                example.id,
                example.question,
                loaded.retriever.dense,
                loaded.retriever.lexical,
                max_depth=MAX_DEPTH,
                rrf_k=RRF_K,
            )
            rankings.append(ranking)
            print(f"[{index}/{len(selected_examples)}] {example.id}", flush=True)
        if args.smoke:
            run_smoke(selected_examples[0], rankings[0])
            return 0
        settings = {
            "production_path_changed": False,
            "query_source": "original_multi_hop_question_only",
            "retrieval_stack": "legacy Qdrant dense + BM25 + pre-rerank RRF",
            "cross_encoder_used": False,
            "ollama_used": False,
            "gold_labels_used_for": "evaluation_only_after_rankings_frozen",
            "depths": list(DEPTHS),
            "max_retrieval_depth": MAX_DEPTH,
            "rrf_k": RRF_K,
            "chunk_count": loaded.metadata.chunk_count,
            "embedding_model": loaded.metadata.embedding_model,
            "embedding_dimension": loaded.metadata.embedding_dimension,
            "collection_name": application_settings.collection_name,
        }
        report = build_candidate_generation_report(
            examples,
            rankings,
            dataset_path=args.dataset,
            chunks_path=application_settings.chunks_path,
            settings=settings,
        )
        write_json(report, args.json_output)
        write_markdown(report, args.markdown_output)
        print(f"JSON report: {args.json_output}")
        print(f"Markdown report: {args.markdown_output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Phase 10I-1 diagnostic failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if loaded is not None:
            loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
