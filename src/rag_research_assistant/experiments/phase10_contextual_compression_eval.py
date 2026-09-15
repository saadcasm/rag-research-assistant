"""Evaluate post-retrieval extractive contextual compression."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..application import (
    ApplicationSettings,
    default_retrieval_config,
    load_retriever,
)
from ..compression import ContextualCompressor
from ..compression.evaluation import (
    context_to_dict,
    evaluate_retention,
    load_evidence_records,
    summarize_condition,
    term_coverage,
)
from ..compression.models import CompressionResult
from ..evaluation import EvaluationExample, load_evaluation_dataset
from ..evaluation import detects_refusal, extract_citations
from ..generation import (
    DEFAULT_TEMPERATURE,
    GenerationError,
    OllamaGenerator,
)
from ..context import build_context, page_display
from ..models import SearchResult
from ..prompting import build_grounded_prompt
from ..reranking import CrossEncoderReranker


DEFAULT_DATASET = Path("data/evaluation/questions-phase-7.5.jsonl")
DEFAULT_JSON_OUTPUT = Path(
    "data/evaluation/benchmarks/phase-10e-contextual-compression-results.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path("docs/phase-10e-contextual-compression-results.md")
DEFAULT_DIAGNOSTIC_JSON = Path(
    "data/evaluation/benchmarks/phase-10e-contextual-compression-diagnostic.json"
)
DEFAULT_DIAGNOSTIC_MARKDOWN = Path(
    "docs/phase-10e-contextual-compression-diagnostic.md"
)
DEFAULT_BUDGETS = (0.75, 0.50, 0.30)
EXPECTED_LEGACY_CHUNKS = 3_793
DEFAULT_GENERATION_JSON = Path(
    "data/evaluation/benchmarks/phase-10e-compression-generation-diagnostic.json"
)
DEFAULT_GENERATION_MARKDOWN = Path(
    "docs/phase-10e-compression-generation-diagnostic.md"
)
GENERATION_QUESTION_IDS = (
    "p75_atlas_few_shot",              # exact number
    "p75_coil_index",                  # difficult distractor
    "p75_doct5query_direction",        # exact terminology / retention stress
    "p75_e5_supervision",              # section detail
    "p75_trec_dl_regime",              # semantic paraphrase / known stress case
    "p75_compare_beir_bright",         # multi-source
    "p75_beir_tradeoff",                # comparison
    "p75_colbert_precompute",           # cross-page / known stress case
    "p75_unanswerable_qdrant_hnsw_m",  # plausible configuration question
    "p75_unanswerable_energy",          # unsupported numerical question
)


def _condition_id(strategy: str, budget_ratio: float) -> str:
    if strategy == "none":
        return "none"
    return f"{strategy}_{round(budget_ratio * 100):02d}pct"


def _validate_provenance(
    result: CompressionResult, retrieved: Sequence[SearchResult]
) -> List[str]:
    failures: List[str] = []
    if result.strategy == "none" and len(result.contexts) != len(retrieved):
        failures.append("baseline source count changed")
    last_rank = 0
    for context in result.contexts:
        rank = context.source_rank
        if rank <= last_rank:
            failures.append(f"source rank order changed at rank {rank}")
        last_rank = rank
        if rank > len(retrieved):
            failures.append(f"unknown source rank {rank}")
            continue
        source = retrieved[rank - 1]
        if context.source_result.chunk.chunk_id != source.chunk.chunk_id:
            failures.append(f"source chunk mismatch at rank {rank}")
        previous_index = -1
        for segment in context.selected_segments:
            if segment.segment_index <= previous_index:
                failures.append(f"segment order changed at source rank {rank}")
            previous_index = segment.segment_index
            if segment.source_chunk_id != source.chunk.chunk_id:
                failures.append(f"segment chunk mismatch at source rank {rank}")
            if source.chunk.text[segment.start_char : segment.end_char] != segment.text:
                failures.append(
                    f"segment offset/text mismatch at source rank {rank} "
                    f"segment {segment.segment_index}"
                )
            if (
                segment.document != source.chunk.document
                or segment.start_page != source.chunk.start_page
                or segment.end_page != source.chunk.end_page
                or segment.section_title != source.chunk.section_title
            ):
                failures.append(
                    f"segment source metadata mismatch at source rank {rank} "
                    f"segment {segment.segment_index}"
                )
    return failures


def _condition_record(
    result: CompressionResult,
    baseline: CompressionResult,
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    retrieval_seconds: float,
) -> Dict[str, Any]:
    provenance_failures = _validate_provenance(result, [
        context.source_result for context in baseline.contexts
    ])
    warnings = []
    if result.empty_output_prevention_count:
        warnings.append(
            f"minimum-evidence safeguard used {result.empty_output_prevention_count} time(s)"
        )
    if result.omitted_source_ranks:
        warnings.append(
            f"global selection omitted source ranks {list(result.omitted_source_ranks)}"
        )
    warnings.extend(provenance_failures)
    return {
        "condition_id": _condition_id(result.strategy, result.budget_ratio),
        "strategy": result.strategy,
        "budget_ratio": result.budget_ratio,
        "budget_unit": "source-text characters",
        "original_context_characters": result.original_context_characters,
        "compressed_context_characters": result.compressed_context_characters,
        "compression_ratio": result.compression_ratio,
        "characters_removed": result.characters_removed,
        "segments_before": result.segments_before,
        "segments_retained": result.segments_retained,
        "segments_discarded": result.segments_discarded,
        "source_chunks_before": len(baseline.contexts),
        "source_segment_counts_before": [
            context.total_segment_count for context in baseline.contexts
        ],
        "segmentation_seconds": result.segmentation_seconds,
        "scoring_seconds": result.scoring_seconds,
        "selection_seconds": result.selection_seconds,
        "compression_seconds": result.compression_seconds,
        "retrieval_plus_compression_seconds": retrieval_seconds + result.compression_seconds,
        "empty_output_prevention_count": result.empty_output_prevention_count,
        "omitted_source_ranks": list(result.omitted_source_ranks),
        "provenance_failures": provenance_failures,
        "warnings": warnings,
        "retention": evaluate_retention(
            question, evidence, baseline, result
        ),
        "contexts": [context_to_dict(context) for context in result.contexts],
    }


def evaluate_experiment(
    examples: Sequence[EvaluationExample],
    evidence_records: Mapping[str, Mapping[str, Any]],
    retriever,
    compressor: ContextualCompressor,
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    dataset_path: str = "",
    configuration: Optional[Dict[str, Any]] = None,
    on_progress=None,
) -> Dict[str, Any]:
    """Retrieve once per question, then compare compression-only conditions."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    for budget in budgets:
        if not 0.0 < budget <= 1.0:
            raise ValueError("budgets must be greater than 0 and at most 1")
    question_records: List[Dict[str, Any]] = []
    for example in examples:
        started = perf_counter()
        retrieved = retriever.search(example.question, top_k=5)
        retrieval_seconds = perf_counter() - started
        prepared = compressor.prepare(example.question, retrieved)
        baseline = compressor.no_compression(prepared)
        conditions = [baseline]
        for budget in budgets:
            conditions.append(compressor.compress_per_chunk(prepared, budget))
        for budget in budgets:
            conditions.append(compressor.compress_global(prepared, budget))
        raw = evidence_records.get(example.id)
        if raw is None:
            raise ValueError(f"missing raw evaluation record for {example.id}")
        evidence = raw.get("evidence", [])
        records = [
            _condition_record(
                condition, baseline, example.question, evidence, retrieval_seconds
            )
            for condition in conditions
        ]
        retrieved_ids = [result.chunk.chunk_id for result in retrieved]
        baseline_ids = [
            context.source_result.chunk.chunk_id for context in baseline.contexts
        ]
        if retrieved_ids != baseline_ids:
            raise ValueError("compression changed frozen retrieval ranking")
        question_record = {
            "question_id": example.id,
            "question": example.question,
            "answerability": example.answerability,
            "category": raw.get("category"),
            "difficulty": raw.get("difficulty"),
            "expected_sources": [asdict(value) for value in example.expected_sources],
            "evidence": evidence,
            "retrieval_seconds": retrieval_seconds,
            "retrieved_chunks": [
                {
                    "rank": rank,
                    "score": result.score,
                    "chunk_id": result.chunk.chunk_id,
                    "document": result.chunk.document,
                    "start_page": result.chunk.start_page,
                    "end_page": result.chunk.end_page,
                    "section_title": result.chunk.section_title,
                    "text_length": len(result.chunk.text),
                }
                for rank, result in enumerate(retrieved, start=1)
            ],
            "shared_segmentation_seconds": prepared.segmentation_seconds,
            "shared_batch_scoring_seconds": prepared.scoring_seconds,
            "conditions": records,
        }
        question_records.append(question_record)
        if on_progress:
            on_progress(len(question_records), len(examples), question_record)

    condition_ids = [
        condition["condition_id"]
        for condition in question_records[0]["conditions"]
    ]
    summaries = {
        condition_id: summarize_condition(
            condition
            for question in question_records
            for condition in question["conditions"]
            if condition["condition_id"] == condition_id
        )
        for condition_id in condition_ids
    }
    return {
        "schema_version": 1,
        "experiment": "phase-10e-extractive-contextual-compression",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_sha256": (
            hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
            if dataset_path and Path(dataset_path).is_file() else None
        ),
        "configuration": configuration or {},
        "metric_notes": {
            "retrieval": "Frozen top-five chunk IDs and order; compression is not rescored as retrieval.",
            "cross_encoder": "Scores rank segments and are not calibrated probabilities.",
            "evidence": "Verified supporting-passage containment and term coverage are retention proxies, not generation quality.",
            "latency": "Segments are batch-scored once per question and reused across budget conditions; each condition reports the scoring cost it would incur alone.",
        },
        "summary": {
            "total_questions": len(question_records),
            "answerable_questions": sum(
                example.answerability != "unanswerable" for example in examples
            ),
            "unanswerable_questions": sum(
                example.answerability == "unanswerable" for example in examples
            ),
            "conditions": summaries,
        },
        "questions": question_records,
    }


def write_json(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _display(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Phase 10E: contextual compression results", "",
        "This is a post-retrieval experiment. Every condition uses the same frozen top-five legacy chunks in the same order. Evidence retention is measured with verified supporting-passage proxies; it is not answer-generation quality.", "",
        f"Questions: {summary['total_questions']} ({summary['answerable_questions']} answerable; {summary['unanswerable_questions']} unanswerable)", "",
        "| Condition | Mean ratio | p50 | p95 | Evidence term retention | Query-term retention | Mean compression latency | Omitted sources | Provenance failures |", 
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id, values in summary["conditions"].items():
        lines.append(
            f"| {condition_id} | {_display(values['mean_compression_ratio'])} | "
            f"{_display(values['p50_compression_ratio'])} | {_display(values['p95_compression_ratio'])} | "
            f"{_display(values['mean_gold_evidence_term_retention_conditional'])} | "
            f"{_display(values['mean_query_term_retention_proxy'])} | "
            f"{_display(values['mean_total_compression_seconds'], 3)}s | "
            f"{values['omitted_source_count']} | {values['provenance_integrity_failures']} |"
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "These results quantify size, latency, and evidence-retention proxies only. Select a candidate after inspecting per-question failures, then compare it with uncompressed context in the separately controlled, small generation diagnostic.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compressed_prompt_context(result: CompressionResult) -> tuple[str, List[Dict[str, Any]]]:
    """Format compressed text with original citation numbers and provenance."""

    entries: List[str] = []
    sources: List[Dict[str, Any]] = []
    for context in result.contexts:
        chunk = context.source_result.chunk
        entries.append(
            "\n".join(
                [
                    f"SOURCE [{context.source_rank}]",
                    f"filename: {chunk.document}",
                    f"page: {page_display(chunk.start_page, chunk.end_page)}",
                    f"section: {chunk.section_title or 'not detected'}",
                    f"chunk_id: {chunk.chunk_id}",
                    "text:",
                    context.compressed_text,
                ]
            )
        )
        sources.append(
            {
                "citation_number": context.source_rank,
                "retrieval_score": context.source_result.score,
                "document": chunk.document,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "chunk_id": chunk.chunk_id,
            }
        )
    return "\n\n---\n\n".join(entries), sources


def _answer_evidence_coverage(
    answer: Optional[str], evidence: Sequence[Mapping[str, Any]]
) -> Optional[float]:
    if not answer:
        return None
    coverages = [
        term_coverage(item["passage"], answer)
        for item in evidence
        if isinstance(item.get("passage"), str) and item["passage"].strip()
    ]
    usable = [value for value in coverages if value is not None]
    return sum(usable) / len(usable) if usable else None


def _generate_once(
    generator,
    question: str,
    context: str,
    sources: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    *,
    source_text_characters: int,
    temperature: float,
) -> Dict[str, Any]:
    started = perf_counter()
    try:
        answer = generator.generate(
            build_grounded_prompt(question, context), temperature=temperature
        )
        error = None
    except (GenerationError, ValueError) as exc:
        answer = None
        error = str(exc)
    latency = perf_counter() - started
    citations = extract_citations(answer) if answer else []
    source_by_citation = {
        int(source["citation_number"]): source for source in sources
    }
    invalid = [number for number in citations if number not in source_by_citation]
    mapped = [source_by_citation[number] for number in citations if number in source_by_citation]
    return {
        "answer": answer,
        "sources": list(sources),
        "detected_citations": citations,
        "invalid_citations": invalid,
        "mapped_citations": mapped,
        "refusal_detected": detects_refusal(answer) if answer else None,
        "source_text_characters": source_text_characters,
        "formatted_context_characters": len(context),
        "generation_seconds": latency,
        "expected_evidence_term_coverage_in_answer_proxy": _answer_evidence_coverage(
            answer, evidence
        ),
        "error": error,
    }


def run_generation_diagnostic(
    examples: Sequence[EvaluationExample],
    evidence_records: Mapping[str, Mapping[str, Any]],
    retriever,
    compressor: ContextualCompressor,
    generator,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    on_progress=None,
) -> Dict[str, Any]:
    """Compare identical retrieval with original versus global-75% context."""

    by_id = {example.id: example for example in examples}
    missing = [question_id for question_id in GENERATION_QUESTION_IDS if question_id not in by_id]
    if missing:
        raise ValueError(f"generation diagnostic question IDs missing: {missing}")
    records = []
    for question_id in GENERATION_QUESTION_IDS:
        example = by_id[question_id]
        raw = evidence_records[question_id]
        evidence = raw.get("evidence", [])
        retrieval_started = perf_counter()
        retrieved = retriever.search(example.question, top_k=5)
        retrieval_seconds = perf_counter() - retrieval_started
        prepared = compressor.prepare(example.question, retrieved)
        baseline = compressor.no_compression(prepared)
        compressed = compressor.compress_global(prepared, 0.75)
        baseline_context, baseline_sources_dataclasses = build_context(retrieved)
        baseline_sources = [
            {
                "citation_number": source.citation_number,
                "retrieval_score": source.score,
                "document": source.document,
                "start_page": source.page_number,
                "end_page": source.end_page,
                "chunk_id": source.chunk_id,
            }
            for source in baseline_sources_dataclasses
        ]
        compressed_context, compressed_sources = _compressed_prompt_context(compressed)
        baseline_answer = _generate_once(
            generator, example.question, baseline_context, baseline_sources, evidence,
            source_text_characters=baseline.original_context_characters,
            temperature=temperature,
        )
        compressed_answer = _generate_once(
            generator, example.question, compressed_context, compressed_sources, evidence,
            source_text_characters=compressed.compressed_context_characters,
            temperature=temperature,
        )
        record = {
            "question_id": question_id,
            "question": example.question,
            "answerability": example.answerability,
            "category": raw.get("category"),
            "difficulty": raw.get("difficulty"),
            "selection_role": (
                "unanswerable abstention check"
                if example.answerability == "unanswerable"
                else "category coverage and compression stress case"
            ),
            "expected_sources": [asdict(source) for source in example.expected_sources],
            "evidence": evidence,
            "retrieval_seconds": retrieval_seconds,
            "retrieved_chunk_ids": [result.chunk.chunk_id for result in retrieved],
            "compression": {
                "strategy": compressed.strategy,
                "budget_ratio": compressed.budget_ratio,
                "actual_compression_ratio": compressed.compression_ratio,
                "segmentation_seconds": compressed.segmentation_seconds,
                "scoring_seconds": compressed.scoring_seconds,
                "selection_seconds": compressed.selection_seconds,
                "omitted_source_ranks": list(compressed.omitted_source_ranks),
                "provenance_failures": _validate_provenance(compressed, retrieved),
                "contexts": [context_to_dict(context) for context in compressed.contexts],
            },
            "original": baseline_answer,
            "compressed": compressed_answer,
        }
        records.append(record)
        if on_progress:
            on_progress(len(records), len(GENERATION_QUESTION_IDS), record)
    return {
        "schema_version": 1,
        "experiment": "phase-10e-global-75-generation-diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "question_ids": list(GENERATION_QUESTION_IDS),
            "compression_strategy": "global",
            "budget_ratio": 0.75,
            "generator_model": generator.model_name,
            "temperature": temperature,
            "prompt": "identical grounded prompt template; context text is the only intended difference",
            "evaluation_warning": "Answer evidence-term overlap is a lexical proxy, not correctness or groundedness.",
        },
        "summary": {
            "questions": len(records),
            "answerable": sum(value["answerability"] != "unanswerable" for value in records),
            "unanswerable": sum(value["answerability"] == "unanswerable" for value in records),
            "original_generation_errors": sum(value["original"]["error"] is not None for value in records),
            "compressed_generation_errors": sum(value["compressed"]["error"] is not None for value in records),
            "original_invalid_citations": sum(len(value["original"]["invalid_citations"]) for value in records),
            "compressed_invalid_citations": sum(len(value["compressed"]["invalid_citations"]) for value in records),
            "original_unanswerable_refusals": sum(
                value["original"]["refusal_detected"] is True
                for value in records if value["answerability"] == "unanswerable"
            ),
            "compressed_unanswerable_refusals": sum(
                value["compressed"]["refusal_detected"] is True
                for value in records if value["answerability"] == "unanswerable"
            ),
            "original_total_source_characters": sum(value["original"]["source_text_characters"] for value in records),
            "compressed_total_source_characters": sum(value["compressed"]["source_text_characters"] for value in records),
            "original_total_generation_seconds": sum(value["original"]["generation_seconds"] for value in records),
            "compressed_total_generation_seconds": sum(value["compressed"]["generation_seconds"] for value in records),
            "provenance_failures": sum(len(value["compression"]["provenance_failures"]) for value in records),
        },
        "questions": records,
    }


def write_generation_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = report["summary"]
    original_chars = summary["original_total_source_characters"]
    compressed_chars = summary["compressed_total_source_characters"]
    ratio = compressed_chars / original_chars if original_chars else 0.0
    lines = [
        "# Phase 10E: compression generation diagnostic", "",
        "This controlled diagnostic compares the exact same retrieved source ranking and grounded prompt with original context versus global 75% extractive compression. It is a small qualitative study, not a statistically powered answer-quality benchmark.", "",
        f"- Questions: {summary['questions']} ({summary['answerable']} answerable; {summary['unanswerable']} unanswerable)",
        f"- Source characters, original/compressed: {original_chars}/{compressed_chars} (ratio {ratio:.4f})",
        f"- Generation time, original/compressed: {summary['original_total_generation_seconds']:.3f}s/{summary['compressed_total_generation_seconds']:.3f}s",
        f"- Generation errors, original/compressed: {summary['original_generation_errors']}/{summary['compressed_generation_errors']}",
        f"- Invalid citations, original/compressed: {summary['original_invalid_citations']}/{summary['compressed_invalid_citations']}",
        f"- Unanswerable refusals, original/compressed: {summary['original_unanswerable_refusals']}/{summary['compressed_unanswerable_refusals']}",
        f"- Provenance failures: {summary['provenance_failures']}", "",
        "## Per-question answers", "",
    ]
    for value in report["questions"]:
        lines.extend([
            f"### {value['question_id']}", "",
            f"**Question:** {value['question']}", "",
            f"Category: {value['category']}; answerability: {value['answerability']}; actual compression ratio: {value['compression']['actual_compression_ratio']:.4f}", "",
            "**Original answer**", "",
            value["original"]["answer"] or f"ERROR: {value['original']['error']}", "",
            "**Compressed answer**", "",
            value["compressed"]["answer"] or f"ERROR: {value['compressed']['error']}", "",
        ])
    lines.extend([
        "## Review boundary", "",
        "Manually assess factual correctness, citation support, clarity, lost multi-sentence dependencies, numerical/entity preservation, and whether either condition should have abstained. Automated citation/refusal/term-overlap fields are diagnostics, not a substitute for that review.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_smoke(
    example: EvaluationExample,
    retrieved: Sequence[SearchResult],
    compressor: ContextualCompressor,
) -> None:
    print("QUESTION\n" + example.question)
    started = perf_counter()
    prepared = compressor.prepare(example.question, retrieved)
    result = compressor.compress_per_chunk(prepared, 0.50)
    print("\nRETRIEVED CHUNKS AND SEGMENTS")
    for rank, (source, segments) in enumerate(
        zip(retrieved, prepared.segments_by_source), start=1
    ):
        chunk = source.chunk
        print(
            f"\nSOURCE {rank}: {chunk.document} p{chunk.start_page}-{chunk.end_page} "
            f"chunk={chunk.chunk_id} retrieval_score={source.score:.4f}"
        )
        print("ORIGINAL TEXT:\n" + chunk.text)
        print("SEGMENTED UNITS:")
        for segment in segments:
            print(
                f"  [{segment.segment_index}] chars={segment.start_char}:{segment.end_char} "
                f"score={segment.relevance_score:.4f} text={segment.text!r}"
            )
    print("\nSELECTED / COMPRESSED CONTEXT (per-chunk 50%)")
    for context in result.contexts:
        chunk = context.source_result.chunk
        print(
            f"\nSOURCE {context.source_rank}: {chunk.document} "
            f"p{chunk.start_page}-{chunk.end_page} chunk={chunk.chunk_id} "
            f"section={chunk.section_title!r}"
        )
        print("selected_segments=" + repr([segment.segment_index for segment in context.selected_segments]))
        print(context.compressed_text)
    print(
        f"\noriginal_chars={result.original_context_characters} "
        f"compressed_chars={result.compressed_context_characters} "
        f"ratio={result.compression_ratio:.4f} "
        f"segmentation={prepared.segmentation_seconds:.4f}s "
        f"scoring={prepared.scoring_seconds:.4f}s "
        f"selection={result.selection_seconds:.4f}s "
        f"total_wall={perf_counter() - started:.4f}s"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--diagnostic", type=int, metavar="COUNT")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--generation-diagnostic", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--question", help="override smoke question; default is dataset row 1")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def _loaded_cross_encoder(loaded) -> CrossEncoderReranker:
    reranker = getattr(loaded.retriever, "reranker", None)
    if not isinstance(reranker, CrossEncoderReranker):
        raise ValueError("frozen retriever does not expose the expected cross-encoder")
    return reranker


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.diagnostic is not None and args.diagnostic <= 0:
        print("Error: --diagnostic must be positive", file=sys.stderr)
        return 2
    loaded = None
    try:
        examples = load_evaluation_dataset(args.dataset)
        evidence_records = load_evidence_records(args.dataset)
        settings = ApplicationSettings.from_environment()
        loaded = load_retriever(default_retrieval_config(settings))
        if loaded.metadata.chunk_count != EXPECTED_LEGACY_CHUNKS:
            raise ValueError(
                f"frozen legacy corpus requires {EXPECTED_LEGACY_CHUNKS} chunks; "
                f"index reports {loaded.metadata.chunk_count}"
            )
        compressor = ContextualCompressor(_loaded_cross_encoder(loaded))
        if args.smoke:
            example = examples[0]
            if args.question:
                example = EvaluationExample(
                    id="smoke_override", question=args.question,
                    answerability="unanswerable", expected_sources=(),
                )
            retrieved = loaded.retriever.search(example.question, top_k=5)
            _print_smoke(example, retrieved, compressor)
            return 0
        if args.generation_diagnostic:
            generator = OllamaGenerator(
                model_name=settings.ollama_model,
                base_url=settings.ollama_url,
                timeout=settings.ollama_timeout,
            )
            generator.ensure_model_available()
            report = run_generation_diagnostic(
                examples,
                evidence_records,
                loaded.retriever,
                compressor,
                generator,
                temperature=DEFAULT_TEMPERATURE,
                on_progress=lambda current, total, value: print(
                    f"[{current}/{total}] {value['question_id']}", flush=True
                ),
            )
            json_output = args.json_output or DEFAULT_GENERATION_JSON
            markdown_output = args.markdown_output or DEFAULT_GENERATION_MARKDOWN
            write_json(report, json_output)
            write_generation_markdown(report, markdown_output)
            print(f"JSON report: {json_output}\nMarkdown report: {markdown_output}")
            return 0
        if args.diagnostic is not None:
            examples = examples[:args.diagnostic]
            json_output = args.json_output or DEFAULT_DIAGNOSTIC_JSON
            markdown_output = args.markdown_output or DEFAULT_DIAGNOSTIC_MARKDOWN
        else:
            json_output = args.json_output or DEFAULT_JSON_OUTPUT
            markdown_output = args.markdown_output or DEFAULT_MARKDOWN_OUTPUT
        report = evaluate_experiment(
            examples,
            evidence_records,
            loaded.retriever,
            compressor,
            budgets=args.budgets,
            dataset_path=str(args.dataset),
            configuration={
                "retrieval_stack": "legacy Qdrant dense + BM25 + RRF + cross-encoder",
                "retrieval_top_k": 5,
                "candidate_depth": 20,
                "embedding_model": loaded.metadata.embedding_model,
                "reranker_model": getattr(loaded.retriever, "reranker_model", None),
                "chunk_count": loaded.metadata.chunk_count,
                "segmentation": {
                    "unit": "sentence groups",
                    "minimum_characters": compressor.segmenter.min_characters,
                    "target_characters": compressor.segmenter.target_characters,
                },
                "budgets": list(args.budgets),
                "budget_unit": "source-text characters",
                "segment_scoring": "one cross-encoder batch per question, reused across conditions",
            },
            on_progress=lambda current, total, value: print(
                f"[{current}/{total}] {value['question_id']}", flush=True
            ),
        )
        write_json(report, json_output)
        write_markdown(report, markdown_output)
        print(f"JSON report: {json_output}\nMarkdown report: {markdown_output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: Phase 10E experiment failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if loaded is not None:
            loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())
