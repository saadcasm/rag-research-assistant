"""Honest retention proxies and aggregate metrics for contextual compression."""

import json
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .models import CompressionResult


_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SPACE = re.compile(r"\s+")
_QUERY_STOPWORDS = {
    "and", "are", "does", "for", "from", "how", "into", "its", "that",
    "the", "their", "this", "use", "what", "when", "where", "which", "why",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize extraction whitespace/case without changing stored evidence."""

    return _SPACE.sub(" ", text).strip().casefold()


def _terms(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN.finditer(text)}


def _coverage(reference: str, context: str) -> Optional[float]:
    expected = _terms(reference)
    if not expected:
        return None
    return len(expected & _terms(context)) / len(expected)


def term_coverage(reference: str, text: str) -> Optional[float]:
    """Return a transparent lexical-overlap proxy for diagnostic reporting."""

    return _coverage(reference, text)


def load_evidence_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load Phase 7.5 fields omitted by the generic retrieval evaluator."""

    records: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            question_id = value.get("id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"line {line_number}: missing evaluation id")
            if question_id in records:
                raise ValueError(f"line {line_number}: duplicate id {question_id!r}")
            evidence = value.get("evidence", [])
            if not isinstance(evidence, list):
                raise ValueError(f"line {line_number}: evidence must be a list")
            records[question_id] = value
    return records


def _context_for_evidence(result: CompressionResult, evidence: Mapping[str, Any]) -> str:
    document = evidence.get("document")
    pages = evidence.get("pages", [])
    page_values = {page for page in pages if isinstance(page, int)}
    texts = []
    for context in result.contexts:
        chunk = context.source_result.chunk
        page_matches = not page_values or any(
            chunk.start_page <= page <= chunk.end_page for page in page_values
        )
        if chunk.document == document and page_matches:
            texts.append(context.compressed_text)
    return "\n\n".join(texts)


def evaluate_retention(
    question: str,
    evidence: Sequence[Mapping[str, Any]],
    baseline: CompressionResult,
    condition: CompressionResult,
) -> Dict[str, Any]:
    """Measure stored-passage retention without pretending to have answer spans."""

    baseline_coverages: List[float] = []
    condition_coverages: List[float] = []
    baseline_contains = 0
    condition_contains = 0
    usable_evidence = 0
    for item in evidence:
        passage = item.get("passage")
        if not isinstance(passage, str) or not passage.strip():
            continue
        usable_evidence += 1
        original_text = _context_for_evidence(baseline, item)
        compressed_text = _context_for_evidence(condition, item)
        original_coverage = _coverage(passage, original_text)
        compressed_coverage = _coverage(passage, compressed_text)
        if original_coverage is not None:
            baseline_coverages.append(original_coverage)
        if compressed_coverage is not None:
            condition_coverages.append(compressed_coverage)
        normalized_passage = normalize_text(passage)
        baseline_contains += int(
            bool(normalized_passage) and normalized_passage in normalize_text(original_text)
        )
        condition_contains += int(
            bool(normalized_passage) and normalized_passage in normalize_text(compressed_text)
        )

    original_all = "\n\n".join(context.compressed_text for context in baseline.contexts)
    compressed_all = "\n\n".join(context.compressed_text for context in condition.contexts)
    query_terms = {
        term for term in _terms(question)
        if len(term) >= 3 and term not in _QUERY_STOPWORDS
    }
    original_query_hits = query_terms & _terms(original_all)
    compressed_query_hits = query_terms & _terms(compressed_all)
    query_retention = (
        len(compressed_query_hits) / len(original_query_hits)
        if original_query_hits else None
    )
    baseline_mean = (
        sum(baseline_coverages) / len(baseline_coverages)
        if baseline_coverages else None
    )
    condition_mean = (
        sum(condition_coverages) / len(condition_coverages)
        if condition_coverages else None
    )
    conditional_retention = (
        condition_mean / baseline_mean
        if baseline_mean not in (None, 0.0) and condition_mean is not None else None
    )
    return {
        "gold_evidence_count": usable_evidence,
        "gold_passage_contained_original": baseline_contains,
        "gold_passage_contained_compressed": condition_contains,
        "gold_evidence_term_coverage_original": baseline_mean,
        "gold_evidence_term_coverage_compressed": condition_mean,
        "gold_evidence_term_retention_conditional": conditional_retention,
        "query_term_retention_proxy": query_retention,
        "metric_warning": (
            "Term coverage is a proxy over verified supporting passages, not an "
            "answer-span or generation-quality metric. Missing baseline coverage can "
            "reflect retrieval failure."
        ),
    }


def percentile(values: Sequence[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_condition(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate one strategy/budget from persisted per-question records."""

    values = list(records)
    ratios = [float(value["compression_ratio"]) for value in values]
    conditional = [
        float(value["retention"]["gold_evidence_term_retention_conditional"])
        for value in values
        if value["retention"]["gold_evidence_term_retention_conditional"] is not None
    ]
    query_retention = [
        float(value["retention"]["query_term_retention_proxy"])
        for value in values
        if value["retention"]["query_term_retention_proxy"] is not None
    ]
    source_chunks_before = sum(value["source_chunks_before"] for value in values)
    return {
        "questions": len(values),
        "original_context_characters": sum(value["original_context_characters"] for value in values),
        "compressed_context_characters": sum(value["compressed_context_characters"] for value in values),
        "characters_removed": sum(value["characters_removed"] for value in values),
        "mean_compression_ratio": statistics.fmean(ratios) if ratios else 0.0,
        "p50_compression_ratio": percentile(ratios, 0.50),
        "p95_compression_ratio": percentile(ratios, 0.95),
        "segments_before": sum(value["segments_before"] for value in values),
        "segments_retained": sum(value["segments_retained"] for value in values),
        "segments_discarded": sum(value["segments_discarded"] for value in values),
        "average_segments_retained_per_chunk": (
            sum(value["segments_retained"] for value in values) / source_chunks_before
            if source_chunks_before else 0.0
        ),
        "minimum_safeguard_count": sum(value["empty_output_prevention_count"] for value in values),
        "chunks_with_only_one_segment": sum(
            count == 1
            for value in values
            for count in value["source_segment_counts_before"]
        ),
        "omitted_source_count": sum(len(value["omitted_source_ranks"]) for value in values),
        "provenance_integrity_failures": sum(len(value["provenance_failures"]) for value in values),
        "mean_segmentation_seconds": statistics.fmean(value["segmentation_seconds"] for value in values) if values else 0.0,
        "mean_segment_scoring_seconds": statistics.fmean(value["scoring_seconds"] for value in values) if values else 0.0,
        "mean_selection_seconds": statistics.fmean(value["selection_seconds"] for value in values) if values else 0.0,
        "mean_total_compression_seconds": statistics.fmean(value["compression_seconds"] for value in values) if values else 0.0,
        "mean_retrieval_plus_compression_seconds": statistics.fmean(value["retrieval_plus_compression_seconds"] for value in values) if values else 0.0,
        "mean_gold_evidence_term_retention_conditional": statistics.fmean(conditional) if conditional else None,
        "mean_query_term_retention_proxy": statistics.fmean(query_retention) if query_retention else None,
        "gold_passages_contained_original": sum(value["retention"]["gold_passage_contained_original"] for value in values),
        "gold_passages_contained_compressed": sum(value["retention"]["gold_passage_contained_compressed"] for value in values),
    }


def context_to_dict(context) -> Dict[str, Any]:
    """Serialize without duplicating a complete SearchResult dataclass tree."""

    chunk = context.source_result.chunk
    return {
        "source_rank": context.source_rank,
        "retrieval_score": context.source_result.score,
        "source_chunk_id": chunk.chunk_id,
        "document": chunk.document,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "section_title": chunk.section_title,
        "original_text_length": context.original_text_length,
        "compressed_text": context.compressed_text,
        "compressed_text_length": context.compressed_text_length,
        "compression_ratio": context.compression_ratio,
        "total_segment_count": context.total_segment_count,
        "discarded_segment_count": context.discarded_segment_count,
        "minimum_safeguard_used": context.minimum_safeguard_used,
        "selected_segments": [asdict(segment) for segment in context.selected_segments],
    }
