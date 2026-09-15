"""Budget-controlled extractive compression applied after frozen retrieval."""

from dataclasses import dataclass
from time import perf_counter
from typing import List, Protocol, Sequence, Tuple

import numpy as np

from ..models import SearchResult
from .models import CompressedContext, CompressedSegment, CompressionResult, TextSegment
from .segmentation import Segmenter


class SegmentScorer(Protocol):
    """A batched scorer whose values are used only for relative ranking."""

    def score_texts(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class PreparedCompression:
    """Segments and scores computed once and reusable across budget conditions."""

    query: str
    results: Tuple[SearchResult, ...]
    segments_by_source: Tuple[Tuple[CompressedSegment, ...], ...]
    segmentation_seconds: float
    scoring_seconds: float


def _validate_budget_ratio(value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError("budget_ratio must be greater than 0 and at most 1")


def _selection_cost(segment: TextSegment, already_selected: bool) -> int:
    return len(segment.text) + (2 if already_selected else 0)


def _select_under_budget(
    segments: Sequence[CompressedSegment], budget: int
) -> Tuple[set[int], bool]:
    """Greedily select by score, with stable ties and one-segment safeguard."""

    if not segments:
        return set(), False
    ranked = sorted(
        range(len(segments)),
        key=lambda index: (
            -segments[index].relevance_score,
            segments[index].source_rank,
            segments[index].segment_index,
        ),
    )
    anchor = ranked[0]
    selected: set[int] = {anchor}
    used = len(segments[anchor].text)
    safeguard = used > budget
    for index in ranked[1:]:
        cost = _selection_cost(segments[index], bool(selected))
        if used + cost <= budget:
            selected.add(index)
            used += cost
    return selected, safeguard


def _build_context(
    result: SearchResult,
    source_rank: int,
    segments: Sequence[CompressedSegment],
    selected_indexes: set[int],
    safeguard: bool,
) -> CompressedContext:
    selected = tuple(
        segment for index, segment in enumerate(segments) if index in selected_indexes
    )
    return CompressedContext(
        source_result=result,
        source_rank=source_rank,
        original_text_length=len(result.chunk.text),
        compressed_text="\n\n".join(segment.text for segment in selected),
        selected_segments=selected,
        total_segment_count=len(segments),
        minimum_safeguard_used=safeguard,
    )


class ContextualCompressor:
    """Segment, batch-score, and select text without changing source ranking."""

    def __init__(self, scorer: SegmentScorer, *, segmenter: Segmenter | None = None) -> None:
        self.scorer = scorer
        self.segmenter = segmenter or Segmenter()

    def prepare(
        self, query: str, results: Sequence[SearchResult]
    ) -> PreparedCompression:
        if not query.strip():
            raise ValueError("query cannot be empty")
        frozen_results = tuple(results)
        started = perf_counter()
        raw_by_source = tuple(
            tuple(self.segmenter.segment(result, source_rank))
            for source_rank, result in enumerate(frozen_results, start=1)
        )
        segmentation_seconds = perf_counter() - started
        flattened = [segment for values in raw_by_source for segment in values]
        started = perf_counter()
        scores = np.asarray(
            self.scorer.score_texts(query, [segment.text for segment in flattened]),
            dtype=np.float32,
        ).reshape(-1)
        scoring_seconds = perf_counter() - started
        if len(scores) != len(flattened) or not np.isfinite(scores).all():
            raise ValueError("segment scorer returned invalid scores")
        scored_flat = [
            CompressedSegment(**segment.__dict__, relevance_score=float(score))
            for segment, score in zip(flattened, scores)
        ]
        scored_by_source: List[Tuple[CompressedSegment, ...]] = []
        offset = 0
        for values in raw_by_source:
            scored_by_source.append(tuple(scored_flat[offset : offset + len(values)]))
            offset += len(values)
        return PreparedCompression(
            query=query,
            results=frozen_results,
            segments_by_source=tuple(scored_by_source),
            segmentation_seconds=segmentation_seconds,
            scoring_seconds=scoring_seconds,
        )

    def no_compression(self, prepared: PreparedCompression) -> CompressionResult:
        """Represent the exact retrieved chunks as the baseline condition."""

        started = perf_counter()
        contexts = tuple(
            CompressedContext(
                source_result=result,
                source_rank=rank,
                original_text_length=len(result.chunk.text),
                compressed_text=result.chunk.text,
                selected_segments=segments,
                total_segment_count=len(segments),
            )
            for rank, (result, segments) in enumerate(
                zip(prepared.results, prepared.segments_by_source), start=1
            )
        )
        selection_seconds = perf_counter() - started
        original = sum(len(result.chunk.text) for result in prepared.results)
        return CompressionResult(
            strategy="none",
            budget_ratio=1.0,
            contexts=contexts,
            original_context_characters=original,
            compressed_context_characters=original,
            segments_before=sum(len(values) for values in prepared.segments_by_source),
            segments_retained=sum(len(values) for values in prepared.segments_by_source),
            segmentation_seconds=0.0,
            scoring_seconds=0.0,
            selection_seconds=selection_seconds,
            empty_output_prevention_count=0,
            omitted_source_ranks=(),
        )

    def compress_per_chunk(
        self, prepared: PreparedCompression, budget_ratio: float
    ) -> CompressionResult:
        _validate_budget_ratio(budget_ratio)
        started = perf_counter()
        contexts: List[CompressedContext] = []
        safeguards = 0
        for rank, (result, segments) in enumerate(
            zip(prepared.results, prepared.segments_by_source), start=1
        ):
            budget = int(len(result.chunk.text) * budget_ratio)
            selected, safeguard = _select_under_budget(segments, budget)
            safeguards += int(safeguard)
            contexts.append(
                _build_context(result, rank, segments, selected, safeguard)
            )
        selection_seconds = perf_counter() - started
        return self._result(
            "per_chunk", budget_ratio, prepared, contexts, selection_seconds,
            safeguards,
        )

    def compress_global(
        self, prepared: PreparedCompression, budget_ratio: float
    ) -> CompressionResult:
        _validate_budget_ratio(budget_ratio)
        started = perf_counter()
        flattened = [segment for values in prepared.segments_by_source for segment in values]
        original_characters = sum(len(result.chunk.text) for result in prepared.results)
        selected_flat, safeguard = _select_under_budget(
            flattened, int(original_characters * budget_ratio)
        )
        selected_keys = {
            (flattened[index].source_rank, flattened[index].segment_index)
            for index in selected_flat
        }
        contexts: List[CompressedContext] = []
        for rank, (result, segments) in enumerate(
            zip(prepared.results, prepared.segments_by_source), start=1
        ):
            local = {
                index for index, segment in enumerate(segments)
                if (segment.source_rank, segment.segment_index) in selected_keys
            }
            if local:
                contexts.append(
                    _build_context(result, rank, segments, local, safeguard and len(contexts) == 0)
                )
        selection_seconds = perf_counter() - started
        return self._result(
            "global", budget_ratio, prepared, contexts, selection_seconds,
            int(safeguard),
        )

    @staticmethod
    def _result(
        strategy: str,
        budget_ratio: float,
        prepared: PreparedCompression,
        contexts: Sequence[CompressedContext],
        selection_seconds: float,
        safeguards: int,
    ) -> CompressionResult:
        original = sum(len(result.chunk.text) for result in prepared.results)
        compressed = sum(len(context.compressed_text) for context in contexts)
        retained_ranks = {context.source_rank for context in contexts}
        return CompressionResult(
            strategy=strategy,
            budget_ratio=budget_ratio,
            contexts=tuple(contexts),
            original_context_characters=original,
            compressed_context_characters=compressed,
            segments_before=sum(len(values) for values in prepared.segments_by_source),
            segments_retained=sum(len(context.selected_segments) for context in contexts),
            segmentation_seconds=prepared.segmentation_seconds,
            scoring_seconds=prepared.scoring_seconds,
            selection_seconds=selection_seconds,
            empty_output_prevention_count=safeguards,
            omitted_source_ranks=tuple(
                rank for rank in range(1, len(prepared.results) + 1)
                if rank not in retained_ranks
            ),
        )
