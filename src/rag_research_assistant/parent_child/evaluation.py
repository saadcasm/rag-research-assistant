"""Parent-level provenance evaluation kept separate from child relevance."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..evaluation import ExpectedSource, expected_source_matches
from ..models import SearchResult
from .index import ParentIndex
from .models import ParentContext, ParentRetrievalResult


@dataclass(frozen=True)
class ParentEvaluation:
    first_relevant_rank: Optional[int]
    hit_at_1: Optional[bool]
    hit_at_3: Optional[bool]
    hit_at_5: Optional[bool]
    recall_at_1: Optional[float]
    recall_at_3: Optional[float]
    recall_at_5: Optional[float]
    relevant_retrieved_child_count: int
    relevant_children_mapped_into_returned_parents: int
    relevant_children_contained_by_returned_parents: int


def parent_matches_expected(parent: ParentContext, expected: ExpectedSource) -> bool:
    """A parent is relevant when its provenance covers the labelled source page."""

    if parent.document != expected.document:
        return False
    if expected.page_number is not None and not (
        parent.start_page <= expected.page_number <= parent.end_page
    ):
        return False
    # Child IDs and parent IDs are different identity domains. A chunk-specific
    # label is validated through contributing children instead of parent ID.
    return True


def evaluate_parent_results(
    expected_sources: Sequence[ExpectedSource],
    child_results: Sequence[SearchResult],
    parent_results: Sequence[ParentRetrievalResult],
    index: ParentIndex,
    *,
    parent_type: str,
    answerability: str = "answerable",
) -> ParentEvaluation:
    """Evaluate parent provenance and evidence preservation without relabeling."""

    if answerability == "unanswerable":
        return ParentEvaluation(None, None, None, None, None, None, None, 0, 0, 0)
    if not expected_sources:
        raise ValueError("answerable parent evaluation requires expected sources")
    matching_parent_ranks = [
        rank
        for rank, result in enumerate(parent_results, start=1)
        if any(parent_matches_expected(result.parent, source) for source in expected_sources)
    ]
    first_rank = min(matching_parent_ranks) if matching_parent_ranks else None
    recalled = {}
    for k in (1, 3, 5):
        matched = {
            source_index
            for source_index, source in enumerate(expected_sources)
            if any(
                parent_matches_expected(result.parent, source)
                for result in parent_results[:k]
            )
        }
        recalled[k] = len(matched) / len(expected_sources)

    returned_ids = {result.parent.parent_id for result in parent_results}
    relevant_children = [
        result
        for result in child_results
        if any(expected_source_matches(source, result) for source in expected_sources)
    ]
    mapped = 0
    contained = 0
    for child in relevant_children:
        mapping = index.mapping_for(child.chunk.chunk_id, parent_type)
        if mapping.parent_id in returned_ids:
            mapped += 1
            if mapping.parent_contains_child_text:
                contained += 1
    return ParentEvaluation(
        first_relevant_rank=first_rank,
        hit_at_1=first_rank is not None and first_rank <= 1,
        hit_at_3=first_rank is not None and first_rank <= 3,
        hit_at_5=first_rank is not None and first_rank <= 5,
        recall_at_1=recalled[1],
        recall_at_3=recalled[3],
        recall_at_5=recalled[5],
        relevant_retrieved_child_count=len(relevant_children),
        relevant_children_mapped_into_returned_parents=mapped,
        relevant_children_contained_by_returned_parents=contained,
    )
