"""Child-first retrieval followed by deterministic parent rank aggregation."""

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

from ..models import SearchResult
from ..retrievers import Retriever
from .index import ParentIndex
from .models import ChildParentMapping, ParentRetrievalResult


def aggregate_parent_results(
    children: Sequence[SearchResult],
    index: ParentIndex,
    *,
    parent_type: str,
    top_k: int,
    rrf_k: int = 60,
    mappings: Sequence[ChildParentMapping] | None = None,
) -> List[ParentRetrievalResult]:
    """Deduplicate parents and sum reciprocal evidence from child ranks."""

    if parent_type not in {"page", "structural"}:
        raise ValueError("parent_type must be page or structural")
    if top_k <= 0 or rrf_k < 0:
        raise ValueError("top_k must be positive and rrf_k non-negative")
    grouped: Dict[str, List[Tuple[int, SearchResult, object]]] = {}
    first_seen: Dict[str, int] = {}
    resolved_mappings = list(mappings) if mappings is not None else [
        index.mapping_for(result.chunk.chunk_id, parent_type) for result in children
    ]
    if len(resolved_mappings) != len(children):
        raise ValueError("mappings and children must have the same length")
    for rank, (child_result, mapping) in enumerate(zip(children, resolved_mappings), start=1):
        first_seen.setdefault(mapping.parent_id, len(first_seen))
        grouped.setdefault(mapping.parent_id, []).append((rank, child_result, mapping))
    ranked = []
    for parent_id, contributions in grouped.items():
        parent = index.parents[parent_id]
        score = sum(1.0 / (rrf_k + rank) for rank, _, _ in contributions)
        best_rank, best_result, _ = min(contributions, key=lambda value: value[0])
        best_size = len(best_result.chunk.text)
        best_child_offset = parent.text.find(best_result.chunk.text)
        ranked.append(
            ParentRetrievalResult(
                score=score,
                parent=parent,
                contributing_child_ids=tuple(value.chunk.chunk_id for _, value, _ in contributions),
                contributing_child_ranks=tuple(rank for rank, _, _ in contributions),
                best_child_score=best_result.score,
                best_child_rank=best_rank,
                best_child_character_count=best_size,
                expansion_ratio=(len(parent.text) / best_size if best_size else 0.0),
                best_child_start_character_in_parent=(
                    best_child_offset if best_child_offset >= 0 else None
                ),
                characters_before_best_child=(
                    best_child_offset if best_child_offset >= 0 else None
                ),
                characters_after_best_child=(
                    len(parent.text) - best_child_offset - best_size
                    if best_child_offset >= 0
                    else None
                ),
                parent_contains_all_contributing_children=all(
                    mapping.parent_contains_child_text for _, _, mapping in contributions
                ),
                mapping_warnings=tuple(
                    dict.fromkeys(
                        mapping.warning
                        for _, _, mapping in contributions
                        if mapping.warning
                    )
                ),
            )
        )
    ranked.sort(key=lambda value: (-value.score, value.best_child_rank, first_seen[value.parent.parent_id]))
    return ranked[: min(top_k, len(ranked))]


@dataclass(frozen=True)
class ParentChildSearchResult:
    child_results: Tuple[SearchResult, ...]
    page_parents: Tuple[ParentRetrievalResult, ...]
    structural_parents: Tuple[ParentRetrievalResult, ...]
    child_retrieval_seconds: float
    parent_mapping_seconds: float
    page_aggregation_seconds: float
    structural_aggregation_seconds: float
    page_unique_parents_before_top_k: int
    structural_unique_parents_before_top_k: int

    @property
    def total_seconds(self) -> float:
        return (
            self.child_retrieval_seconds
            + self.parent_mapping_seconds
            + self.page_aggregation_seconds
            + self.structural_aggregation_seconds
        )


class ParentChildRetriever:
    """Retrieve and rerank children first, then expand without another model."""

    def __init__(
        self,
        child_retriever: Retriever,
        parent_index: ParentIndex,
        *,
        child_depth: int = 20,
        parent_rrf_k: int = 60,
    ) -> None:
        if child_depth <= 0:
            raise ValueError("child_depth must be positive")
        self.child_retriever = child_retriever
        self.parent_index = parent_index
        self.child_depth = child_depth
        self.parent_rrf_k = parent_rrf_k

    def search(self, query: str, *, parent_top_k: int = 5) -> ParentChildSearchResult:
        started = perf_counter()
        children = self.child_retriever.search(query, top_k=self.child_depth)
        child_seconds = perf_counter() - started
        mapping_started = perf_counter()
        page_mappings = [
            self.parent_index.mapping_for(result.chunk.chunk_id, "page")
            for result in children
        ]
        structural_mappings = [
            self.parent_index.mapping_for(result.chunk.chunk_id, "structural")
            for result in children
        ]
        mapping_seconds = perf_counter() - mapping_started
        page_started = perf_counter()
        pages = aggregate_parent_results(
            children,
            self.parent_index,
            parent_type="page",
            top_k=parent_top_k,
            rrf_k=self.parent_rrf_k,
            mappings=page_mappings,
        )
        page_seconds = perf_counter() - page_started
        structural_started = perf_counter()
        structural = aggregate_parent_results(
            children,
            self.parent_index,
            parent_type="structural",
            top_k=parent_top_k,
            rrf_k=self.parent_rrf_k,
            mappings=structural_mappings,
        )
        structural_seconds = perf_counter() - structural_started
        return ParentChildSearchResult(
            tuple(children),
            tuple(pages),
            tuple(structural),
            child_seconds,
            mapping_seconds,
            page_seconds,
            structural_seconds,
            len({mapping.parent_id for mapping in page_mappings}),
            len({mapping.parent_id for mapping in structural_mappings}),
        )
