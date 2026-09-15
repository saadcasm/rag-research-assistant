"""Explicit, serializable parent-child provenance records."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ParentContext:
    parent_id: str
    parent_type: str
    document: str
    start_page: int
    end_page: int
    section_title: Optional[str]
    text: str
    child_ids: Tuple[str, ...]

    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class ChildParentMapping:
    child_id: str
    requested_parent_type: str
    parent_id: str
    mapping_status: str
    warning: Optional[str]
    parent_contains_child_text: bool


@dataclass(frozen=True)
class ParentRetrievalResult:
    score: float
    parent: ParentContext
    contributing_child_ids: Tuple[str, ...]
    contributing_child_ranks: Tuple[int, ...]
    best_child_score: float
    best_child_rank: int
    best_child_character_count: int
    expansion_ratio: float
    best_child_start_character_in_parent: Optional[int]
    characters_before_best_child: Optional[int]
    characters_after_best_child: Optional[int]
    parent_contains_all_contributing_children: bool
    mapping_warnings: Tuple[str, ...]
