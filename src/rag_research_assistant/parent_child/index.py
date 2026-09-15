"""Build stable page and conservative structural parents for child chunks."""

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from ..chunking import normalize_text
from ..models import Chunk, PageText
from .models import ChildParentMapping, ParentContext


def _stable_parent_id(parent_type: str, identity: str) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]
    return f"parent:{parent_type}:{digest}"


def _normalized_for_containment(text: str) -> str:
    return " ".join(normalize_text(text).split())


def _page_parent_id(document: str, page_number: int) -> str:
    return _stable_parent_id("page", f"{document}\0{page_number}")


def _structural_parent_id(chunk: Chunk) -> str:
    return _stable_parent_id("structural", f"{chunk.document}\0{chunk.chunk_id}")


def _content_words(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 2}


@dataclass(frozen=True)
class ParentIndex:
    parents: Mapping[str, ParentContext]
    page_mappings: Mapping[str, ChildParentMapping]
    structural_mappings: Mapping[str, ChildParentMapping]
    total_children: int
    children_with_section_metadata: int
    structural_exact_mappings: int
    structural_fallback_mappings: int

    def mapping_for(self, child_id: str, parent_type: str) -> ChildParentMapping:
        if parent_type not in {"page", "structural"}:
            raise ValueError("parent_type must be page or structural")
        mappings = self.page_mappings if parent_type == "page" else self.structural_mappings
        try:
            return mappings[child_id]
        except KeyError as exc:
            raise KeyError(f"no {parent_type} parent mapping for child {child_id!r}") from exc

    @property
    def structural_coverage(self) -> float:
        return self.structural_exact_mappings / self.total_children if self.total_children else 0.0


def _choose_page(child: Chunk, pages: Sequence[PageText]) -> Tuple[PageText, str | None]:
    candidates = [page for page in pages if child.start_page <= page.page_number <= child.end_page]
    if not candidates:
        raise ValueError(f"{child.chunk_id}: no physical page covers child span")
    if len(candidates) == 1:
        return candidates[0], None
    words = _content_words(child.text)
    chosen = max(
        candidates,
        key=lambda page: (len(words & _content_words(page.text)), -page.page_number),
    )
    return chosen, "cross_page_child_mapped_to_highest_overlap_page"


def build_parent_index(
    children: Sequence[Chunk],
    pages: Sequence[PageText],
    structural_chunks: Sequence[Chunk],
) -> ParentIndex:
    """Construct parents once; structural mappings require unique containment."""

    child_ids = [child.chunk_id for child in children]
    if len(set(child_ids)) != len(child_ids):
        raise ValueError("child IDs must be unique")
    pages_by_document: Dict[str, List[PageText]] = defaultdict(list)
    for page in pages:
        pages_by_document[page.document].append(page)
    structural_by_document: Dict[str, List[Tuple[Chunk, str]]] = defaultdict(list)
    for chunk in structural_chunks:
        structural_by_document[chunk.document].append(
            (chunk, _normalized_for_containment(chunk.text))
        )

    parent_values: Dict[str, ParentContext] = {}
    page_child_ids: Dict[str, List[str]] = defaultdict(list)
    structural_child_ids: Dict[str, List[str]] = defaultdict(list)
    page_mappings: Dict[str, ChildParentMapping] = {}
    structural_mappings: Dict[str, ChildParentMapping] = {}

    page_lookup: Dict[Tuple[str, int], PageText] = {}
    for document_pages in pages_by_document.values():
        for page in document_pages:
            page_lookup[(page.document, page.page_number)] = page

    for child in children:
        page, page_warning = _choose_page(child, pages_by_document[child.document])
        page_id = _page_parent_id(page.document, page.page_number)
        page_text = normalize_text(page.text)
        page_child_ids[page_id].append(child.chunk_id)
        page_contains = _normalized_for_containment(child.text) in _normalized_for_containment(page_text)
        page_mappings[child.chunk_id] = ChildParentMapping(
            child.chunk_id, "page", page_id, "page", page_warning, page_contains
        )

        child_text = _normalized_for_containment(child.text)
        exact = [
            candidate
            for candidate, candidate_text in structural_by_document[child.document]
            if candidate.section_title == child.section_title
            and child_text in candidate_text
        ]
        if len(exact) == 1 and child.section_title:
            structural = exact[0]
            structural_id = _structural_parent_id(structural)
            structural_child_ids[structural_id].append(child.chunk_id)
            parent_values[structural_id] = ParentContext(
                structural_id,
                "structural",
                structural.document,
                structural.start_page,
                structural.end_page,
                structural.section_title,
                structural.text,
                (),
            )
            structural_mappings[child.chunk_id] = ChildParentMapping(
                child.chunk_id, "structural", structural_id, "structural_exact", None, True
            )
        else:
            reason = (
                "structural_section_missing"
                if not child.section_title
                else "structural_parent_ambiguous"
                if len(exact) > 1
                else "structural_parent_does_not_uniquely_contain_child"
            )
            structural_mappings[child.chunk_id] = ChildParentMapping(
                child.chunk_id,
                "structural",
                page_id,
                "page_fallback",
                reason,
                page_contains,
            )
            page_child_ids[page_id].append(child.chunk_id)

    for (document, page_number), page in page_lookup.items():
        page_id = _page_parent_id(document, page_number)
        if page_id in page_child_ids:
            parent_values[page_id] = ParentContext(
                page_id,
                "page",
                document,
                page_number,
                page_number,
                None,
                normalize_text(page.text),
                tuple(dict.fromkeys(page_child_ids[page_id])),
            )
    for parent_id, child_list in structural_child_ids.items():
        parent = parent_values[parent_id]
        parent_values[parent_id] = ParentContext(
            parent.parent_id,
            parent.parent_type,
            parent.document,
            parent.start_page,
            parent.end_page,
            parent.section_title,
            parent.text,
            tuple(dict.fromkeys(child_list)),
        )
    exact_count = sum(mapping.mapping_status == "structural_exact" for mapping in structural_mappings.values())
    return ParentIndex(
        parents=parent_values,
        page_mappings=page_mappings,
        structural_mappings=structural_mappings,
        total_children=len(children),
        children_with_section_metadata=sum(bool(child.section_title) for child in children),
        structural_exact_mappings=exact_count,
        structural_fallback_mappings=len(children) - exact_count,
    )
