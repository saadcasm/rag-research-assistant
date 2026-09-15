from rag_research_assistant.models import Chunk, PageText, SearchResult
from rag_research_assistant.parent_child.evaluation import (
    evaluate_parent_results,
    parent_matches_expected,
)
from rag_research_assistant.parent_child.index import build_parent_index
from rag_research_assistant.parent_child.retrieval import (
    ParentChildRetriever,
    aggregate_parent_results,
)
from rag_research_assistant.evaluation import ExpectedSource


def child(
    child_id: str,
    text: str,
    page: int,
    section: str | None = "Methods",
    *,
    end_page: int | None = None,
) -> Chunk:
    return Chunk(
        child_id, "paper.pdf", page, 1, 0, len(text), text,
        page, end_page or page, section, "semantic",
    )


def structural(chunk_id: str, text: str, start: int, end: int = 1) -> Chunk:
    return Chunk(
        chunk_id, "paper.pdf", start, 1, 0, len(text), text,
        start, end, "Methods", "structural",
    )


def fixtures():
    children = [
        child("c1", "alpha evidence", 1),
        child("c2", "beta evidence", 1),
        child("c3", "gamma evidence", 2, "Results"),
    ]
    pages = [
        PageText("paper.pdf", 1, "Methods\nalpha evidence and beta evidence with context"),
        PageText("paper.pdf", 2, "Results\ngamma evidence with more context"),
    ]
    structures = [
        structural("s1", "alpha evidence and beta evidence with context", 1),
    ]
    return children, pages, structures


def test_page_and_structural_mapping_have_stable_ids_and_fallback() -> None:
    children, pages, structures = fixtures()
    first = build_parent_index(children, pages, structures)
    second = build_parent_index(children, pages, structures)

    page_c1 = first.mapping_for("c1", "page")
    page_c2 = first.mapping_for("c2", "page")
    structural_c1 = first.mapping_for("c1", "structural")
    structural_c3 = first.mapping_for("c3", "structural")
    assert page_c1.parent_id == page_c2.parent_id
    assert page_c1.parent_id == second.mapping_for("c1", "page").parent_id
    assert structural_c1.parent_id == second.mapping_for("c1", "structural").parent_id
    assert structural_c1.mapping_status == "structural_exact"
    assert structural_c3.mapping_status == "page_fallback"
    assert structural_c3.warning == "structural_parent_does_not_uniquely_contain_child"
    assert first.structural_exact_mappings == 2
    assert first.structural_fallback_mappings == 1
    assert first.parents[page_c1.parent_id].start_page == 1
    assert first.parents[page_c1.parent_id].end_page == 1


def test_cross_page_child_chooses_page_with_more_text_overlap() -> None:
    cross = child(
        "cross", "second page unique words evidence", 1, "Methods", end_page=2
    )
    pages = [
        PageText("paper.pdf", 1, "first page content"),
        PageText("paper.pdf", 2, "second page unique words evidence and context"),
    ]
    index = build_parent_index([cross], pages, [])
    mapping = index.mapping_for("cross", "page")

    assert index.parents[mapping.parent_id].start_page == 2
    assert mapping.warning == "cross_page_child_mapped_to_highest_overlap_page"
    assert mapping.parent_contains_child_text is True


def test_parent_aggregation_deduplicates_and_rewards_multiple_children() -> None:
    children, pages, structures = fixtures()
    index = build_parent_index(children, pages, structures)
    ranked_children = [
        SearchResult(9.0, children[0]),
        SearchResult(8.0, children[1]),
        SearchResult(7.0, children[2]),
    ]

    parents = aggregate_parent_results(
        ranked_children, index, parent_type="page", top_k=5, rrf_k=0
    )

    assert len(parents) == 2
    assert parents[0].contributing_child_ids == ("c1", "c2")
    assert parents[0].contributing_child_ranks == (1, 2)
    assert parents[0].score == 1.5
    assert parents[0].best_child_rank == 1
    assert parents[0].parent_contains_all_contributing_children is True
    assert parents[0].expansion_ratio == len(parents[0].parent.text) / len("alpha evidence")
    assert parents[0].best_child_start_character_in_parent is not None
    assert parents[0].characters_before_best_child is not None
    assert parents[0].characters_after_best_child is not None


def test_parent_evaluation_uses_provenance_and_checks_child_containment() -> None:
    children, pages, structures = fixtures()
    index = build_parent_index(children, pages, structures)
    ranked_children = [SearchResult(9.0, children[0]), SearchResult(8.0, children[2])]
    parents = aggregate_parent_results(
        ranked_children, index, parent_type="page", top_k=5
    )
    expected = [ExpectedSource("paper.pdf", 1)]

    evaluated = evaluate_parent_results(
        expected, ranked_children, parents, index, parent_type="page"
    )

    assert parent_matches_expected(parents[0].parent, expected[0]) is True
    assert evaluated.first_relevant_rank == 1
    assert evaluated.hit_at_1 is True
    assert evaluated.recall_at_5 == 1.0
    assert evaluated.relevant_retrieved_child_count == 1
    assert evaluated.relevant_children_mapped_into_returned_parents == 1
    assert evaluated.relevant_children_contained_by_returned_parents == 1


class FakeChildRetriever:
    name = "hybrid+rerank"
    score_name = "cross-encoder"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_parent_child_retriever_reranks_children_before_mapping() -> None:
    children, pages, structures = fixtures()
    index = build_parent_index(children, pages, structures)
    child_results = [SearchResult(9.0, children[0]), SearchResult(8.0, children[1])]
    fake = FakeChildRetriever(child_results)

    result = ParentChildRetriever(fake, index, child_depth=20).search(
        "question", parent_top_k=5
    )

    assert fake.calls == [("question", 20)]
    assert result.child_results == tuple(child_results)
    assert len(result.page_parents) == 1
    assert result.page_unique_parents_before_top_k == 1
    assert result.structural_unique_parents_before_top_k == 1
    assert result.parent_mapping_seconds >= 0
    assert result.total_seconds >= result.child_retrieval_seconds
