import json

import pytest

from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.multihop import (
    ExpectedHop,
    HopEvidence,
    MultiHopExample,
    MultiHopRetriever,
    aggregate_hop_coverage,
    evaluate_hop_coverage,
    load_multihop_dataset,
    verify_evidence_passages,
)
from rag_research_assistant.query_transform.decomposition import DecompositionResult
from rag_research_assistant.retrievers import RerankingRetriever


def result(chunk_id: str, document: str, page: int, score: float = 1.0) -> SearchResult:
    return SearchResult(score, Chunk(chunk_id, document, page, 0, 0, 8, f"evidence {chunk_id}"))


class FakeBase:
    name = "fake"
    score_name = "fake"

    def __init__(self, rankings):
        self.rankings = rankings
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        value = self.rankings[query]
        if isinstance(value, Exception):
            raise value
        return value[:top_k]


class FakeReranker:
    model_name = "fake-reranker"

    def __init__(self):
        self.queries = []

    def rerank(self, query, candidates, top_k):
        self.queries.append(query)
        return list(candidates[:top_k])


def decision(*subquestions: str) -> DecompositionResult:
    return DecompositionResult(
        "Compare A and B", True, "{}", tuple(subquestions), tuple(subquestions),
        (), "fake", 0.1, True, False, None,
    )


def example(answerability="answerable") -> MultiHopExample:
    second = ExpectedHop(
        "B", ("fact B",),
        (HopEvidence("b.pdf", (2,), "evidence B"),) if answerability == "answerable" else (),
        answerability == "answerable",
    )
    return MultiHopExample(
        "mh1", "Compare A and B", answerability, "comparison",
        (ExpectedHop("A", ("fact A",), (HopEvidence("a.pdf", (1,), "evidence A"),)), second),
        "draft_verified",
    )


def test_per_query_fusion_deduplication_provenance_and_original_rerank() -> None:
    shared = result("shared", "a.pdf", 1)
    base = FakeBase({
        "Compare A and B": [shared, result("original", "x.pdf", 1)],
        "A evidence?": [shared, result("a", "a.pdf", 1)],
        "B evidence?": [result("b", "b.pdf", 2), shared],
    })
    reranker = FakeReranker()
    retriever = MultiHopRetriever(RerankingRetriever(base, reranker, candidate_depth=4))
    output = retriever.search("Compare A and B", decision("A evidence?", "B evidence?"), top_k=3)

    assert [query for query, _ in base.queries] == ["Compare A and B", "A evidence?", "B evidence?"]
    assert output.always_decompose.deduplicated_occurrences == 1
    assert output.original_plus_decomposed.contributions["shared"] == (
        "Compare A and B", "A evidence?", "B evidence?",
    )
    assert reranker.queries == ["Compare A and B"] * 3


def test_failed_subquery_safely_returns_baseline() -> None:
    baseline = [result("original", "a.pdf", 1)]
    base = FakeBase({"Compare A and B": baseline, "A?": RuntimeError("boom"), "B?": []})
    retriever = MultiHopRetriever(RerankingRetriever(base, FakeReranker(), candidate_depth=4))
    output = retriever.search("Compare A and B", decision("A?", "B?"), top_k=3)
    assert output.fallback is True
    assert output.original_plus_decomposed.results == output.baseline.results
    assert "boom" in output.error


def test_full_and_partial_hop_coverage() -> None:
    values = [result("a", "a.pdf", 1), result("b", "b.pdf", 2)]
    full = evaluate_hop_coverage(example(), values, top_k=2)
    partial = evaluate_hop_coverage(example(), values[:1], top_k=2)
    assert (full.full, full.partial, full.recovered_hops) == (True, 1.0, 2)
    assert (partial.full, partial.partial, partial.recovered_hops) == (False, 0.5, 1)
    summary = aggregate_hop_coverage([example(), example()], [values, values[:1]], top_k=2)
    assert summary == {
        "questions": 2,
        "full_hop_coverage": 0.5,
        "partial_hop_coverage": 0.75,
        "average_hops_recovered": 1.5,
        "zero_hops": 0,
        "one_hop": 1,
        "all_hops": 1,
    }


def test_same_page_chunk_must_contain_hop_passage_not_just_page_label() -> None:
    same_page = MultiHopExample(
        "same", "Two facts", "answerable", "cross_section",
        (
            ExpectedHop("first", (), (HopEvidence("paper.pdf", (1,), "alpha evidence"),)),
            ExpectedHop("second", (), (HopEvidence("paper.pdf", (1,), "beta evidence"),)),
        ),
        "draft",
    )
    only_alpha = SearchResult(1.0, Chunk("alpha", "paper.pdf", 1, 0, 0, 20, "alpha evidence only"))
    coverage = evaluate_hop_coverage(same_page, [only_alpha], top_k=1)
    assert coverage.recovered_hops == 1
    assert coverage.full is False


def test_partially_answerable_missing_hop_cannot_be_counted_as_recovered() -> None:
    coverage = evaluate_hop_coverage(example("partially_answerable"), [result("a", "a.pdf", 1)], top_k=5)
    assert coverage.partial == 0.5
    assert coverage.full is False


def test_evidence_verification_handles_pdf_hyphenation_but_not_wrong_text() -> None:
    chunks = [Chunk("c1", "a.pdf", 1, 0, 0, 30, "Approximate neigh-\nbor retrieval")]
    good = MultiHopExample(
        "mh", "Q", "answerable", "comparison",
        (
            ExpectedHop("A", (), (HopEvidence("a.pdf", (1,), "Approximate neighbor retrieval"),)),
            ExpectedHop("B", (), (HopEvidence("a.pdf", (1,), "neighbor retrieval"),)),
        ),
        "draft",
    )
    assert verify_evidence_passages([good], chunks) == []
    bad_hop = ExpectedHop("B", (), (HopEvidence("a.pdf", (1,), "unrelated claim"),))
    bad = MultiHopExample("bad", "Q", "answerable", "comparison", (good.hops[0], bad_hop), "draft")
    assert verify_evidence_passages([bad], chunks)[0].reason == "passage_not_found"


def test_dataset_loader_validates_and_serializes_auditable_schema(tmp_path) -> None:
    path = tmp_path / "questions.jsonl"
    record = {
        "id": "mh1", "question": "Compare A and B", "answerability": "answerable",
        "category": "cross_paper_comparison", "verification_status": "draft_verified",
        "hops": [
            {"description": "A", "expected_key_facts": ["fact A"], "evidence": [
                {"document": "a.pdf", "pages": [1], "passage": "evidence A"}
            ]},
            {"description": "B", "expected_key_facts": ["fact B"], "evidence": [
                {"document": "b.pdf", "pages": [2], "passage": "evidence B"}
            ]},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_multihop_dataset(path)
    assert loaded[0].hops[1].evidence[0].document == "b.pdf"


def test_loader_rejects_partially_answerable_without_missing_hop(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({
        "id": "bad", "question": "Q", "answerability": "partially_answerable",
        "category": "partial", "verification_status": "draft",
        "hops": [
            {"description": "A", "expected_key_facts": [], "evidence": [{"document": "a.pdf", "pages": [1], "passage": "A"}]},
            {"description": "B", "expected_key_facts": [], "evidence": [{"document": "b.pdf", "pages": [1], "passage": "B"}]},
        ],
    }) + "\n")
    with pytest.raises(ValueError, match="needs a missing hop"):
        load_multihop_dataset(path)
