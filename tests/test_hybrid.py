from rag_research_assistant.hybrid import reciprocal_rank_fusion
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.retrievers import HybridRetriever


def _result(chunk_id: str, score: float, page: int = 1) -> SearchResult:
    chunk = Chunk(chunk_id, f"{chunk_id}.pdf", page, 1, 0, 8, "evidence")
    return SearchResult(score, chunk)


def test_rrf_merges_duplicate_chunks_and_rewards_agreement() -> None:
    a, b, c = _result("a", 0.9), _result("b", 0.8), _result("c", 7.0)

    results = reciprocal_rank_fusion([[a, b], [c, a]], top_k=3, rrf_k=60)

    assert [result.chunk.chunk_id for result in results] == ["a", "c", "b"]
    assert results[0].score == 1 / 61 + 1 / 62


def test_rrf_ignores_duplicate_within_one_ranking_and_has_stable_ties() -> None:
    a, b = _result("a", 1.0), _result("b", 1.0)

    results = reciprocal_rank_fusion([[a, a, b]], top_k=2, rrf_k=60)

    assert [result.chunk.chunk_id for result in results] == ["a", "b"]
    assert results[0].score == 1 / 61


class RecordingRetriever:
    def __init__(self, name, results):
        self.name = name
        self.score_name = name
        self.results = results
        self.depths = []

    def search(self, query, top_k=5):
        self.depths.append(top_k)
        return self.results[:top_k]


def test_hybrid_uses_deeper_candidate_pools_and_preserves_chunk_metadata() -> None:
    a = _result("a", 0.9, page=3)
    b = _result("b", 0.8, page=7)
    dense = RecordingRetriever("dense", [a, b])
    lexical = RecordingRetriever("bm25", [b, a])
    hybrid = HybridRetriever(dense, lexical, candidate_depth=20)

    results = hybrid.search("question", top_k=1)

    assert dense.depths == [20]
    assert lexical.depths == [20]
    assert results[0].chunk.chunk_id == "a"
    assert results[0].chunk.page_number == 3
