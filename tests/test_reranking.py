from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.reranking import CrossEncoderReranker
from rag_research_assistant.retrievers import RerankingRetriever


def _result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(0.5, Chunk(chunk_id, "paper.pdf", 2, 1, 0, len(text), text))


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.scores


def test_cross_encoder_reranks_pairs_without_loading_a_model() -> None:
    scorer = FakeCrossEncoder([0.1, 0.9, 0.9])
    candidates = [_result("a", "A"), _result("b", "B"), _result("c", "C")]
    reranker = CrossEncoderReranker("fake/model", scorer=scorer)

    results = reranker.rerank("question", candidates, top_k=3)

    assert [result.chunk.chunk_id for result in results] == ["b", "c", "a"]
    assert results[0].score == results[1].score
    assert scorer.calls[0][0] == [("question", "A"), ("question", "B"), ("question", "C")]
    assert scorer.calls[0][1]["show_progress_bar"] is False


class FakeBaseRetriever:
    name = "hybrid"
    score_name = "rrf"

    def __init__(self, candidates):
        self.candidates = candidates
        self.depth = None

    def search(self, query, top_k=5):
        self.depth = top_k
        return self.candidates[:top_k]


def test_reranking_retriever_requests_candidates_then_returns_final_top_k() -> None:
    candidates = [_result("a", "A"), _result("b", "B")]
    base = FakeBaseRetriever(candidates)
    reranker = CrossEncoderReranker("fake/model", scorer=FakeCrossEncoder([0.1, 0.9]))
    retriever = RerankingRetriever(base, reranker, candidate_depth=20)

    results = retriever.search("question", top_k=1)

    assert base.depth == 20
    assert [result.chunk.chunk_id for result in results] == ["b"]
    assert retriever.name == "hybrid+rerank"
