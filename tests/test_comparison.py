import numpy as np

from rag_research_assistant.comparison import compare_reports
from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource, evaluate
from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.models import Chunk, SearchResult


class UnusedEmbedder:
    model_name = "test/model"

    def embed_query(self, text):
        raise AssertionError("custom retriever should be used")


class RankedRetriever:
    score_name = "test"

    def __init__(self, name, chunks):
        self.name = name
        self.chunks = chunks

    def search(self, query, top_k=5):
        return [SearchResult(float(10 - rank), chunk) for rank, chunk in enumerate(self.chunks)][:top_k]


def test_comparison_identifies_hybrid_and_reranker_rank_changes() -> None:
    correct = Chunk("correct", "paper.pdf", 1, 1, 0, 7, "correct")
    wrong = [Chunk(f"wrong-{i}", "other.pdf", i, 1, 0, 5, "wrong") for i in range(1, 5)]
    index = EmbeddingIndex(
        np.eye(5, dtype=np.float32), [correct, *wrong], "test/model"
    )
    example = EvaluationExample(
        "question", "question", "answerable", (ExpectedSource("paper.pdf", 1),)
    )
    strategies = [
        RankedRetriever("dense", [wrong[0], wrong[1], correct, wrong[2], wrong[3]]),
        RankedRetriever("bm25", [correct, *wrong]),
        RankedRetriever("hybrid", [wrong[0], correct, wrong[1], wrong[2], wrong[3]]),
        RankedRetriever("hybrid+rerank", [correct, *wrong]),
    ]
    reports = [
        evaluate([example], index, UnusedEmbedder(), retriever=strategy)
        for strategy in strategies
    ]

    comparison = compare_reports(reports)

    assert comparison.comparisons[0].improved[0].from_rank == 3
    assert comparison.comparisons[0].improved[0].to_rank == 2
    assert comparison.comparisons[1].improved[0].to_rank == 1
    assert comparison.comparisons[0].degraded == []
