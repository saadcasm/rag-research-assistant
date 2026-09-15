from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource
from rag_research_assistant.experiments.phase10_multiquery_eval import (
    evaluate_multi_query,
)
from rag_research_assistant.models import Chunk, CorpusIndexMetadata, SearchResult
from rag_research_assistant.query_transform.multi_query import (
    MultiQueryGenerationResult,
)
from rag_research_assistant.retrievers import RerankingRetriever


def result(document: str) -> SearchResult:
    return SearchResult(
        1.0,
        Chunk(f"{document}:p1", document, 1, 0, 0, 8, "evidence"),
    )


class FakeHybrid:
    name = "hybrid"
    rrf_k = 60
    backend_name = "qdrant"

    def search(self, query: str, top_k: int = 5):
        if query == "original":
            return [result("wrong.pdf")]
        return [result("right.pdf")]


class PreserveOrderReranker:
    model_name = "fake-reranker"

    def rerank(self, query, candidates, top_k):
        assert query == "original"
        return list(candidates[:top_k])


class FakeMultiQueryGenerator:
    model_name = "fake-generator"
    temperature = 0.0
    max_chars_per_chunk = 100

    def generate_queries(self, original_query, preliminary_results, *, count):
        assert original_query == "original"
        assert preliminary_results[0].chunk.document == "wrong.pdf"
        assert count == 3
        return MultiQueryGenerationResult(
            original_query="original",
            requested_count=3,
            generated_queries=("variant one", "variant two", "variant three"),
            valid_queries=("variant one", "variant two", "variant three"),
            rejected_queries=(),
            model_name=self.model_name,
            latency_seconds=0.2,
            fallback=False,
            error=None,
            diagnostics=(),
        )


def test_experiment_compares_nested_conditions_with_frozen_relevance_logic() -> None:
    retriever = RerankingRetriever(
        FakeHybrid(), PreserveOrderReranker(), candidate_depth=1
    )
    report = evaluate_multi_query(
        [
            EvaluationExample(
                "q1",
                "original",
                "answerable",
                (ExpectedSource("right.pdf", 1),),
            )
        ],
        retriever,
        FakeMultiQueryGenerator(),
        CorpusIndexMetadata("embed/model", 384, 3793),
        final_top_k=5,
        dataset_path="frozen.jsonl",
    )

    item = report.questions[0]
    assert item.baseline_first_relevant_rank is None
    assert item.multi_query_2_first_relevant_rank == 1
    assert item.multi_query_3_first_relevant_rank == 1
    assert item.multi_query_2_outcome == "improved"
    assert item.multi_query_3_outcome == "improved"
    assert item.valid_queries == ["variant one", "variant two", "variant three"]
    assert item.multi_query_2_shortfall == 0
    assert item.multi_query_3_shortfall == 0
    assert report.summary.baseline.hit_at_1 == 0.0
    assert report.summary.multi_query_2.hit_at_1 == 1.0
    assert report.summary.multi_query_3.hit_at_1 == 1.0
    assert report.summary.multi_query_2_improved == 1
    assert report.summary.multi_query_3_improved == 1
    assert report.settings["indexed_chunks"] == 3793
    assert report.settings["cross_encoder_query"] == "original_question"
    assert report.dataset_sha256 is None
