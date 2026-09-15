import pytest

from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.query_transform.multi_query import (
    CorpusGroundedMultiQueryGenerator,
    MultiQueryRetriever,
    parse_query_variants,
)
from rag_research_assistant.retrievers import RerankingRetriever


def result(chunk_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        score,
        Chunk(chunk_id, "paper.pdf", 1, 0, 0, 8, f"text for {chunk_id}"),
    )


class FakeGenerator:
    model_name = "fake"

    def __init__(self, response: str = "", failure: Exception | None = None) -> None:
        self.response = response
        self.failure = failure
        self.prompts = []

    def generate(self, prompt: str, temperature: float) -> str:
        self.prompts.append((prompt, temperature))
        if self.failure:
            raise self.failure
        return self.response


@pytest.mark.parametrize(
    "raw",
    [
        '{"queries": ["first", "second"]}',
        '["first", "second"]',
        '```json\n{"queries": ["first", "second"]}\n```',
    ],
)
def test_multi_query_parsing(raw: str) -> None:
    assert parse_query_variants(raw) == ["first", "second"]


def test_generation_deduplicates_variants_and_original() -> None:
    fake = FakeGenerator(
        '{"queries": ["  Original   Query ", "Useful variant", '
        '"useful VARIANT", "Second variant"]}'
    )
    generated = CorpusGroundedMultiQueryGenerator(fake).generate_queries(
        "Original Query", [], count=3
    )

    assert generated.queries == (
        "Original Query",
        "Useful variant",
        "Second variant",
    )
    assert [item.reason for item in generated.rejected_queries] == [
        "duplicates_original",
        "duplicate_variant",
    ]
    assert generated.shortfall == 1
    assert generated.fallback is False


def test_partial_generation_uses_valid_queries_and_reports_shortfall() -> None:
    generated = CorpusGroundedMultiQueryGenerator(
        FakeGenerator('{"queries": ["one alternative"]}')
    ).generate_queries("original", [], count=3)

    assert generated.valid_queries == ("one alternative",)
    assert generated.queries[0] == "original"
    assert generated.shortfall == 2


@pytest.mark.parametrize(
    "generator",
    [FakeGenerator(""), FakeGenerator(failure=RuntimeError("stopped"))],
)
def test_empty_or_exception_falls_back_to_original(generator: FakeGenerator) -> None:
    generated = CorpusGroundedMultiQueryGenerator(generator).generate_queries(
        "original", [], count=3
    )

    assert generated.queries == ("original",)
    assert generated.valid_queries == ()
    assert generated.fallback is True
    assert generated.error


def test_prompt_is_corpus_grounded_and_requests_json() -> None:
    fake = FakeGenerator('{"queries": ["alternative"]}')
    generated = CorpusGroundedMultiQueryGenerator(fake).generate_queries(
        "How does DPR work?", [result("dpr:p1")], count=2
    )

    prompt, temperature = fake.prompts[0]
    assert '"queries"' in prompt
    assert "How does DPR work?" in prompt
    assert "chunk_id: dpr:p1" in prompt
    assert "Do not answer the question" in prompt
    assert generated.shortfall == 1
    assert temperature == 0.0


class FakeBase:
    name = "hybrid"
    rrf_k = 60
    backend_name = "fake"

    def __init__(self) -> None:
        self.rankings = {
            "original": [result("shared", 9), result("original-only", 8)],
            "alternative": [result("alternative-only", 9), result("shared", 8)],
        }

    def search(self, query: str, top_k: int = 5):
        return self.rankings[query][:top_k]


class RecordingReranker:
    model_name = "fake-reranker"

    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, list(candidates), top_k))
        return list(candidates[:top_k])


def test_multi_query_fusion_deduplicates_rewards_consensus_then_reranks() -> None:
    reranker = RecordingReranker()
    frozen = RerankingRetriever(FakeBase(), reranker, candidate_depth=3)

    retrieved = MultiQueryRetriever(frozen, rrf_k=0).search(
        "original", ["alternative"], top_k=2
    )

    fused_ids = [item.chunk.chunk_id for item in retrieved.fused_candidates]
    assert fused_ids[0] == "shared"
    assert fused_ids.count("shared") == 1
    assert retrieved.deduplicated_occurrences == 1
    assert reranker.calls[0][0] == "original"
    assert [item.chunk.chunk_id for item in reranker.calls[0][1]] == fused_ids
    assert [item.chunk.chunk_id for item in retrieved.final_results] == fused_ids[:2]
