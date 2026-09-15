from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource
from rag_research_assistant.experiments.phase10_rewrite_eval import (
    classify_rank_change,
    evaluate_rewriting,
)
from rag_research_assistant.models import Chunk, CorpusIndexMetadata, SearchResult
from rag_research_assistant.query_transform.rewriting import (
    IdentityDiagnostics,
    RewriteResult,
)


def result(document: str, page: int, score: float = 1.0) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=Chunk(
            chunk_id=f"{document}:p{page}",
            document=document,
            page_number=page,
            chunk_index=0,
            char_start=0,
            char_end=8,
            text="evidence",
        ),
    )


def test_rank_change_classification_handles_found_missing_and_unanswerable() -> None:
    assert classify_rank_change("answerable", 4, 1) == "improved"
    assert classify_rank_change("answerable", None, 5) == "improved"
    assert classify_rank_change("answerable", 1, 4) == "degraded"
    assert classify_rank_change("answerable", 5, None) == "degraded"
    assert classify_rank_change("answerable", 2, 2) == "unchanged"
    assert classify_rank_change("answerable", None, None) == "unchanged"
    assert classify_rank_change("unanswerable", None, None) == "not_scored"


class FakeRetriever:
    name = "hybrid+rerank"
    score_name = "cross-encoder"
    backend_name = "qdrant"

    def __init__(self) -> None:
        self.queries = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        if query == "weak question":
            return [result("wrong.pdf", 1), result("right.pdf", 1)]
        return [result("right.pdf", 1), result("wrong.pdf", 1)]


class FakeRewriter:
    model_name = "fake"
    temperature = 0.0
    max_chars_per_chunk = 100

    def rewrite(self, original_query, preliminary_results):
        assert preliminary_results[0].chunk.document == "wrong.pdf"
        return RewriteResult(
            original_query=original_query,
            rewritten_query="strong question",
            changed=True,
            latency_seconds=0.25,
            model_name=self.model_name,
            fallback=False,
            error=None,
            identity=IdentityDiagnostics(("DPR",), (), ("DPR",)),
        )


def test_experiment_reuses_relevance_logic_and_records_comparison() -> None:
    example = EvaluationExample(
        "q1",
        "weak question",
        "answerable",
        (ExpectedSource("right.pdf", 1),),
    )
    retriever = FakeRetriever()

    report = evaluate_rewriting(
        [example],
        retriever,
        FakeRewriter(),
        CorpusIndexMetadata("embed/model", 384, 10),
        preliminary_top_k=3,
        final_top_k=5,
        dataset_path="frozen.jsonl",
    )

    item = report.questions[0]
    assert retriever.queries == [("weak question", 5), ("strong question", 5)]
    assert item.baseline_first_relevant_rank == 2
    assert item.rewritten_first_relevant_rank == 1
    assert item.outcome == "improved"
    assert item.preliminary_results[0].document == "wrong.pdf"
    assert item.preliminary_results[0].text_preview == "evidence"
    assert item.expected_sources == [
        {"document": "right.pdf", "page_number": 1, "chunk_id": None}
    ]
    assert item.identity_missing_terms == ["DPR"]
    assert report.summary.improved_questions == 1
    assert report.summary.degraded_questions == 0
    assert report.summary.queries_changed == 1
    assert report.summary.identity_warning_questions == 1
    assert report.summary.baseline.hit_at_1 == 0.0
    assert report.summary.rewritten.hit_at_1 == 1.0
    assert report.settings["preliminary_results_reuse_baseline_call"] is True
    assert report.dataset_sha256 is None
