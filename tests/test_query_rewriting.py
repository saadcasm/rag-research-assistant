import pytest

from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.query_transform.rewriting import (
    CorpusGroundedLLMRewriter,
    build_rewrite_prompt,
    diagnose_identity_preservation,
    parse_rewritten_query,
)


def result() -> SearchResult:
    return SearchResult(
        score=4.2,
        chunk=Chunk(
            chunk_id="dpr:p1:c0",
            document="dpr.pdf",
            page_number=1,
            chunk_index=0,
            char_start=0,
            char_end=74,
            text=(
                "Dense Passage Retrieval for Open-Domain Question Answering "
                "Vladimir Karpukhin"
            ),
        ),
    )


class FakeGenerator:
    model_name = "fake-rewriter"

    def __init__(self, response: str = "", failure: Exception | None = None) -> None:
        self.response = response
        self.failure = failure
        self.prompts: list[tuple[str, float]] = []

    def generate(self, prompt: str, temperature: float) -> str:
        self.prompts.append((prompt, temperature))
        if self.failure:
            raise self.failure
        return self.response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("rewritten words", "rewritten words"),
        ('"rewritten words"', "rewritten words"),
        ("Rewritten query: rewritten words", "rewritten words"),
        ("```\nrewritten words\n```", "rewritten words"),
    ],
)
def test_parse_rewritten_query_accepts_one_query(raw: str, expected: str) -> None:
    assert parse_rewritten_query(raw) == expected


def test_original_query_can_be_returned_unchanged() -> None:
    generator = FakeGenerator("How does DPR represent passages?")
    rewriter = CorpusGroundedLLMRewriter(generator)

    rewritten = rewriter.rewrite("How does DPR represent passages?", [result()])

    assert rewritten.changed is False
    assert rewritten.fallback is False
    assert rewritten.rewritten_query == rewritten.original_query


def test_successful_rewrite_is_structured_and_preserves_identity() -> None:
    generator = FakeGenerator(
        "Who are the authors of Dense Passage Retrieval for Open-Domain "
        "Question Answering (DPR)?"
    )
    rewriter = CorpusGroundedLLMRewriter(generator, temperature=0.0)

    rewritten = rewriter.rewrite("Who wrote the DPR paper?", [result()])

    assert rewritten.changed is True
    assert rewritten.fallback is False
    assert rewritten.error is None
    assert rewritten.identity.missing_terms == ()
    assert rewritten.identity.protected_terms == ("DPR",)


def test_empty_response_falls_back_to_original_query() -> None:
    rewritten = CorpusGroundedLLMRewriter(FakeGenerator("   ")).rewrite(
        "Who wrote the DPR paper?", [result()]
    )

    assert rewritten.rewritten_query == "Who wrote the DPR paper?"
    assert rewritten.changed is False
    assert rewritten.fallback is True
    assert "empty response" in rewritten.error


def test_model_exception_falls_back_to_original_query() -> None:
    rewritten = CorpusGroundedLLMRewriter(
        FakeGenerator(failure=RuntimeError("model stopped"))
    ).rewrite("Who wrote the DPR paper?", [result()])

    assert rewritten.rewritten_query == "Who wrote the DPR paper?"
    assert rewritten.fallback is True
    assert rewritten.error == "RuntimeError: model stopped"


def test_preliminary_evidence_and_provenance_are_in_prompt() -> None:
    generator = FakeGenerator("Who wrote the DPR paper?")
    rewriter = CorpusGroundedLLMRewriter(generator, max_chars_per_chunk=500)

    rewriter.rewrite("Who wrote the DPR paper?", [result()])

    prompt, temperature = generator.prompts[0]
    assert "Who wrote the DPR paper?" in prompt
    assert "Dense Passage Retrieval for Open-Domain Question Answering" in prompt
    assert "document: dpr.pdf" in prompt
    assert "page: 1" in prompt
    assert "chunk_id: dpr:p1:c0" in prompt
    assert "Do not answer the question" in prompt
    assert temperature == 0.0


def test_identity_diagnostic_surfaces_lost_acronym_quote_and_model_token() -> None:
    diagnostic = diagnose_identity_preservation(
        'How does DPR use "Natural Questions" with MiniLM-L6-v2?',
        "How does dense retrieval use Natural Questions?",
    )

    assert diagnostic.protected_terms == (
        "Natural Questions",
        "DPR",
        "MiniLM-L6-v2",
    )
    assert diagnostic.preserved_terms == ("Natural Questions",)
    assert diagnostic.missing_terms == ("DPR", "MiniLM-L6-v2")
