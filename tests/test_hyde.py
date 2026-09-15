import numpy as np
import pytest

from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.query_transform.hyde import (
    HyDEDocumentGenerator,
    HyDEExperimentalRetriever,
    HypotheticalDocument,
    build_hyde_prompt,
    diagnose_hypothetical_document,
    normalize_hypothetical_document,
)


def result(chunk_id: str, score: float = 1.0) -> SearchResult:
    return SearchResult(
        score,
        Chunk(chunk_id, "paper.pdf", 1, 0, 0, 10, f"text {chunk_id}"),
    )


class FakeGenerator:
    model_name = "fake-qwen"

    def __init__(self, response="passage", error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, prompt, temperature):
        self.calls.append((prompt, temperature))
        if self.error:
            raise self.error
        return self.response


class FakeEmbedder:
    model_name = "same/model"

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def embed_query(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return np.asarray([0.2, 0.8], dtype=np.float32)


class FakeDense:
    def __init__(self, ranking, embedder=None):
        self.ranking = ranking
        self.embedder = embedder or FakeEmbedder()
        self.calls = []

    def search_vector(self, vector, *, top_k=5):
        self.calls.append((np.asarray(vector), top_k))
        return self.ranking[:top_k]


class FakeBaseline:
    name = "frozen"

    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.ranking[:top_k]


class RecordingReranker:
    model_name = "same/reranker"

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, list(candidates), top_k))
        return list(candidates[:top_k])


def hypothetical(text="generated semantic probe", fallback=False, error=None):
    return HypotheticalDocument(
        question="original question",
        text=text,
        model_name="fake-qwen",
        word_count=len(text.split()),
        character_count=len(text),
        generation_seconds=0.25,
        truncated=False,
        fallback=fallback,
        error=error,
        diagnostics=None,
    )


def test_hyde_prompt_requests_short_passage_and_preserves_question() -> None:
    prompt = build_hyde_prompt("How does DPR encode passages?")

    assert "How does DPR encode passages?" in prompt
    assert "2 to 5 sentences" in prompt
    assert "60 to 150 words" in prompt
    assert "Do not fabricate or include citations" in prompt
    assert "search query" in prompt
    assert "final grounded answer" not in prompt


def test_hypothetical_normalization_removes_label_and_caps_words() -> None:
    text, truncated = normalize_hypothetical_document(
        "  Hypothetical document:  One\n two   three four  ", max_words=3
    )

    assert text == "One two three"
    assert truncated is True


@pytest.mark.parametrize(
    "fake",
    [FakeGenerator("   "), FakeGenerator(error=RuntimeError("offline"))],
)
def test_empty_or_generator_error_produces_safe_fallback(fake) -> None:
    generated = HyDEDocumentGenerator(fake).generate("question")

    assert generated.fallback is True
    assert generated.text == ""
    assert generated.error
    assert generated.generation_seconds >= 0.0


def test_document_generator_records_lengths_diagnostics_and_zero_temperature() -> None:
    fake = FakeGenerator("DPR retrieves 42 passages using BERT embeddings.")

    generated = HyDEDocumentGenerator(fake, temperature=0.0).generate(
        "How does DPR use BERT?"
    )

    assert generated.fallback is False
    assert generated.word_count == 7
    assert generated.character_count == len(generated.text)
    assert generated.diagnostics is not None
    assert "DPR" in generated.diagnostics.preserved_protected_terms
    assert "42" in generated.diagnostics.introduced_numbers
    assert fake.calls[0][1] == 0.0


def test_diagnostics_surface_missing_identity_expansion_citation_and_overlap() -> None:
    diagnostic = diagnose_hypothetical_document(
        "How does DPR work?",
        "The Dense Passage Retriever (new expansion) uses 128 vectors [4].",
    )

    assert "DPR" in diagnostic.missing_protected_terms
    assert "128" in diagnostic.introduced_numbers
    assert "new expansion" in diagnostic.suspicious_expansions
    assert "[4]" in diagnostic.citation_like_patterns
    assert diagnostic.query_term_coverage == 0.0
    assert "no_query_content_term_overlap" in diagnostic.warnings
    assert diagnostic.warnings


def test_hyde_dense_embeds_hypothetical_but_reranks_with_original_question() -> None:
    baseline = FakeBaseline([result("base"), result("shared")])
    embedder = FakeEmbedder()
    dense = FakeDense([result("hyde"), result("shared")], embedder)
    reranker = RecordingReranker()
    retriever = HyDEExperimentalRetriever(
        baseline, dense, reranker, candidate_depth=2, rrf_k=0
    )

    retrieved = retriever.search("original question", hypothetical(), top_k=2)

    assert embedder.calls == ["generated semantic probe"]
    assert dense.calls[0][0].tolist() == pytest.approx([0.2, 0.8])
    assert all(call[0] == "original question" for call in reranker.calls)
    assert len(reranker.calls) == 2
    assert retrieved.fallback is False
    assert retrieved.embedding_seconds >= 0.0
    assert retrieved.dense_seconds >= 0.0


def test_fusion_deduplicates_stable_chunk_ids_and_rewards_consensus() -> None:
    baseline = FakeBaseline([result("shared"), result("base")])
    dense = FakeDense([result("hyde"), result("shared")])
    reranker = RecordingReranker()
    retriever = HyDEExperimentalRetriever(
        baseline, dense, reranker, candidate_depth=3, rrf_k=0
    )

    retrieved = retriever.search("original question", hypothetical(), top_k=3)

    fused_ids = [item.chunk.chunk_id for item in retrieved.fused_candidates]
    assert fused_ids[0] == "shared"
    assert fused_ids.count("shared") == 1
    assert set(fused_ids) == {"shared", "base", "hyde"}


def test_generation_fallback_reproduces_baseline_for_both_hyde_conditions() -> None:
    baseline_ranking = [result("one"), result("two")]
    baseline = FakeBaseline(baseline_ranking)
    dense = FakeDense([result("unused")])
    reranker = RecordingReranker()
    retriever = HyDEExperimentalRetriever(baseline, dense, reranker)

    retrieved = retriever.search(
        "original question",
        hypothetical("", fallback=True, error="generation failed"),
        top_k=2,
    )

    expected = ["one", "two"]
    assert [item.chunk.chunk_id for item in retrieved.hyde_only_results] == expected
    assert [item.chunk.chunk_id for item in retrieved.fused_results] == expected
    assert retrieved.fallback is True
    assert retrieved.error == "generation failed"
    assert dense.calls == []
    assert reranker.calls == []


def test_embedding_error_also_falls_back_to_exact_baseline() -> None:
    baseline = FakeBaseline([result("one"), result("two")])
    dense = FakeDense([], FakeEmbedder(error=RuntimeError("bad vector")))
    retriever = HyDEExperimentalRetriever(baseline, dense, RecordingReranker())

    retrieved = retriever.search("original question", hypothetical(), top_k=2)

    assert retrieved.fallback is True
    assert "bad vector" in retrieved.error
    assert retrieved.hyde_only_results == retrieved.baseline_results
    assert retrieved.fused_results == retrieved.baseline_results
