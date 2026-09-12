from math import isclose

from rag_research_assistant.bm25 import BM25Index, tokenize
from rag_research_assistant.models import Chunk


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id, "paper.pdf", 1, 1, 0, len(text), text)


def test_tokenize_is_unicode_safe_case_insensitive_and_removes_punctuation() -> None:
    assert tokenize("Café, RETRIEVAL_v2!") == ["café", "retrieval", "v2"]


def test_idf_is_higher_for_a_rarer_term() -> None:
    index = BM25Index(
        [_chunk("one", "common rare"), _chunk("two", "common"), _chunk("three", "common")]
    )

    assert index.inverse_document_frequency("rare") > index.inverse_document_frequency(
        "common"
    )


def test_term_frequency_increases_with_saturation() -> None:
    index = BM25Index(
        [_chunk("one", "memory"), _chunk("three", "memory memory memory")], b=0.0
    )

    scores = index.score("memory")
    assert scores[1] > scores[0] > 0
    assert scores[1] < scores[0] * 3


def test_document_length_normalization_favors_concise_matching_chunk() -> None:
    index = BM25Index(
        [
            _chunk("short", "retrieval"),
            _chunk("long", "retrieval plus many unrelated filler words here"),
        ]
    )

    results = index.search("retrieval", top_k=2)
    assert [result.chunk.chunk_id for result in results] == ["short", "long"]
    assert results[0].score > results[1].score


def test_bm25_ranking_preserves_metadata_and_stable_ties() -> None:
    exact = Chunk("exact", "exact.pdf", 4, 2, 0, 21, "non-parametric memory")
    other = Chunk("other", "other.pdf", 7, 3, 0, 14, "semantic model")
    tied = Chunk("tied", "tied.pdf", 9, 1, 0, 12, "another text")
    results = BM25Index([exact, other, tied]).search("non-parametric memory", top_k=3)

    assert results[0].chunk == exact
    assert [result.chunk.chunk_id for result in results[1:]] == ["other", "tied"]
    assert isclose(results[1].score, 0.0)
