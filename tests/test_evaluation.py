import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pytest

from rag_research_assistant.evaluation import (
    EvaluationExample,
    ExpectedSource,
    detects_refusal,
    evaluate,
    expected_source_matches,
    extract_citations,
    invalid_citations,
    load_evaluation_dataset,
    write_evaluation_report,
)
from rag_research_assistant.index import EmbeddingIndex
from rag_research_assistant.generation import GenerationError
from rag_research_assistant.models import Chunk, SearchResult


class RankedFakeEmbedder:
    model_name = "test/embedding-model"

    def __init__(self, vectors: Dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.vectors[text]


class ConditionalFakeGenerator:
    model_name = "test/generator"

    def generate(self, prompt: str, temperature: float) -> str:
        assert temperature == 0.1
        if "No answer exists" in prompt:
            return "The supplied evidence is insufficient to answer this question."
        return "A generated claim [1] plus an invalid citation [7]."


class FailingFakeGenerator:
    model_name = "test/failing-generator"

    def generate(self, prompt: str, temperature: float) -> str:
        raise GenerationError("local generation failed")


def _chunk(index: int, document: str, page: int) -> Chunk:
    text = f"Evidence in ranked chunk {index}"
    return Chunk(f"chunk-{index}", document, page, 1, 0, len(text), text)


def _fixture_index() -> EmbeddingIndex:
    chunks = [
        _chunk(1, "first.pdf", 1),
        _chunk(2, "other.pdf", 2),
        _chunk(3, "third.pdf", 3),
        _chunk(4, "other.pdf", 4),
        _chunk(5, "fifth.pdf", 5),
        _chunk(6, "sixth.pdf", 6),
    ]
    return EmbeddingIndex(
        embeddings=np.eye(6, dtype=np.float32),
        chunks=chunks,
        model_name="test/embedding-model",
    )


def _question(example_id: str, expected: Sequence[ExpectedSource]) -> EvaluationExample:
    return EvaluationExample(
        id=example_id,
        question=example_id,
        answerability="answerable",
        expected_sources=tuple(expected),
    )


def test_load_evaluation_dataset_parses_all_answerability_categories(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "questions.jsonl"
    rows = [
        {
            "id": "a",
            "question": "Answerable?",
            "answerability": "answerable",
            "expected_sources": [{"document": "a.pdf", "page_number": 2}],
            "notes": "verified",
        },
        {
            "id": "p",
            "question": "Partly?",
            "answerability": "partially_answerable",
            "expected_sources": [{"document": "p.pdf", "chunk_id": "p-1"}],
        },
        {
            "id": "u",
            "question": "Absent?",
            "answerability": "unanswerable",
            "expected_sources": [],
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    examples = load_evaluation_dataset(dataset)

    assert [example.answerability for example in examples] == [
        "answerable",
        "partially_answerable",
        "unanswerable",
    ]
    assert examples[0].expected_sources[0].page_number == 2
    assert examples[1].expected_sources[0].chunk_id == "p-1"
    assert examples[0].notes == "verified"


@pytest.mark.parametrize(
    "row,error",
    [
        (
            {
                "id": "bad",
                "question": "Question",
                "answerability": "unknown",
                "expected_sources": [],
            },
            "answerability",
        ),
        (
            {
                "id": "bad",
                "question": "Question",
                "answerability": "unanswerable",
                "expected_sources": [{"document": "paper.pdf", "page_number": 1}],
            },
            "cannot have expected_sources",
        ),
        (
            {
                "id": "bad",
                "question": "Question",
                "answerability": "answerable",
                "expected_sources": [],
            },
            "need expected_sources",
        ),
    ],
)
def test_load_evaluation_dataset_rejects_invalid_rows(
    tmp_path: Path, row: dict, error: str
) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match=error):
        load_evaluation_dataset(dataset)


def test_load_evaluation_dataset_rejects_duplicate_expected_sources(
    tmp_path: Path,
) -> None:
    source = {"document": "paper.pdf", "page_number": 1}
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "duplicate",
                "question": "Question",
                "answerability": "answerable",
                "expected_sources": [source, source],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="duplicates"):
        load_evaluation_dataset(dataset)


def test_expected_source_matching_uses_optional_page_and_chunk_id() -> None:
    chunk = _chunk(1, "paper.pdf", 4)
    result = SearchResult(0.9, chunk)

    assert expected_source_matches(ExpectedSource("paper.pdf"), result)
    assert expected_source_matches(ExpectedSource("paper.pdf", 4), result)
    assert expected_source_matches(ExpectedSource("paper.pdf", 4, "chunk-1"), result)
    assert not expected_source_matches(ExpectedSource("paper.pdf", 5), result)
    assert not expected_source_matches(ExpectedSource("other.pdf", 4), result)
    assert not expected_source_matches(ExpectedSource("paper.pdf", 4, "wrong"), result)


def test_evaluation_calculates_hit_ranks_recall_and_unanswerable_handling() -> None:
    query_vector = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    examples = [
        _question("rank_1", [ExpectedSource("first.pdf", 1)]),
        _question("rank_3", [ExpectedSource("third.pdf", 3)]),
        _question("rank_5", [ExpectedSource("fifth.pdf", 5)]),
        _question("missing", [ExpectedSource("sixth.pdf", 6)]),
        _question(
            "multiple",
            [ExpectedSource("first.pdf", 1), ExpectedSource("third.pdf", 3)],
        ),
        EvaluationExample("unanswerable", "No answer exists", "unanswerable", ()),
    ]
    embedder = RankedFakeEmbedder(
        {example.question: query_vector for example in examples}
    )

    report = evaluate(examples, _fixture_index(), embedder, retrieval_depth=5)

    by_id = {result.id: result for result in report.questions}
    assert by_id["rank_1"].first_correct_rank == 1
    assert by_id["rank_3"].first_correct_rank == 3
    assert by_id["rank_5"].first_correct_rank == 5
    assert by_id["missing"].first_correct_rank is None
    assert by_id["rank_3"].hit_at_1 is False
    assert by_id["rank_3"].hit_at_3 is True
    assert by_id["multiple"].expected_source_recall_at_1 == 0.5
    assert by_id["multiple"].expected_source_recall_at_3 == 1.0
    assert by_id["unanswerable"].hit_at_5 is None
    assert len(by_id["unanswerable"].retrieved_sources) == 5

    assert report.summary.retrieval_questions == 5
    assert report.summary.unanswerable_questions == 1
    assert report.summary.hit_at_1 == pytest.approx(0.4)
    assert report.summary.hit_at_3 == pytest.approx(0.6)
    assert report.summary.hit_at_5 == pytest.approx(0.8)
    assert report.summary.mean_first_correct_rank == pytest.approx(2.5)
    assert report.summary.retrieval_failure_ids == ["missing"]


def test_evaluation_uses_supplied_retrieval_strategy_without_changing_metrics() -> None:
    class FixedRetriever:
        name = "bm25"
        score_name = "bm25"

        def __init__(self) -> None:
            self.calls = []

        def search(self, query: str, top_k: int = 5):
            self.calls.append((query, top_k))
            return [SearchResult(4.2, _chunk(1, "first.pdf", 1))]

    example = _question("one", [ExpectedSource("first.pdf", 1)])
    retriever = FixedRetriever()
    embedder = RankedFakeEmbedder({})

    report = evaluate(
        [example],
        _fixture_index(),
        embedder,
        retrieval_depth=5,
        retriever=retriever,
    )

    assert retriever.calls == [("one", 5)]
    assert report.retrieval_strategy == "bm25"
    assert report.retrieval_score_type == "bm25"
    assert report.summary.hit_at_1 == 1.0


def test_generation_evaluation_records_refusals_and_validates_citations() -> None:
    query_vector = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)
    examples = [
        _question("answerable", [ExpectedSource("first.pdf", 1)]),
        EvaluationExample("unanswerable", "No answer exists", "unanswerable", ()),
    ]
    embedder = RankedFakeEmbedder(
        {example.question: query_vector for example in examples}
    )

    report = evaluate(
        examples,
        _fixture_index(),
        embedder,
        generator=ConditionalFakeGenerator(),
    )

    answerable = report.questions[0].generation
    unanswerable = report.questions[1].generation
    assert answerable is not None
    assert answerable.detected_citations == [1, 7]
    assert answerable.invalid_citations == [7]
    assert answerable.mapped_citations[0].document == "first.pdf"
    assert answerable.refusal_detected is False
    assert unanswerable is not None and unanswerable.refusal_detected is True
    assert report.summary.unanswerable_refusal_accuracy == 1.0
    assert report.summary.invalid_citation_references == 1
    assert report.summary.invalid_citation_question_ids == ["answerable"]


def test_citation_and_refusal_helpers_are_explicit_heuristics() -> None:
    assert extract_citations("Claims [2][1], repeated [2], invalid [0].") == [2, 1, 0]
    assert invalid_citations([2, 1, 0, 7], context_count=2) == [0, 7]
    assert detects_refusal("There is NOT ENOUGH information in the context.")
    assert detects_refusal("The provided sources do not contain enough information.")
    assert not detects_refusal("The answer is probably something else.")


def test_generation_errors_are_recorded_without_becoming_refusal_successes() -> None:
    example = EvaluationExample("unanswerable", "No answer exists", "unanswerable", ())
    embedder = RankedFakeEmbedder(
        {
            "No answer exists": np.asarray(
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32
            )
        }
    )

    report = evaluate(
        [example], _fixture_index(), embedder, generator=FailingFakeGenerator()
    )

    generation = report.questions[0].generation
    assert generation is not None
    assert generation.error == "local generation failed"
    assert generation.refusal_detected is None
    assert report.summary.unanswerable_refusal_accuracy == 0.0
    assert report.summary.generation_error_ids == ["unanswerable"]


def test_report_serialization_contains_per_question_diagnostics(tmp_path: Path) -> None:
    example = _question("one", [ExpectedSource("first.pdf", 1)])
    embedder = RankedFakeEmbedder(
        {"one": np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)}
    )
    report = evaluate(
        [example], _fixture_index(), embedder, dataset_path="questions.jsonl"
    )
    output = tmp_path / "nested" / "report.json"

    write_evaluation_report(report, output)
    value = json.loads(output.read_text())

    assert value["schema_version"] == 2
    assert value["retrieval_strategy"] == "dense"
    assert value["retrieval_score_type"] == "cosine"
    assert value["dataset_path"] == "questions.jsonl"
    assert value["summary"]["hit_at_1"] == 1.0
    assert value["questions"][0]["retrieved_sources"][0]["chunk_id"] == "chunk-1"
    assert value["questions"][0]["retrieved_sources"][0]["text"]


def test_evaluation_requires_depth_for_all_phase_four_hit_metrics() -> None:
    example = _question("one", [ExpectedSource("first.pdf", 1)])
    embedder = RankedFakeEmbedder(
        {"one": np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=np.float32)}
    )

    with pytest.raises(ValueError, match="at least 5"):
        evaluate([example], _fixture_index(), embedder, retrieval_depth=3)
