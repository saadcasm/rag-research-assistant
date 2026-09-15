import json

import numpy as np

from rag_research_assistant.compression import ContextualCompressor, Segmenter
from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource
from rag_research_assistant.experiments.phase10_contextual_compression_eval import (
    GENERATION_QUESTION_IDS,
    _validate_provenance,
    evaluate_experiment,
    run_generation_diagnostic,
    write_json,
    write_markdown,
)
from rag_research_assistant.models import Chunk, SearchResult


class FakeRetriever:
    name = "frozen"

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeScorer:
    def __init__(self):
        self.calls = 0

    def score_texts(self, query, texts):
        self.calls += 1
        return np.arange(len(texts), dtype=np.float32)


class FakeGenerator:
    model_name = "fake-generator"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, temperature):
        self.calls.append((prompt, temperature))
        return "Grounded diagnostic answer [1]."


def result(text, chunk_id, document="paper.pdf", page=1):
    return SearchResult(
        1.0,
        Chunk(
            chunk_id=chunk_id,
            document=document,
            page_number=page,
            chunk_index=0,
            char_start=0,
            char_end=len(text),
            text=text,
        ),
    )


def test_experiment_retrieves_once_and_preserves_ranking_for_all_conditions(tmp_path) -> None:
    results = [result("Evidence one. Noise one.", "a"), result("Evidence two.", "b")]
    retriever = FakeRetriever(results)
    scorer = FakeScorer()
    compressor = ContextualCompressor(
        scorer, segmenter=Segmenter(min_characters=1, target_characters=1)
    )
    example = EvaluationExample(
        id="q1",
        question="Where is evidence?",
        answerability="answerable",
        expected_sources=(ExpectedSource("paper.pdf", 1),),
    )
    raw = {
        "q1": {
            "category": "semantic_paraphrase",
            "difficulty": "easy",
            "evidence": [
                {"document": "paper.pdf", "pages": [1], "passage": "Evidence one."}
            ],
        }
    }

    report = evaluate_experiment(
        [example], raw, retriever, compressor, budgets=[0.5]
    )

    assert retriever.calls == [("Where is evidence?", 5)]
    assert scorer.calls == 1
    question = report["questions"][0]
    assert [item["condition_id"] for item in question["conditions"]] == [
        "none", "per_chunk_50pct", "global_50pct"
    ]
    assert [item["chunk_id"] for item in question["retrieved_chunks"]] == ["a", "b"]
    for condition in question["conditions"]:
        ranks = [context["source_rank"] for context in condition["contexts"]]
        assert ranks == sorted(ranks)
        assert condition["provenance_failures"] == []
    assert report["summary"]["answerable_questions"] == 1

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_json(report, json_path)
    write_markdown(report, markdown_path)
    assert json.loads(json_path.read_text())["experiment"].startswith("phase-10e")
    assert "Evidence term retention" in markdown_path.read_text()


def test_provenance_validator_detects_no_failure_for_exact_segments() -> None:
    source = result("One. Two.", "stable")
    compressor = ContextualCompressor(
        FakeScorer(), segmenter=Segmenter(min_characters=1, target_characters=1)
    )
    compressed = compressor.compress_per_chunk(
        compressor.prepare("q", [source]), 0.6
    )

    assert _validate_provenance(compressed, [source]) == []


def test_generation_diagnostic_uses_one_retrieval_and_two_prompts_per_question() -> None:
    retrieved = [
        result(
            "First evidence sentence. Second useful sentence. Third background sentence. Fourth noise sentence.",
            "stable",
        )
    ]
    retriever = FakeRetriever(retrieved)
    scorer = FakeScorer()
    compressor = ContextualCompressor(
        scorer, segmenter=Segmenter(min_characters=1, target_characters=1)
    )
    generator = FakeGenerator()
    examples = []
    raw = {}
    for index, question_id in enumerate(GENERATION_QUESTION_IDS):
        unanswerable = "unanswerable" in question_id
        examples.append(
            EvaluationExample(
                id=question_id,
                question=f"Diagnostic question {index}?",
                answerability="unanswerable" if unanswerable else "answerable",
                expected_sources=() if unanswerable else (ExpectedSource("paper.pdf", 1),),
            )
        )
        raw[question_id] = {
            "category": "unanswerable" if unanswerable else "exact_number",
            "difficulty": "hard",
            "evidence": [] if unanswerable else [
                {"document": "paper.pdf", "pages": [1], "passage": "First evidence sentence."}
            ],
        }

    report = run_generation_diagnostic(
        examples, raw, retriever, compressor, generator
    )

    assert len(retriever.calls) == len(GENERATION_QUESTION_IDS)
    assert len(generator.calls) == 2 * len(GENERATION_QUESTION_IDS)
    assert scorer.calls == len(GENERATION_QUESTION_IDS)
    assert report["summary"]["questions"] == 10
    assert report["summary"]["unanswerable"] == 2
    assert report["summary"]["provenance_failures"] == 0
    for question in report["questions"]:
        assert question["retrieved_chunk_ids"] == ["stable"]
        assert question["original"]["answer"].endswith("[1].")
        assert question["compressed"]["answer"].endswith("[1].")
        assert question["compressed"]["source_text_characters"] <= question["original"]["source_text_characters"]
