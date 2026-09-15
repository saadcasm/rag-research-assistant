import json

from rag_research_assistant.evaluation import EvaluationExample, ExpectedSource
from rag_research_assistant.experiments.phase10_hyde_eval import (
    evaluate_experiment,
    write_json,
    write_markdown,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.query_transform.hyde import (
    HyDEDiagnostics,
    HyDERetrievalResult,
    HypotheticalDocument,
)


def result(chunk_id, document="paper.pdf"):
    return SearchResult(
        1.0,
        Chunk(chunk_id, document, 1, 0, 0, 10, f"text {chunk_id}"),
    )


class FakeHyDEGenerator:
    model_name = "fake"

    def generate(self, question):
        return HypotheticalDocument(
            question=question,
            text="technical passage",
            model_name="fake",
            word_count=2,
            character_count=17,
            generation_seconds=1.0,
            truncated=False,
            fallback=False,
            error=None,
            diagnostics=HyDEDiagnostics((), (), (), (), (), (), (), 0.5, 0.1, ()),
        )


class FakeHyDERetriever:
    def __init__(self):
        self.calls = []

    def search(self, question, hypothetical, *, top_k):
        self.calls.append((question, hypothetical.text, top_k))
        baseline = (result("base"), result("relevant"))
        hyde = (result("relevant"), result("hyde"))
        fused = (result("relevant"), result("base"))
        return HyDERetrievalResult(
            baseline, baseline, hyde, hyde, fused, fused,
            0.1, 0.02, 0.03, 0.04, 0.001, 0.04,
            1.09, 1.231, False, None,
        )


def test_report_serializes_rank_changes_metrics_diagnostics_and_latency(tmp_path) -> None:
    examples = [
        EvaluationExample(
            id="q1",
            question="question",
            answerability="answerable",
            expected_sources=(ExpectedSource("paper.pdf", 1, "relevant"),),
        ),
        EvaluationExample(
            id="u1", question="unsupported", answerability="unanswerable", expected_sources=()
        ),
    ]
    retriever = FakeHyDERetriever()

    report = evaluate_experiment(examples, FakeHyDEGenerator(), retriever)

    assert len(retriever.calls) == 2
    assert report["summary"]["total_questions"] == 2
    assert report["summary"]["retrieval_scored_questions"] == 1
    assert report["summary"]["unanswerable_questions"] == 1
    assert report["summary"]["baseline"]["hit_at_1"] == 0.0
    assert report["summary"]["hyde_only"]["hit_at_1"] == 1.0
    assert report["summary"]["fused"]["hit_at_1"] == 1.0
    assert report["summary"]["hyde_only_improved"] == 1
    assert report["questions"][0]["hyde_generation_seconds"] == 1.0
    assert report["questions"][0]["hyde_only_total_seconds"] == 1.09
    assert report["questions"][0]["hypothetical_document"]["text"] == "technical passage"

    json_path = tmp_path / "result.json"
    md_path = tmp_path / "result.md"
    write_json(report, json_path)
    write_markdown(report, md_path)
    assert json.loads(json_path.read_text())["experiment"].startswith("phase-10f")
    assert "HyDE-only" in md_path.read_text()
