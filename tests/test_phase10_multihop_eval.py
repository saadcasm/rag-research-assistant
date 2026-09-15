import json
from pathlib import Path

from rag_research_assistant.experiments.phase10_multihop_eval import (
    evaluate_multihop,
    write_json,
    write_markdown,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.multihop import (
    ExpectedHop,
    FusedCondition,
    HopEvidence,
    MultiHopExample,
    MultiHopRetrievalResult,
)
from rag_research_assistant.query_transform.decomposition import DecompositionResult


def search_result(identifier, document, page):
    return SearchResult(1.0, Chunk(identifier, document, page, 0, 0, 5, f"{identifier.upper()} evidence"))


def condition(results, query="Q"):
    return FusedCondition((), tuple(results), tuple(results), {item.chunk.chunk_id: (query,) for item in results}, 0, 0.01, 0.02)


class FakeDecomposer:
    def decompose(self, question):
        return DecompositionResult(question, True, '{"needs_decomposition":true}', ("A?", "B?"), ("A?", "B?"), (), "fake", 0.1, True, False, None)


class FakeRetriever:
    def search(self, question, decomposition, top_k=10):
        a = search_result("a", "a.pdf", 1)
        b = search_result("b", "b.pdf", 2)
        baseline = condition([a])
        improved = condition([a, b])
        return MultiHopRetrievalResult(decomposition, baseline, improved, improved, False, None)


def test_experiment_serializes_hop_metrics_and_raw_decomposition(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    example = MultiHopExample(
        "mh", "Compare A and B", "answerable", "comparison",
        (
            ExpectedHop("A", (), (HopEvidence("a.pdf", (1,), "A"),)),
            ExpectedHop("B", (), (HopEvidence("b.pdf", (2,), "B"),)),
        ),
        "draft",
    )
    report = evaluate_multihop([example], FakeDecomposer(), FakeRetriever(), dataset_path=dataset, settings={"rrf_k": 60})
    assert report["summary"]["conditions"]["baseline"]["hop_coverage_at_5"]["full_hop_coverage"] == 0.0
    assert report["summary"]["conditions"]["original_plus_decomposed"]["hop_coverage_at_5"]["full_hop_coverage"] == 1.0
    assert report["questions"][0]["decomposition"]["raw_response"]

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_json(report, json_path)
    write_markdown(report, markdown_path)
    assert json.loads(json_path.read_text())["experiment"].startswith("phase-10g")
    assert "Full@5" in markdown_path.read_text()


def test_phase10g_draft_dataset_is_small_multihop_and_original_is_unchanged() -> None:
    root = Path(__file__).parents[1]
    draft = [json.loads(line) for line in (root / "data/evaluation/questions-phase-10g-multihop-draft.jsonl").read_text().splitlines()]
    frozen = [json.loads(line) for line in (root / "data/evaluation/questions-phase-7.5.jsonl").read_text().splitlines()]
    assert 20 <= len(draft) <= 30
    assert sum(row["answerability"] == "partially_answerable" for row in draft) == 4
    assert all(len(row["hops"]) >= 2 for row in draft)
    assert len(frozen) == 100


def test_partial_question_papers_have_no_dedicated_corpus_record_or_pdf() -> None:
    root = Path(__file__).parents[1]
    manifest = [
        json.loads(line)
        for line in (root / "data/corpus/manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    searchable = "\n".join(
        " ".join(str(row.get(field, "")) for field in ("paper_id", "title", "filename"))
        for row in manifest
    ).casefold()
    pdf_names = "\n".join(
        path.name for path in (root / "data/corpus/pdfs").glob("*.pdf")
    ).casefold()

    for absent_method in ("plaid", "unicoil", "diskann", "miracl"):
        assert absent_method not in searchable
        assert absent_method not in pdf_names
