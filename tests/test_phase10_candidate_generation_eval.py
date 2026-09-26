import json
from pathlib import Path

from rag_research_assistant.candidate_generation.diagnostics import ComponentRankings
from rag_research_assistant.experiments.phase10_candidate_generation_eval import (
    build_candidate_generation_report,
    write_json,
    write_markdown,
)
from rag_research_assistant.models import Chunk, SearchResult
from rag_research_assistant.multihop import ExpectedHop, HopEvidence, MultiHopExample


def test_report_contains_provenance_categories_and_no_model_generation(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("{}\n", encoding="utf-8")
    text = "DPR uses in-batch negatives"
    candidate = SearchResult(0.9, Chunk("d", "dpr.pdf", 3, 1, 0, len(text), text))
    example = MultiHopExample(
        "q1",
        "How does DPR train?",
        "partially_answerable",
        "partially_answerable",
        (
            ExpectedHop("DPR", (), (HopEvidence("dpr.pdf", (3,), text),)),
            ExpectedHop("missing", (), (), False),
        ),
        "verified",
    )
    ranking = ComponentRankings(
        "q1",
        "How does DPR train?",
        (candidate,),
        (),
        (candidate,),
        {"dense": 0.1, "bm25": 0.1, "rrf": 0.01, "total": 0.21},
        5,
        60,
    )
    settings = {
        "production_path_changed": False,
        "cross_encoder_used": False,
        "ollama_used": False,
        "gold_labels_used_for": "evaluation_only_after_rankings_frozen",
        "depths": [1, 5],
        "rrf_k": 60,
    }

    report = build_candidate_generation_report(
        [example],
        [ranking],
        dataset_path=dataset,
        chunks_path=chunks_path,
        settings=settings,
        depths=(1, 5),
    )
    assert report["experiment"] == "phase-10i-component-level-candidate-generation"
    assert report["summary"]["all_questions"]["available_hops"] == 1
    assert report["summary"]["partially_answerable"]["questions"] == 1
    assert report["summary"]["categories"]["partially_answerable"]["questions"] == 1
    assert report["questions"][0]["rankings"]["dense"][0]["chunk_id"] == "d"

    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_json(report, json_path)
    write_markdown(report, markdown_path)
    assert json.loads(json_path.read_text())["settings"]["ollama_used"] is False
    assert "Passage recall" in markdown_path.read_text()
    assert "Correct document found, exact passage missing" in markdown_path.read_text()
    assert "Category-level hybrid coverage" in markdown_path.read_text()
    assert "does not select or implement" in markdown_path.read_text()


def test_phase10i_experiment_is_not_imported_by_production_paths() -> None:
    root = Path(__file__).parents[1] / "src/rag_research_assistant"
    for relative in (
        "api.py",
        "application.py",
        "cli.py",
        "rag.py",
        "frameworks/langgraph_workflow.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "candidate_generation" not in text
        assert "phase10_candidate_generation_eval" not in text
