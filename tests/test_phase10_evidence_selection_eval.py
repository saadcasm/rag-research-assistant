import json
from pathlib import Path

from rag_research_assistant.experiments.phase10_evidence_selection_eval import (
    build_candidate_coverage_report,
    write_json,
    write_markdown,
)
from rag_research_assistant.models import Chunk
from rag_research_assistant.multihop import ExpectedHop, HopEvidence, MultiHopExample


def test_candidate_coverage_report_marks_unpersisted_depth_unavailable(tmp_path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("{}\n", encoding="utf-8")
    artifact_path = tmp_path / "phase10g.json"
    chunks = [
        Chunk("a", "a.pdf", 1, 0, 0, 14, "alpha evidence"),
        Chunk("b", "b.pdf", 2, 0, 0, 13, "beta evidence"),
    ]
    example = MultiHopExample(
        "q1", "Compare alpha and beta", "answerable", "comparison",
        (
            ExpectedHop("alpha", (), (HopEvidence("a.pdf", (1,), "alpha evidence"),)),
            ExpectedHop("beta", (), (HopEvidence("b.pdf", (2,), "beta evidence"),)),
        ),
        "verified",
    )
    import hashlib
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    artifact = {
        "experiment": "phase-10g-query-decomposition-multihop-retrieval",
        "dataset_sha256": dataset_hash,
        "questions": [{
            "question_id": "q1", "question": "Compare alpha and beta",
            "conditions": {"baseline": {
                "branch_rankings": [{"results": [
                    {"rank": 1, "score": 1.0, "chunk_id": "a"},
                    {"rank": 2, "score": 0.5, "chunk_id": "b"},
                ]}],
                "final_results": [
                    {"rank": 1, "score": 1.0, "chunk_id": "a"},
                    {"rank": 2, "score": 0.5, "chunk_id": "b"},
                ],
            }},
        }],
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    report = build_candidate_coverage_report(
        artifact, [example], chunks,
        artifact_path=artifact_path, dataset_path=dataset, chunks_path=chunks_path,
        requested_depths=(1, 2, 5),
    )
    assert report["settings"]["available_depths"] == [1, 2]
    assert report["settings"]["unavailable_depths"] == [5]
    assert report["summary"]["oracle_candidate_pool"]["2"]["oracle_full_hop_availability"] == 1.0
    assert report["summary"]["oracle_candidate_pool"]["5"] is None
    assert report["summary"]["analysis_seconds"] >= 0.0

    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    write_json(report, json_output)
    write_markdown(report, markdown_output)
    assert json.loads(json_output.read_text())["settings"]["retrieval_or_model_rerun"] is False
    assert "unavailable" in markdown_output.read_text()


def test_phase10h_module_is_not_imported_by_production_paths() -> None:
    root = Path(__file__).parents[1] / "src/rag_research_assistant"
    for relative in ("api.py", "application.py", "cli.py", "rag.py"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "evidence_selection" not in text
        assert "phase10_evidence_selection" not in text
