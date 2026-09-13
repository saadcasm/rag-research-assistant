import json
from collections import Counter
from pathlib import Path

from rag_research_assistant.evaluation import load_evaluation_dataset


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "data/evaluation/questions-phase-7.5.jsonl"
ORIGINAL_DATASET = ROOT / "data/evaluation/questions.jsonl"
CORPUS_MANIFEST = ROOT / "data/corpus/manifest.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_phase75_dataset_has_intended_size_split_and_unique_ids() -> None:
    rows = _read_jsonl(DATASET)

    assert len(rows) == 100
    assert len({row["id"] for row in rows}) == 100
    assert Counter(row["answerability"] for row in rows) == {
        "answerable": 90,
        "unanswerable": 10,
    }
    assert len(load_evaluation_dataset(DATASET)) == 100


def test_phase75_answerable_records_have_consistent_ground_truth() -> None:
    rows = _read_jsonl(DATASET)

    for row in rows:
        if row["answerability"] == "unanswerable":
            assert row["category"] == "unanswerable"
            assert row["expected_sources"] == []
            assert row["evidence"] == []
            assert row["verification_status"] == "verified_absent"
            continue

        assert row["verification_status"] == "verified"
        assert row["expected_sources"]
        assert row["evidence"]
        assert row["evidence_explanation"].strip()
        source_pairs = {
            (source["document"], source["page_number"])
            for source in row["expected_sources"]
        }
        evidence_pairs = {
            (evidence["document"], page)
            for evidence in row["evidence"]
            for page in evidence["pages"]
        }
        assert source_pairs == evidence_pairs
        assert all(evidence["passage"].strip() for evidence in row["evidence"])


def test_phase75_dataset_covers_every_approved_corpus_pdf() -> None:
    rows = _read_jsonl(DATASET)
    manifest = _read_jsonl(CORPUS_MANIFEST)
    approved_documents = {
        record["filename"]
        for record in manifest
        if record["download_status"] in {"verified", "existing_verified"}
    }
    labelled_documents = {
        source["document"]
        for row in rows
        for source in row["expected_sources"]
    }

    assert len(approved_documents) == 52
    assert labelled_documents == approved_documents


def test_phase75_dataset_is_separate_from_original_benchmark() -> None:
    original_rows = _read_jsonl(ORIGINAL_DATASET)
    phase75_rows = _read_jsonl(DATASET)

    assert len(original_rows) == 20
    assert {row["id"] for row in original_rows}.isdisjoint(
        row["id"] for row in phase75_rows
    )
