"""Offline tests for reproducible corpus provenance and acquisition."""

import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

import rag_research_assistant.corpus as corpus


def _pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    output = __import__("io").BytesIO()
    writer.write(output)
    return output.getvalue()


def _spec(paper_id: str, *, existing: bool = False, approved: bool = True) -> corpus.PaperSpec:
    return corpus.PaperSpec(
        paper_id, "Expected Paper", "Test Author", 2024,
        f"https://example.test/{paper_id}.pdf", "Test", existing, approved,
    )


def test_inspect_pdf_rejects_html_response() -> None:
    with pytest.raises(corpus.CorpusValidationError, match="not a non-empty PDF"):
        corpus.inspect_pdf_bytes(b"<html>not a PDF</html>", "Expected Paper")


def test_inspect_pdf_accepts_structurally_valid_pdf_when_title_check_is_disabled() -> None:
    info = corpus.inspect_pdf_bytes(_pdf_bytes(), "")
    assert info["page_count"] == 1


def test_checksum_and_deterministic_filename(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"same bytes")
    assert corpus.sha256_file(path) == corpus.sha256_file(path)
    assert _spec("stable_id").filename == "stable_id.pdf"


def test_held_record_is_not_fetched_and_manifest_is_written(monkeypatch, tmp_path: Path) -> None:
    held = _spec("held", approved=False)
    monkeypatch.setattr(corpus, "load_proposed_corpus", lambda _: [held])
    records = corpus.build_corpus(proposal_path=tmp_path / "proposal.md", corpus_dir=tmp_path / "corpus", fetcher=lambda *_: pytest.fail("fetch should not run"))
    assert records[0]["download_status"] == "held_source_unverified"
    assert json.loads((tmp_path / "corpus" / "manifest.jsonl").read_text())["paper_id"] == "held"


def test_existing_file_reconciles_without_download(monkeypatch, tmp_path: Path) -> None:
    spec = _spec("rag_lewis_2020", existing=True)
    papers = tmp_path / "papers"; papers.mkdir()
    source = papers / "rag-lewis-2020.pdf"; source.write_bytes(_pdf_bytes())
    monkeypatch.setattr(corpus, "load_proposed_corpus", lambda _: [spec])
    monkeypatch.setattr(corpus, "inspect_pdf_bytes", lambda *_: {"page_count": 1, "title_word_overlap": 1.0})
    records = corpus.build_corpus(proposal_path=tmp_path / "proposal.md", corpus_dir=tmp_path / "corpus", existing_papers_dir=papers, fetcher=lambda *_: pytest.fail("fetch should not run"))
    assert records[0]["download_status"] == "existing_verified"
    assert (tmp_path / "corpus" / "pdfs" / "rag_lewis_2020.pdf").exists()


def test_download_is_idempotent_and_duplicate_checksums_are_reported(monkeypatch, tmp_path: Path) -> None:
    specs = [_spec("one"), _spec("two")]
    calls = []
    monkeypatch.setattr(corpus, "load_proposed_corpus", lambda _: specs)
    monkeypatch.setattr(corpus, "inspect_pdf_bytes", lambda *_: {"page_count": 1, "title_word_overlap": 1.0})
    def fetch(url: str, timeout: float):
        calls.append(url); return _pdf_bytes(), "application/pdf", url
    first = corpus.build_corpus(proposal_path=tmp_path / "proposal.md", corpus_dir=tmp_path / "corpus", fetcher=fetch)
    assert all(record["download_status"] == "duplicate_checksum" for record in first)
    corpus.build_corpus(proposal_path=tmp_path / "proposal.md", corpus_dir=tmp_path / "corpus", fetcher=lambda *_: pytest.fail("rerun should use local files"))
    assert len(calls) == 2


def test_failed_download_does_not_stop_other_records(monkeypatch, tmp_path: Path) -> None:
    specs = [_spec("bad"), _spec("good")]
    monkeypatch.setattr(corpus, "load_proposed_corpus", lambda _: specs)
    monkeypatch.setattr(corpus, "inspect_pdf_bytes", lambda *_: {"page_count": 1, "title_word_overlap": 1.0})
    def fetch(url: str, timeout: float):
        if "bad" in url: raise OSError("offline")
        return _pdf_bytes(), "application/pdf", url
    records = corpus.build_corpus(proposal_path=tmp_path / "proposal.md", corpus_dir=tmp_path / "corpus", fetcher=fetch)
    assert [record["download_status"] for record in records] == ["download_failed", "verified"]
