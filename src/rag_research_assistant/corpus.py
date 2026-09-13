"""Reproducible, network-optional acquisition of the Phase 7.5 paper corpus."""

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

from pypdf import PdfReader


DEFAULT_PROPOSAL = Path("docs/proposed-phase-7.5-corpus.md")
DEFAULT_CORPUS_DIR = Path("data/corpus")
_EXISTING_FILES = {
    "rag_lewis_2020": "rag-lewis-2020.pdf",
    "dpr_karpukhin_2020": "dense-passage-retrieval.pdf",
    "sgpt": "sgpt-semantic-search.pdf",
}
_DIRECT_PDF_PREFIXES = (
    "https://arxiv.org/pdf/",
    "https://aclanthology.org/",
    "https://trec.nist.gov/",
)


@dataclass(frozen=True)
class PaperSpec:
    paper_id: str
    title: str
    authors: str
    year: int
    source_url: str
    primary_cluster: str
    existing: bool
    approved: bool

    @property
    def filename(self) -> str:
        return f"{self.paper_id}.pdf"


class CorpusValidationError(RuntimeError):
    """Raised when downloaded bytes are not a plausible expected PDF."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading a whole PDF in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_rows(proposal_path: Path) -> Iterable[List[str]]:
    for line in proposal_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ID") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8:
            yield cells


def _markdown_url(value: str) -> str:
    match = re.search(r"\((https?://[^)]+)\)", value)
    if not match:
        raise ValueError(f"manifest source is missing an HTTP URL: {value!r}")
    return match.group(1)


def load_proposed_corpus(proposal_path: Path = DEFAULT_PROPOSAL) -> List[PaperSpec]:
    """Derive a stable machine-readable corpus definition from the reviewed table."""
    records: List[PaperSpec] = []
    seen = set()
    for paper_id_cell, title, authors, year, source, cluster, _, _ in _table_rows(proposal_path):
        existing = "*(existing)*" in paper_id_cell
        paper_id = paper_id_cell.replace("*(existing)*", "").strip()
        if not re.fullmatch(r"[a-z0-9_]+", paper_id):
            raise ValueError(f"invalid paper ID: {paper_id!r}")
        if paper_id in seen:
            raise ValueError(f"duplicate paper ID: {paper_id}")
        seen.add(paper_id)
        source_url = _markdown_url(source)
        approved = source_url.startswith(_DIRECT_PDF_PREFIXES)
        records.append(PaperSpec(paper_id, title, authors, int(year), source_url, cluster, existing, approved))
    if len(records) != 57:
        raise ValueError(f"expected 57 proposed corpus records, found {len(records)}")
    if sum(record.approved for record in records) != 52:
        raise ValueError("expected 52 approved public-PDF records")
    return records


def _normal_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 1}


def inspect_pdf_bytes(data: bytes, expected_title: str) -> Dict[str, object]:
    """Reject HTML/empty/corrupt files and compare first-page text to the expected title."""
    if len(data) < 256 or not data.startswith(b"%PDF-"):
        raise CorpusValidationError("response is not a non-empty PDF file")
    try:
        reader = PdfReader(BytesIO(data))
        first_page = reader.pages[0].extract_text() or ""
    except Exception as exc:
        raise CorpusValidationError(f"PDF reader could not open response: {exc}") from exc
    expected_words = _normal_words(expected_title)
    found_words = _normal_words(first_page)
    overlap = len(expected_words & found_words) / max(1, len(expected_words))
    if expected_words and overlap < 0.55:
        raise CorpusValidationError(
            f"first-page title mismatch (word overlap {overlap:.0%}): {first_page[:160]!r}"
        )
    return {"page_count": len(reader.pages), "title_word_overlap": round(overlap, 3)}


def _http_fetch(url: str, timeout: float) -> Tuple[bytes, str, str]:
    request = Request(url, headers={"User-Agent": "rag-research-assistant-corpus/0.7"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310: source URLs are reviewed manifest entries
        content_type = response.headers.get_content_type()
        return response.read(), content_type, response.geturl()


def _manifest_record(spec: PaperSpec) -> Dict[str, object]:
    arxiv = re.search(r"arxiv\.org/pdf/([^/?]+)", spec.source_url)
    return {
        "paper_id": spec.paper_id,
        "title": spec.title,
        "authors": spec.authors,
        "year": spec.year,
        "primary_cluster": spec.primary_cluster,
        "source_type": "arXiv" if arxiv else "proceedings_or_canonical",
        "source_url": spec.source_url,
        "preferred_pdf_source": spec.source_url if spec.approved else None,
        "source_version": arxiv.group(1) if arxiv else None,
        "filename": spec.filename,
        "existing": spec.existing,
        "metadata_verified": True,
        "pdf_available": spec.approved,
        "download_status": "pending" if spec.approved else "held_source_unverified",
        "sha256": None,
        "file_size_bytes": None,
        "content_type": None,
        "final_resolved_url": None,
        "downloaded_at": None,
        "page_count": None,
        "title_word_overlap": None,
        "validation_notes": "",
    }


def _write_manifest(records: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_path.parent) as temporary:
        for record in records:
            temporary.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)


def _verify_existing(path: Path, expected_title: str) -> Dict[str, object]:
    info = inspect_pdf_bytes(path.read_bytes(), expected_title)
    return {**info, "sha256": sha256_file(path), "file_size_bytes": path.stat().st_size}


def build_corpus(
    *,
    proposal_path: Path = DEFAULT_PROPOSAL,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    existing_papers_dir: Path = Path("data/papers"),
    timeout: float = 30.0,
    fetcher: Callable[[str, float], Tuple[bytes, str, str]] = _http_fetch,
) -> List[Dict[str, object]]:
    """Reconcile existing PDFs, fetch approved ones, and atomically write JSONL state."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    pdf_dir = corpus_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for spec in load_proposed_corpus(proposal_path):
        result = _manifest_record(spec)
        target = pdf_dir / spec.filename
        try:
            if not spec.approved:
                result["validation_notes"] = "Held: no approved public PDF source."
            elif target.exists():
                info = _verify_existing(target, spec.title)
                result.update(info)
                result["download_status"] = "existing_verified" if spec.existing else "verified"
                result["validation_notes"] = "Validated local file; no network request made."
            elif spec.existing:
                source_name = _EXISTING_FILES.get(spec.paper_id)
                source = existing_papers_dir / source_name if source_name else None
                if source is None or not source.exists():
                    result["download_status"] = "validation_failed"
                    result["validation_notes"] = "Expected existing Phase 1--7 PDF was not found."
                else:
                    info = _verify_existing(source, spec.title)
                    shutil.copy2(source, target)
                    result.update(info)
                    result["download_status"] = "existing_verified"
                    result["validation_notes"] = f"Reconciled local source {source.name}; no redownload."
            else:
                data, content_type, final_url = fetcher(spec.source_url, timeout)
                info = inspect_pdf_bytes(data, spec.title)
                temporary = target.with_suffix(".pdf.part")
                temporary.write_bytes(data)
                temporary.replace(target)
                result.update(info)
                result.update({
                    "sha256": sha256_file(target), "file_size_bytes": target.stat().st_size,
                    "content_type": content_type, "final_resolved_url": final_url,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "download_status": "verified",
                    "validation_notes": "Downloaded, PDF-validated, and title-page matched.",
                })
        except Exception as exc:
            # Continue the corpus build; each failed source remains observable in JSONL.
            if target.exists() and result["download_status"] != "existing_verified":
                target.unlink()
            result["download_status"] = "validation_failed" if isinstance(exc, CorpusValidationError) else "download_failed"
            result["validation_notes"] = str(exc)
        results.append(result)

    checksums: Dict[str, List[str]] = {}
    for result in results:
        if result["sha256"]:
            checksums.setdefault(str(result["sha256"]), []).append(str(result["paper_id"]))
    for result in results:
        same = checksums.get(str(result["sha256"]), []) if result["sha256"] else []
        if len(same) > 1:
            result["download_status"] = "duplicate_checksum"
            result["validation_notes"] += f" Duplicate SHA-256 shared by: {', '.join(same)}."
    _write_manifest(results, corpus_dir / "manifest.jsonl")
    return results


def corpus_summary(records: Iterable[Dict[str, object]]) -> Dict[str, int]:
    records = list(records)
    return {
        "total": len(records),
        "approved": sum(bool(record["pdf_available"]) for record in records),
        "verified": sum(record["download_status"] in {"verified", "existing_verified"} for record in records),
        "existing_reconciled": sum(record["download_status"] == "existing_verified" for record in records),
        "held": sum(record["download_status"] == "held_source_unverified" for record in records),
        "failed": sum(record["download_status"] in {"download_failed", "validation_failed"} for record in records),
        "storage_bytes": sum(int(record["file_size_bytes"] or 0) for record in records),
        "checksummed": sum(record["sha256"] is not None for record in records),
        "duplicates": sum(record["download_status"] == "duplicate_checksum" for record in records),
    }
