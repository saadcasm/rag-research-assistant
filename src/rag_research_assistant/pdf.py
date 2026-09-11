"""PDF discovery and page-level text extraction."""

from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from .models import PageText


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened or read."""


def discover_pdfs(input_dir: Path) -> List[Path]:
    """Return PDFs in deterministic filename order."""

    if not input_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {input_dir}")
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )


def extract_pdf(path: Path) -> List[PageText]:
    """Extract one PageText record per page, using one-based page numbers."""

    try:
        reader = PdfReader(path)
        return [
            PageText(
                document=path.name,
                page_number=page_number,
                text=page.extract_text() or "",
            )
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:
        raise PdfExtractionError(f"Could not extract text from {path.name}: {exc}") from exc


def extract_directory(input_dir: Path) -> Iterable[PageText]:
    """Yield pages from all PDFs in an input directory."""

    for path in discover_pdfs(input_dir):
        yield from extract_pdf(path)

