from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_research_assistant.pdf import discover_pdfs, extract_pdf


def test_discover_pdfs_is_case_insensitive_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "zeta.PDF").touch()
    (tmp_path / "Alpha.pdf").touch()
    (tmp_path / "notes.txt").touch()

    assert [path.name for path in discover_pdfs(tmp_path)] == ["Alpha.pdf", "zeta.PDF"]


def test_extract_pdf_returns_text_with_page_metadata(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Research paper text.) Tj ET")
    page[NameObject("/Resources")] = resources
    page[NameObject("/Contents")] = writer._add_object(stream)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    pages = extract_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0].document == "paper.pdf"
    assert pages[0].page_number == 1
    assert "Research paper text." in pages[0].text
