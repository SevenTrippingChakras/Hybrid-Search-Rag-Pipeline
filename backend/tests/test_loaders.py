"""Tests for the multi-format document loaders."""

import pytest

from app.rag.loaders import load_file


def test_markdown_splits_on_headings(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text("# One\n\nAlpha text.\n\n## Two\n\nBeta text.\n")
    segments = load_file(path)

    assert [s.heading for s in segments] == ["One", "Two"]
    assert segments[0].text == "Alpha text."
    assert segments[1].text == "Beta text."
    assert all(s.source == "doc.md" and s.page is None for s in segments)


def test_text_is_single_segment(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Line one.\nLine two.")
    segments = load_file(path)

    assert len(segments) == 1
    assert segments[0].heading is None
    assert segments[0].text == "Line one.\nLine two."


def test_html_splits_on_headings_and_folds_lists(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<h1>Title</h1><p>Intro.</p>"
        "<h2>Sub</h2><p>Body.</p><ul><li>a</li><li>b</li></ul>"
    )
    segments = load_file(path)

    assert [s.heading for s in segments] == ["Title", "Sub"]
    assert segments[1].text == "Body.\na\nb"


def test_normalize_collapses_blank_lines_and_trailing_space(tmp_path):
    path = tmp_path / "messy.txt"
    path.write_text("Hello   \n\n\n\nWorld  ")
    (segment,) = load_file(path)

    assert segment.text == "Hello\n\nWorld"


def test_pdf_yields_one_segment_per_page(tmp_path):
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "doc.pdf"
    pdf = reportlab.Canvas(str(path))
    pdf.drawString(72, 720, "Page one body.")
    pdf.showPage()
    pdf.drawString(72, 720, "Page two body.")
    pdf.showPage()
    pdf.save()

    segments = load_file(path)

    assert [s.page for s in segments] == [1, 2]
    assert "Page one" in segments[0].text
    assert all(s.heading is None for s in segments)


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("whatever")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(path)
