"""Comprehensive tests for forge.ingest.document_loader -- 20+ tests.

Tests each format handler, the dispatcher, and error cases.
"""
import csv
import os
import zipfile
import pytest
from unittest.mock import patch, MagicMock

from forge.ingest.document_loader import (
    load_document,
    load_pdf,
    load_docx,
    load_text,
    load_html,
    load_csv,
    LOADERS,
)


# ---------------------------------------------------------------------------
# load_text / load_text (markdown)
# ---------------------------------------------------------------------------


class TestLoadText:
    """Tests for load_text (txt and md)."""

    def test_load_txt(self, tmp_path):
        """Plain text file is loaded correctly."""
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!")
        result = load_text(str(f))
        assert result["text"] == "Hello, world!"
        assert result["title"] == "test"
        assert result["metadata"]["format"] == "txt"

    def test_load_md_with_heading(self, tmp_path):
        """Markdown file uses first heading as title."""
        f = tmp_path / "readme.md"
        f.write_text("# My Document\n\nSome content here.")
        result = load_text(str(f))
        assert result["title"] == "My Document"
        assert result["metadata"]["format"] == "md"

    def test_load_md_without_heading(self, tmp_path):
        """Markdown without heading uses filename as title."""
        f = tmp_path / "notes.md"
        f.write_text("No heading, just text.")
        result = load_text(str(f))
        assert result["title"] == "notes"

    def test_load_text_empty(self, tmp_path):
        """Empty text file returns empty text."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = load_text(str(f))
        assert result["text"] == ""

    def test_load_text_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/file.txt")

    def test_load_text_size_in_metadata(self, tmp_path):
        """Metadata includes file size."""
        f = tmp_path / "sized.txt"
        f.write_text("Some content")
        result = load_text(str(f))
        assert result["metadata"]["size_bytes"] > 0


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    """Tests for load_csv."""

    def test_load_csv_basic(self, tmp_path):
        """CSV file is loaded as text."""
        f = tmp_path / "data.csv"
        with open(str(f), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", "age", "city"])
            writer.writerow(["Alice", "30", "NYC"])
            writer.writerow(["Bob", "25", "LA"])
        result = load_csv(str(f))
        assert "Alice" in result["text"]
        assert "Bob" in result["text"]
        assert result["metadata"]["rows"] == 2
        assert result["metadata"]["columns"] == 3

    def test_load_csv_empty(self, tmp_path):
        """Empty CSV returns empty text."""
        f = tmp_path / "empty.csv"
        f.write_text("")
        result = load_csv(str(f))
        assert result["text"] == ""
        assert result["metadata"]["rows"] == 0

    def test_load_csv_header_only(self, tmp_path):
        """CSV with only headers returns header info."""
        f = tmp_path / "headers.csv"
        with open(str(f), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["col1", "col2"])
        result = load_csv(str(f))
        assert "Columns: col1, col2" in result["text"]
        assert result["metadata"]["rows"] == 0

    def test_load_csv_missing_file(self):
        """Missing CSV raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/data.csv")


# ---------------------------------------------------------------------------
# load_html
# ---------------------------------------------------------------------------


class TestLoadHtml:
    """Tests for load_html."""

    def test_load_html_basic(self, tmp_path):
        """HTML file is loaded and text extracted."""
        f = tmp_path / "page.html"
        f.write_text("<html><head><title>Test Page</title></head><body><p>Content</p></body></html>")

        with patch("trafilatura.extract", return_value="Content"):
            result = load_html(str(f))

        assert result["text"] == "Content"
        assert result["title"] == "Test Page"
        assert result["metadata"]["format"] == "html"

    def test_load_html_no_title(self, tmp_path):
        """HTML without title uses filename."""
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Content</p></body></html>")

        with patch("trafilatura.extract", return_value="Content"):
            result = load_html(str(f))

        assert result["title"] == "page"

    def test_load_html_extraction_fails(self, tmp_path):
        """Returns empty text when trafilatura returns None."""
        f = tmp_path / "bad.html"
        f.write_text("<html></html>")

        with patch("trafilatura.extract", return_value=None):
            result = load_html(str(f))

        assert result["text"] == ""

    def test_load_html_missing_file(self):
        """Missing HTML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_html("/nonexistent/page.html")


# ---------------------------------------------------------------------------
# load_pdf
# ---------------------------------------------------------------------------


class TestLoadPdf:
    """Tests for load_pdf."""

    def test_load_pdf_basic(self):
        """PDF text is extracted via PyMuPDF."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page 1 content"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.metadata = {"title": "Test PDF", "author": "Alice"}
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            with patch("os.path.isfile", return_value=True):
                result = load_pdf("/fake/test.pdf")

        assert "Page 1 content" in result["text"]
        assert result["title"] == "Test PDF"
        assert result["metadata"]["format"] == "pdf"

    def test_load_pdf_no_title_in_metadata(self):
        """PDF without title metadata uses filename stem."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Content"

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page])
        mock_doc.metadata = {}
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            with patch("os.path.isfile", return_value=True):
                result = load_pdf("/fake/document.pdf")

        assert result["title"] == "document"

    def test_load_pdf_missing_file(self):
        """Missing PDF raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pdf("/nonexistent/missing.pdf")

    def test_load_pdf_multiple_pages(self):
        """Multi-page PDF combines all page text."""
        pages = []
        for i in range(3):
            p = MagicMock()
            p.get_text.return_value = f"Page {i+1} content"
            pages.append(p)

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter(pages)
        mock_doc.metadata = {"title": "Multi"}
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            with patch("os.path.isfile", return_value=True):
                result = load_pdf("/fake/multi.pdf")

        assert result["metadata"]["pages"] == 3
        assert "Page 1 content" in result["text"]
        assert "Page 3 content" in result["text"]


# ---------------------------------------------------------------------------
# load_docx
# ---------------------------------------------------------------------------


class TestLoadDocx:
    """Tests for load_docx."""

    def test_load_docx_basic(self, tmp_path):
        """DOCX text is extracted from document.xml."""
        docx_path = tmp_path / "test.docx"

        with zipfile.ZipFile(str(docx_path), "w") as zf:
            zf.writestr(
                "word/document.xml",
                '<?xml version="1.0"?>'
                '<document><body><w:p><w:r><w:t>Hello World</w:t></w:r></w:p></body></document>',
            )
            zf.writestr(
                "docProps/core.xml",
                '<?xml version="1.0"?>'
                '<cp:coreProperties><dc:title>My Doc</dc:title></cp:coreProperties>',
            )

        result = load_docx(str(docx_path))
        assert "Hello World" in result["text"]
        assert result["title"] == "My Doc"
        assert result["metadata"]["format"] == "docx"

    def test_load_docx_no_title(self, tmp_path):
        """DOCX without title uses filename."""
        docx_path = tmp_path / "untitled.docx"

        with zipfile.ZipFile(str(docx_path), "w") as zf:
            zf.writestr(
                "word/document.xml",
                '<document><body><w:p><w:r><w:t>Content</w:t></w:r></w:p></body></document>',
            )

        result = load_docx(str(docx_path))
        assert result["title"] == "untitled"

    def test_load_docx_missing_file(self):
        """Missing DOCX raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_docx("/nonexistent/missing.docx")


# ---------------------------------------------------------------------------
# load_document (dispatcher)
# ---------------------------------------------------------------------------


class TestLoadDocument:
    """Tests for load_document dispatcher."""

    def test_dispatch_txt(self, tmp_path):
        """Dispatches .txt to load_text."""
        f = tmp_path / "file.txt"
        f.write_text("hello")
        result = load_document(str(f))
        assert result["text"] == "hello"

    def test_dispatch_md(self, tmp_path):
        """Dispatches .md to load_text."""
        f = tmp_path / "file.md"
        f.write_text("# Title\nContent")
        result = load_document(str(f))
        assert result["title"] == "Title"

    def test_unsupported_extension(self, tmp_path):
        """Unsupported extension raises ValueError."""
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(str(f))

    def test_loaders_dict_has_all_types(self):
        """LOADERS dict contains all expected extensions."""
        expected = {".pdf", ".docx", ".txt", ".md", ".html", ".csv"}
        assert set(LOADERS.keys()) == expected

    def test_dispatch_csv(self, tmp_path):
        """Dispatches .csv to load_csv."""
        f = tmp_path / "data.csv"
        with open(str(f), "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["a", "b"])
            writer.writerow(["1", "2"])
        result = load_document(str(f))
        assert result["metadata"]["format"] == "csv"
