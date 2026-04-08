"""Document loading for PDF, DOCX, TXT, HTML, CSV.

Each loader returns a dict with ``text``, ``title``, and ``metadata`` keys.
The top-level :func:`load_document` dispatches to the correct loader
based on file extension.
"""
import csv
import io
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_pdf(filepath: str) -> dict:
    """Extract text from a PDF using PyMuPDF (fitz).

    Parameters
    ----------
    filepath:
        Path to the PDF file.

    Returns
    -------
    dict
        ``{text: str, title: str, metadata: dict}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If PyMuPDF is not installed.
    """
    _check_exists(filepath)
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required to load PDFs: pip install pymupdf")

    doc = fitz.open(filepath)
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if text and text.strip():
            pages.append(text.strip())

    full_text = "\n\n".join(pages)

    # Try to get title from metadata
    meta = doc.metadata or {}
    title = meta.get("title", "") or Path(filepath).stem

    doc.close()

    return {
        "text": full_text,
        "title": title,
        "metadata": {
            "format": "pdf",
            "pages": len(pages),
            "filepath": filepath,
            **{k: v for k, v in meta.items() if v},
        },
    }


def load_docx(filepath: str) -> dict:
    """Extract text from a DOCX file.

    Uses ``zipfile`` to read the underlying XML directly, avoiding
    a dependency on ``python-docx`` while still extracting paragraph text.

    Parameters
    ----------
    filepath:
        Path to the DOCX file.

    Returns
    -------
    dict
        ``{text: str, title: str, metadata: dict}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    import re
    import zipfile

    _check_exists(filepath)

    paragraphs: list[str] = []
    title = Path(filepath).stem

    with zipfile.ZipFile(filepath, "r") as zf:
        # Extract document.xml for body text
        if "word/document.xml" in zf.namelist():
            xml = zf.read("word/document.xml").decode("utf-8")
            # Extract text from <w:t> elements
            texts = re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.DOTALL)
            # Group by paragraphs (rough: split on </w:p>)
            raw = " ".join(texts)
            if raw.strip():
                paragraphs.append(raw.strip())

        # Try to get title from core.xml
        if "docProps/core.xml" in zf.namelist():
            core_xml = zf.read("docProps/core.xml").decode("utf-8")
            title_match = re.search(
                r"<dc:title>(.*?)</dc:title>", core_xml, re.DOTALL
            )
            if title_match and title_match.group(1).strip():
                title = title_match.group(1).strip()

    full_text = "\n\n".join(paragraphs)

    return {
        "text": full_text,
        "title": title,
        "metadata": {
            "format": "docx",
            "filepath": filepath,
        },
    }


def load_text(filepath: str) -> dict:
    """Load a plain text or markdown file.

    Parameters
    ----------
    filepath:
        Path to the .txt or .md file.

    Returns
    -------
    dict
        ``{text: str, title: str, metadata: dict}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    _check_exists(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    ext = Path(filepath).suffix.lower()
    title = Path(filepath).stem

    # For markdown, try to extract first heading as title
    if ext == ".md":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped.lstrip("# ").strip()
                break

    return {
        "text": text,
        "title": title,
        "metadata": {
            "format": ext.lstrip("."),
            "filepath": filepath,
            "size_bytes": os.path.getsize(filepath),
        },
    }


def load_html(filepath: str) -> dict:
    """Extract text from an HTML file using trafilatura.

    Parameters
    ----------
    filepath:
        Path to the HTML file.

    Returns
    -------
    dict
        ``{text: str, title: str, metadata: dict}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ImportError
        If trafilatura is not installed.
    """
    import re

    _check_exists(filepath)

    try:
        import trafilatura
    except ImportError:
        raise ImportError("trafilatura is required to load HTML: pip install trafilatura")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()

    text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
    )
    if text is None:
        text = ""

    # Try to extract title
    title = Path(filepath).stem
    title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE)
    if title_match and title_match.group(1).strip():
        title = title_match.group(1).strip()

    return {
        "text": text,
        "title": title,
        "metadata": {
            "format": "html",
            "filepath": filepath,
        },
    }


def load_csv(filepath: str) -> dict:
    """Load a CSV file as a text representation.

    Converts the CSV into a readable text format with column headers
    and row data for ingestion into the knowledge graph.

    Parameters
    ----------
    filepath:
        Path to the CSV file.

    Returns
    -------
    dict
        ``{text: str, title: str, metadata: dict}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    _check_exists(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return {
            "text": "",
            "title": Path(filepath).stem,
            "metadata": {"format": "csv", "filepath": filepath, "rows": 0, "columns": 0},
        }

    headers = rows[0]
    lines: list[str] = []
    lines.append("Columns: " + ", ".join(headers))
    lines.append("")

    for i, row in enumerate(rows[1:], start=1):
        parts = []
        for header, value in zip(headers, row):
            parts.append(f"{header}: {value}")
        lines.append(f"Row {i}: " + " | ".join(parts))

    full_text = "\n".join(lines)

    return {
        "text": full_text,
        "title": Path(filepath).stem,
        "metadata": {
            "format": "csv",
            "filepath": filepath,
            "rows": len(rows) - 1,
            "columns": len(headers),
        },
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_text,
    ".md": load_text,
    ".html": load_html,
    ".csv": load_csv,
}


def load_document(filepath: str) -> dict:
    """Load any supported document by dispatching on file extension.

    Supported extensions: ``.pdf``, ``.docx``, ``.txt``, ``.md``,
    ``.html``, ``.csv``.

    Parameters
    ----------
    filepath:
        Path to the document file.

    Returns
    -------
    dict
        ``{text: str, title: str, metadata: dict}``

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported.
    """
    ext = Path(filepath).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        supported = ", ".join(sorted(LOADERS.keys()))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )
    return loader(filepath)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_exists(filepath: str) -> None:
    """Raise FileNotFoundError if the file does not exist."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
