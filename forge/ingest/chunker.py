"""Text chunker with configurable size and overlap."""
import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into chunks of approximately *chunk_size* characters.

    Parameters
    ----------
    text:
        The input text to chunk.
    chunk_size:
        Target number of characters per chunk.
    overlap:
        Number of characters to overlap between consecutive chunks.
        Must be less than *chunk_size*.

    Returns
    -------
    list[str]
        List of non-empty text chunks.

    Raises
    ------
    ValueError
        If *chunk_size* < 1 or *overlap* >= *chunk_size* or *overlap* < 0.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    if not text or not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)
    step = chunk_size - overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks
