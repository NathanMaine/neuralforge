"""Shared utility functions."""
import re
import hashlib
from datetime import datetime


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug.

    Lowercases, strips non-alphanumeric characters (except hyphens),
    collapses consecutive hyphens, and trims leading/trailing hyphens.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    return text


def content_hash(text: str) -> str:
    """Return the MD5 hex digest of the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    """Return the current datetime as an ISO 8601 string."""
    return datetime.now().isoformat()


def human_size(nbytes: int | float) -> str:
    """Convert a byte count into a human-readable string (e.g. '1.5 GB').

    Uses binary-style thresholds (1024) with decimal labels.
    """
    if nbytes < 0:
        raise ValueError("Byte count must be non-negative")
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(nbytes)} B"
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    # Should never reach here, but just in case
    return f"{nbytes:.1f} TB"


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in the given text.

    Uses the heuristic of ~0.75 tokens per character / ~1.33 tokens per word.
    More precisely: split on whitespace, count words, multiply by 1.33.
    """
    if not text or not text.strip():
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.33))
