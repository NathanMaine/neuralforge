"""AAAK fact-preserving compression for NeuralForge.

Three compression levels that reduce token count while preserving:
- Facts, numbers, and technical claims
- Code blocks and URLs
- Confidence qualifiers (may, likely, approximately, etc.)
- Expert attribution (names and citations)

Levels:
    0 -- No compression (passthrough)
    1 -- Light: filler removal + common abbreviations
    2 -- Aggressive: level 1 + sentence-level deduplication + trimming
"""
import re
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Filler words / phrases removed at level >= 1
# ------------------------------------------------------------------

_FILLER_PHRASES: list[str] = [
    "it is important to note that",
    "it should be noted that",
    "it is worth mentioning that",
    "it is worth noting that",
    "as a matter of fact",
    "in order to",
    "due to the fact that",
    "the fact that",
    "at the end of the day",
    "as we all know",
    "needless to say",
    "for all intents and purposes",
    "in the event that",
    "on the other hand",
    "in other words",
    "that being said",
    "with that being said",
    "it goes without saying",
    "basically",
    "essentially",
    "actually",
    "literally",
    "obviously",
    "clearly",
]

# Compiled pattern for filler removal (case-insensitive)
_FILLER_PATTERN = re.compile(
    "|".join(re.escape(f) for f in _FILLER_PHRASES),
    re.IGNORECASE,
)

# ------------------------------------------------------------------
# Abbreviations applied at level >= 1
# ------------------------------------------------------------------

_ABBREVIATIONS: dict[str, str] = {
    "for example": "e.g.",
    "for instance": "e.g.",
    "that is": "i.e.",
    "in other words": "i.e.",
    "and so on": "etc.",
    "and so forth": "etc.",
    "et cetera": "etc.",
    "approximately": "approx.",
    "approximately equal": "~=",
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "with respect to": "w.r.t.",
    "as soon as possible": "ASAP",
    "application programming interface": "API",
    "artificial intelligence": "AI",
    "machine learning": "ML",
    "deep learning": "DL",
    "natural language processing": "NLP",
    "large language model": "LLM",
    "graphics processing unit": "GPU",
    "central processing unit": "CPU",
}

# Compiled pattern for abbreviation replacement
_ABBREV_PATTERN = re.compile(
    "|".join(re.escape(k) for k in _ABBREVIATIONS),
    re.IGNORECASE,
)

# ------------------------------------------------------------------
# Protected patterns (never compressed)
# ------------------------------------------------------------------

# Code blocks: ```...``` or `...`
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

# URLs
_URL_RE = re.compile(r"https?://\S+")

# Confidence qualifiers we must preserve
_CONFIDENCE_QUALIFIERS = {
    "may", "might", "could", "possibly", "likely", "unlikely",
    "probably", "perhaps", "approximately", "roughly", "estimated",
    "uncertain", "tentatively", "arguably", "potentially",
    "reportedly", "allegedly", "seemingly",
}

# Expert attribution patterns
_ATTRIBUTION_RE = re.compile(
    r"(?:According to |as (?:noted|stated|argued|suggested|recommended) by )"
    r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compress(text: str, level: int = 1) -> str:
    """Compress text while preserving facts, code, URLs, and qualifiers.

    Parameters
    ----------
    text:
        The text to compress.
    level:
        Compression level (0=none, 1=light, 2=aggressive).

    Returns
    -------
    str
        The compressed text.
    """
    if level <= 0 or not text.strip():
        return text

    # Extract and protect code blocks & URLs
    protected: dict[str, str] = {}
    result = _protect(text, protected)

    # Level 1: filler removal + abbreviations
    result = _remove_filler(result)
    result = _apply_abbreviations(result)

    # Level 2: sentence deduplication + whitespace trimming
    if level >= 2:
        result = _deduplicate_sentences(result)
        result = _trim_whitespace(result)

    # Restore protected blocks
    result = _restore(result, protected)

    # Final cleanup: collapse multiple spaces/newlines
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()

    return result


def compression_ratio(original: str, compressed: str) -> float:
    """Calculate compression ratio (0.0 to 1.0, lower = more compressed).

    Returns
    -------
    float
        ``len(compressed) / len(original)``.  Returns 1.0 if original
        is empty.
    """
    if not original:
        return 1.0
    return len(compressed) / len(original)


def estimate_savings(original: str, compressed: str) -> dict:
    """Return compression statistics.

    Returns
    -------
    dict
        Keys: ``original_chars``, ``compressed_chars``, ``saved_chars``,
        ``ratio``, ``pct_saved``.
    """
    orig_len = len(original)
    comp_len = len(compressed)
    saved = orig_len - comp_len
    ratio = compression_ratio(original, compressed)
    return {
        "original_chars": orig_len,
        "compressed_chars": comp_len,
        "saved_chars": saved,
        "ratio": round(ratio, 4),
        "pct_saved": round((1 - ratio) * 100, 1) if orig_len > 0 else 0.0,
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

_PLACEHOLDER_PREFIX = "\x00PROTECTED_"


def _protect(text: str, store: dict[str, str]) -> str:
    """Replace code blocks and URLs with placeholders."""
    counter = 0

    def _replace(match: re.Match) -> str:
        nonlocal counter
        key = f"{_PLACEHOLDER_PREFIX}{counter}\x00"
        store[key] = match.group(0)
        counter += 1
        return key

    text = _CODE_BLOCK_RE.sub(_replace, text)
    text = _INLINE_CODE_RE.sub(_replace, text)
    text = _URL_RE.sub(_replace, text)
    # Protect attribution phrases
    text = _ATTRIBUTION_RE.sub(_replace, text)
    return text


def _restore(text: str, store: dict[str, str]) -> str:
    """Restore protected blocks from placeholders."""
    for key, value in store.items():
        text = text.replace(key, value)
    return text


def _remove_filler(text: str) -> str:
    """Remove filler words and phrases."""
    result = _FILLER_PATTERN.sub("", text)
    # Clean up orphaned commas and extra spaces from removals
    result = re.sub(r",\s*,", ",", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result


def _apply_abbreviations(text: str) -> str:
    """Replace common phrases with their abbreviations."""
    def _sub(match: re.Match) -> str:
        key = match.group(0).lower()
        return _ABBREVIATIONS.get(key, match.group(0))
    return _ABBREV_PATTERN.sub(_sub, text)


def _deduplicate_sentences(text: str) -> str:
    """Remove duplicate sentences (exact match, case-insensitive)."""
    lines = text.split("\n")
    result_lines: list[str] = []

    for line in lines:
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", line)
        seen: set[str] = set()
        unique: list[str] = []
        for sent in sentences:
            normalized = sent.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(sent)
        result_lines.append(" ".join(unique))

    return "\n".join(result_lines)


def _trim_whitespace(text: str) -> str:
    """Aggressively trim whitespace while preserving structure."""
    lines = text.split("\n")
    trimmed = [line.strip() for line in lines]
    # Remove empty lines that appear more than twice in a row
    result: list[str] = []
    empty_count = 0
    for line in trimmed:
        if not line:
            empty_count += 1
            if empty_count <= 1:
                result.append(line)
        else:
            empty_count = 0
            result.append(line)
    return "\n".join(result)
