"""PII scrubber -- detect and remove personally identifiable information.

Supports SSN, email, phone, credit card, and IP address patterns.
"""
import re
import logging

logger = logging.getLogger(__name__)

PII_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}


def scrub_pii(text: str, replacement: str = "[REDACTED]") -> tuple[str, dict[str, int]]:
    """Remove PII from text.

    Parameters
    ----------
    text:
        Input text to scrub.
    replacement:
        String to replace PII matches with.

    Returns
    -------
    tuple[str, dict[str, int]]
        ``(scrubbed_text, counts)`` where *counts* maps each PII type
        to the number of occurrences removed.
    """
    counts: dict[str, int] = {}
    scrubbed = text

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(scrubbed)
        if matches:
            counts[pii_type] = len(matches)
            scrubbed = pattern.sub(replacement, scrubbed)

    return scrubbed, counts


def detect_pii(text: str) -> dict[str, list[str]]:
    """Detect PII without removing it.

    Parameters
    ----------
    text:
        Input text to scan.

    Returns
    -------
    dict[str, list[str]]
        Maps each PII type to a list of matched strings found.
        Only includes types with at least one match.
    """
    found: dict[str, list[str]] = {}

    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[pii_type] = matches

    return found
