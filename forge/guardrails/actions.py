"""Custom guardrail actions for NeuralForge.

These actions are registered with NeMo Guardrails and called from
Colang flows.  Each action can be invoked standalone (for testing)
or via the rails engine.
"""
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# PII patterns (re-used from forge.ingest.pii_scrubber for consistency)
_PII_PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


async def check_hallucination(
    context: dict,
    response: str,
    graph_engine=None,
) -> bool:
    """Verify expert citations exist in knowledge graph.

    Returns ``True`` if no hallucinated experts are detected (i.e. all
    cited experts exist), ``False`` if any fabricated expert names are
    found.
    """
    if graph_engine is None:
        # No graph engine available -- skip check
        return True

    # Extract quoted names that look like expert citations
    cited_names = _extract_expert_names(response)
    if not cited_names:
        return True

    for name in cited_names:
        if not graph_engine.has_expert(name):
            logger.warning("Hallucinated expert detected: %r", name)
            return False

    return True


async def check_attribution(
    context: dict,
    response: str,
    graph_engine=None,
) -> bool:
    """Ensure every expert name in response is a real graph node.

    Similar to :func:`check_hallucination` but returns attribution
    details for auditing.
    """
    if graph_engine is None:
        return True

    cited = _extract_expert_names(response)
    for name in cited:
        if not graph_engine.has_expert(name):
            logger.warning("Unattributable expert reference: %r", name)
            return False
    return True


async def add_provenance(
    context: dict,
    response: str,
    chunks_used: list[dict] | None = None,
) -> str:
    """Append provenance metadata to response.

    Adds a ``[Sources]`` footer listing the chunks that were used to
    generate the response, with expert names and chunk indices.
    """
    if not chunks_used:
        return response

    sources: list[str] = []
    seen: set[str] = set()
    for chunk in chunks_used:
        expert = chunk.get("expert", "unknown")
        title = chunk.get("title", "")
        key = f"{expert}:{title}"
        if key not in seen:
            seen.add(key)
            sources.append(f"  - {expert}: {title}" if title else f"  - {expert}")

    if sources:
        footer = "\n\n[Sources]\n" + "\n".join(sources)
        return response + footer

    return response


async def scrub_pii_input(
    context: dict,
    text: str,
) -> tuple[str, dict]:
    """Scrub PII from input query.

    Returns
    -------
    tuple[str, dict]
        ``(scrubbed_text, pii_counts)`` where *pii_counts* maps each
        PII type to the number of occurrences removed.
    """
    counts: dict[str, int] = {}
    scrubbed = text

    for pii_type, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(scrubbed)
        if matches:
            counts[pii_type] = len(matches)
            scrubbed = pattern.sub("[REDACTED]", scrubbed)

    return scrubbed, counts


async def self_correction(
    context: dict,
    response: str,
    hallucination_detected: bool,
    generate_fn=None,
    max_retries: int = 2,
) -> str:
    """Re-query with tighter constraints if hallucination detected.

    Performs up to ``max_retries`` attempts to get a non-hallucinated
    response using a stricter prompt.

    Parameters
    ----------
    context:
        The retrieval context dict.
    response:
        The original (potentially hallucinated) response.
    hallucination_detected:
        Whether hallucination was detected in the response.
    generate_fn:
        Async callable ``(query, context) -> str``.
    max_retries:
        Maximum number of correction attempts.
    """
    if not hallucination_detected:
        return response

    if generate_fn is None:
        return response

    query = context.get("query", "")
    correction_prefix = (
        "IMPORTANT: Only cite experts and facts that appear in the provided context. "
        "Do not invent names or references. "
    )

    corrected = response
    for attempt in range(1, max_retries + 1):
        try:
            corrected = await generate_fn(
                correction_prefix + query,
                context,
            )
            # If we got a response, return it (caller can re-check)
            logger.info("Self-correction attempt %d/%d produced new response", attempt, max_retries)
            return corrected
        except Exception as exc:
            logger.warning("Self-correction attempt %d/%d failed: %s", attempt, max_retries, exc)

    return corrected


async def log_rail_decision(
    rail_name: str,
    decision: bool,
    reason: str,
    query: str = "",
) -> None:
    """Audit log for rail decisions.

    Parameters
    ----------
    rail_name:
        Name of the rail (e.g. ``input_pii``, ``output_hallucination``).
    decision:
        ``True`` if allowed, ``False`` if blocked.
    reason:
        Human-readable reason for the decision.
    query:
        The original query (for audit context).
    """
    ts = datetime.now(timezone.utc).isoformat()
    status = "ALLOWED" if decision else "BLOCKED"
    logger.info(
        "[RAIL_AUDIT] %s | rail=%s | status=%s | reason=%s | query=%.100s",
        ts,
        rail_name,
        status,
        reason,
        query,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _extract_expert_names(text: str) -> list[str]:
    """Extract likely expert names from response text.

    Looks for patterns like ``According to <Name>``, ``<Name> suggests``,
    or names in quotes after attribution verbs.
    """
    names: list[str] = []

    # "According to Name" pattern
    for match in re.finditer(r"[Aa]ccording to ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text):
        names.append(match.group(1))

    # "Name suggests/recommends/states/argues/claims" pattern
    for match in re.finditer(
        r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(?:suggests?|recommends?|states?|argues?|claims?|notes?|explains?)",
        text,
    ):
        names.append(match.group(1))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    return unique
