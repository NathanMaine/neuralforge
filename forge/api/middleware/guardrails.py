"""NeMo Guardrails middleware for query processing.

Wraps incoming search/proxy queries through the guardrails engine
for input validation (PII scrub, jailbreak detection) and output
validation (hallucination check, attribution).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def check_query_input(
    query: str,
    guardrails_engine=None,
) -> dict:
    """Run input guardrails on a query.

    Returns
    -------
    dict
        ``{allowed: bool, reason: str | None, scrubbed_query: str}``
    """
    if guardrails_engine is None or not guardrails_engine.enabled:
        return {"allowed": True, "reason": None, "scrubbed_query": query}

    return await guardrails_engine.check_input(query)


async def check_response_output(
    query: str,
    response: str,
    guardrails_engine=None,
    context: Optional[dict] = None,
) -> dict:
    """Run output guardrails on a response.

    Returns
    -------
    dict
        ``{allowed: bool, response: str, provenance: dict}``
    """
    if guardrails_engine is None or not guardrails_engine.enabled:
        return {"allowed": True, "response": response, "provenance": {}}

    return await guardrails_engine.check_output(query, response, context)


async def guarded_pipeline(
    query: str,
    guardrails_engine=None,
    generate_fn=None,
    context: Optional[dict] = None,
) -> dict:
    """Full guarded generation: input -> generate -> output.

    Returns
    -------
    dict
        ``{response: str, input_check: dict, output_check: dict}``
    """
    if guardrails_engine is None or not guardrails_engine.enabled:
        # No guardrails -- just generate
        if generate_fn is not None:
            response = await generate_fn(query, context or {})
        else:
            response = query
        return {
            "response": response,
            "input_check": {"allowed": True, "reason": None, "scrubbed_query": query},
            "output_check": {"allowed": True, "response": response, "provenance": {}},
        }

    return await guardrails_engine.guarded_generate(
        query=query,
        context=context,
        generate_fn=generate_fn,
    )
