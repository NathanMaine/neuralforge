"""OpenAI-compatible proxy with auto-RAG + guardrails + provenance.

POST /v1/chat/completions -- auto-injects retrieved context, applies
guardrails, adds provenance, and routes to NIM.

GET /v1/models -- lists available models.

Supports X-NeuralForge-Bypass header to skip RAG injection.
Supports streaming and non-streaming responses.
"""
import json
import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from forge.core import nim_client
from forge.config import NIM_MODEL

router = APIRouter(tags=["proxy"])
logger = logging.getLogger(__name__)

# Bypass header name
_BYPASS_HEADER = "x-neuralforge-bypass"


@router.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": NIM_MODEL,
                "object": "model",
                "created": 1700000000,
                "owned_by": "neuralforge",
                "permission": [],
            },
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_neuralforge_bypass: Optional[str] = Header(None, alias="x-neuralforge-bypass"),
):
    """OpenAI-compatible chat completion with auto-RAG + guardrails.

    When X-NeuralForge-Bypass is set to any truthy value, the request
    is forwarded directly to NIM without context injection or guardrails.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
        )

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "messages is required", "type": "invalid_request_error"}},
        )

    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 1000)
    temperature = body.get("temperature", 0.7)

    # Check bypass
    bypass = x_neuralforge_bypass and x_neuralforge_bypass.lower() in ("true", "1", "yes")

    # --- Audit log ---
    _log_proxy_request(messages, stream=stream, bypass=bypass)

    if bypass:
        # Direct passthrough to NIM
        if stream:
            return await _stream_response(messages, max_tokens)
        result = await nim_client.chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        if result is None:
            return _nim_error_response()
        return JSONResponse(content=result)

    # --- Auto-RAG pipeline ---
    # Extract user query from the last user message
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        # No user message -- just forward
        if stream:
            return await _stream_response(messages, max_tokens)
        result = await nim_client.chat_completion(
            messages, max_tokens=max_tokens, temperature=temperature
        )
        if result is None:
            return _nim_error_response()
        return JSONResponse(content=result)

    # --- Input guardrails ---
    from forge.api.main import get_guardrails_engine
    guardrails = get_guardrails_engine()
    input_check = {"allowed": True, "reason": None, "scrubbed_query": user_query}

    if guardrails and guardrails.enabled:
        input_check = await guardrails.check_input(user_query)
        if not input_check["allowed"]:
            return _blocked_response(input_check.get("reason", "Input blocked by guardrails"))

    effective_query = input_check["scrubbed_query"]

    # --- Retrieve context ---
    context_text = ""
    experts_referenced = []
    try:
        from forge.api.main import get_graph_engine
        from forge.core import embeddings, qdrant_client
        from forge.layers.engine import get_context

        engine = get_graph_engine()

        async def _search_fn(q, limit=10, expert=None):
            vec = await embeddings.get_embedding(q)
            if vec is None:
                return []
            return qdrant_client.search_vectors(vec, limit=limit, expert=expert)

        ctx = await get_context(
            query=effective_query,
            max_tokens=2000,
            graph_engine=engine,
            search_fn=_search_fn,
        )
        context_text = ctx.as_text()
        experts_referenced = ctx.experts_referenced
    except Exception as exc:
        logger.warning("Context retrieval failed (non-fatal): %s", exc)

    # --- Inject context into messages ---
    augmented_messages = list(messages)
    if context_text:
        # Insert context as a system message before the user messages
        context_msg = {
            "role": "system",
            "content": context_text,
        }
        # Insert after existing system messages
        insert_idx = 0
        for i, msg in enumerate(augmented_messages):
            if msg.get("role") == "system":
                insert_idx = i + 1
            else:
                break
        augmented_messages.insert(insert_idx, context_msg)

    # --- Generate ---
    if stream:
        return await _stream_response(augmented_messages, max_tokens)

    result = await nim_client.chat_completion(
        augmented_messages, max_tokens=max_tokens, temperature=temperature
    )
    if result is None:
        return _nim_error_response()

    # --- Output guardrails ---
    response_text = ""
    try:
        response_text = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        pass

    if guardrails and guardrails.enabled and response_text:
        output_check = await guardrails.check_output(effective_query, response_text)
        if not output_check["allowed"]:
            return _blocked_response("Output blocked by guardrails")
        # Use the potentially modified response
        if output_check.get("response") != response_text:
            result["choices"][0]["message"]["content"] = output_check["response"]

    # --- Add provenance metadata ---
    if experts_referenced:
        provenance = {
            "experts_referenced": experts_referenced,
            "context_injected": bool(context_text),
            "guardrails_applied": guardrails.enabled if guardrails else False,
        }
        # Add as a custom field in the response
        result["neuralforge_provenance"] = provenance

    return JSONResponse(content=result)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

async def _stream_response(messages: list[dict], max_tokens: int):
    """Stream SSE chunks from NIM."""
    async def _generate():
        try:
            async for chunk in nim_client.stream_completion(messages, max_tokens=max_tokens):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc)
            error_chunk = {
                "error": {"message": str(exc), "type": "server_error"},
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _nim_error_response():
    """Return a 502 error when NIM is unreachable."""
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": "NIM backend unreachable or returned an error",
                "type": "server_error",
                "code": "nim_unavailable",
            }
        },
    )


def _blocked_response(reason: str):
    """Return a 403 response when guardrails block the request."""
    return JSONResponse(
        status_code=403,
        content={
            "error": {
                "message": reason,
                "type": "guardrails_blocked",
            }
        },
    )


def _log_proxy_request(messages: list[dict], stream: bool, bypass: bool) -> None:
    """Log proxy requests for audit."""
    msg_count = len(messages)
    roles = [m.get("role", "?") for m in messages]
    logger.info(
        "Proxy request: %d messages (%s), stream=%s, bypass=%s",
        msg_count,
        ", ".join(roles),
        stream,
        bypass,
    )
