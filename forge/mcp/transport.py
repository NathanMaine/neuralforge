"""MCP transport -- FastAPI router for JSON-RPC 2.0 over HTTP.

Supports both single and batch JSON-RPC requests at POST /mcp.
"""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from forge.mcp.server import (
    INVALID_REQUEST,
    PARSE_ERROR,
    _error_response,
    handle_batch,
    handle_request,
)

router = APIRouter(tags=["mcp"])
logger = logging.getLogger(__name__)


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    """JSON-RPC 2.0 endpoint for Model Context Protocol.

    Accepts single requests or batch arrays.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content=_error_response(None, PARSE_ERROR, "Invalid JSON"),
            status_code=200,  # JSON-RPC always returns 200
        )

    # Batch request
    if isinstance(body, list):
        if not body:
            return JSONResponse(
                content=_error_response(None, INVALID_REQUEST, "Empty batch"),
                status_code=200,
            )
        responses = await handle_batch(body)
        return JSONResponse(content=responses, status_code=200)

    # Single request
    if isinstance(body, dict):
        response = await handle_request(body)
        return JSONResponse(content=response, status_code=200)

    return JSONResponse(
        content=_error_response(None, INVALID_REQUEST, "Expected object or array"),
        status_code=200,
    )
