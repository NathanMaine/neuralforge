"""MCP server -- JSON-RPC 2.0 handler with TOOL_DISPATCH.

Implements the Model Context Protocol for NeuralForge, providing
17+ tools for search, graph queries, context retrieval, and ingestion.
"""
import logging
import traceback
from typing import Any, Optional

from forge.mcp.tools import TOOL_DISPATCH, TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """JSON-RPC error with code and message."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _error_response(
    req_id: Any, code: int, message: str, data: Any = None
) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    resp: dict = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": req_id,
    }
    if data is not None:
        resp["error"]["data"] = data
    return resp


def _success_response(req_id: Any, result: Any) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": req_id,
    }


async def handle_request(request: dict) -> dict:
    """Handle a single JSON-RPC 2.0 request.

    Supports methods:
    - ``initialize`` / ``initialized`` -- MCP handshake
    - ``tools/list`` -- list available tools
    - ``tools/call`` -- invoke a tool by name

    Parameters
    ----------
    request:
        Parsed JSON-RPC 2.0 request dict.

    Returns
    -------
    dict
        JSON-RPC 2.0 response dict.
    """
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params", {})

    if not method:
        return _error_response(req_id, INVALID_REQUEST, "Missing 'method'")

    try:
        if method == "initialize":
            return _success_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "neuralforge-mcp",
                    "version": "1.0.0",
                },
            })

        elif method == "notifications/initialized":
            # Notification -- no response needed, but return ack for convenience
            return _success_response(req_id, {"acknowledged": True})

        elif method == "tools/list":
            return _success_response(req_id, {
                "tools": TOOL_DEFINITIONS,
            })

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            if not tool_name:
                return _error_response(req_id, INVALID_PARAMS, "Missing tool 'name'")

            if tool_name not in TOOL_DISPATCH:
                return _error_response(
                    req_id,
                    METHOD_NOT_FOUND,
                    f"Unknown tool: {tool_name}",
                    {"available_tools": list(TOOL_DISPATCH.keys())},
                )

            handler = TOOL_DISPATCH[tool_name]
            try:
                result = await handler(**tool_args)
                return _success_response(req_id, {
                    "content": [{"type": "text", "text": str(result)}],
                    "isError": False,
                })
            except TypeError as exc:
                return _error_response(
                    req_id, INVALID_PARAMS,
                    f"Invalid arguments for tool '{tool_name}': {exc}",
                )
            except Exception as exc:
                logger.exception("Tool '%s' failed: %s", tool_name, exc)
                return _success_response(req_id, {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "isError": True,
                })

        else:
            return _error_response(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

    except Exception as exc:
        logger.exception("Internal error handling request: %s", exc)
        return _error_response(req_id, INTERNAL_ERROR, str(exc))


async def handle_batch(requests: list[dict]) -> list[dict]:
    """Handle a batch of JSON-RPC 2.0 requests.

    Parameters
    ----------
    requests:
        List of JSON-RPC 2.0 request dicts.

    Returns
    -------
    list[dict]
        List of JSON-RPC 2.0 response dicts.
    """
    responses = []
    for req in requests:
        resp = await handle_request(req)
        responses.append(resp)
    return responses
