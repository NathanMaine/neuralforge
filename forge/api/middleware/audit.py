"""Query audit logging middleware -- writes JSONL to data/logs/audit.jsonl."""
import json
import logging
import os
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from forge.config import LOG_DIR

logger = logging.getLogger(__name__)

_AUDIT_FILE = os.path.join(LOG_DIR, "audit.jsonl")


class AuditMiddleware(BaseHTTPMiddleware):
    """Logs every request as a JSONL line for audit and debugging."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response: Response | None = None
        error: str | None = None

        try:
            response = await call_next(request)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            status = response.status_code if response else 500

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": request.method,
                "path": str(request.url.path),
                "query": str(request.url.query) if request.url.query else None,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "client": request.client.host if request.client else None,
                "error": error,
            }

            _write_audit(entry)

        return response


def _write_audit(entry: dict) -> None:
    """Append an audit entry to the JSONL log file."""
    try:
        os.makedirs(os.path.dirname(_AUDIT_FILE), exist_ok=True)
        with open(_AUDIT_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write audit log: %s", exc)
