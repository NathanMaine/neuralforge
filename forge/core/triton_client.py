"""Triton Inference Server HTTP client for embeddings and reranking."""
import logging
from typing import Any

import httpx

from forge.config import (
    TRITON_URL,
    EMBED_MODEL,
    RERANK_MODEL,
    EMBED_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# Shared async client — created lazily per event-loop via _get_client()
_client: httpx.AsyncClient | None = None

_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


async def _get_client() -> httpx.AsyncClient:
    """Return (and lazily create) a module-level async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


def _build_text_payload(texts: list[str]) -> dict[str, Any]:
    """Build the Triton v2 inference request payload for TEXT inputs."""
    return {
        "inputs": [
            {
                "name": "TEXT",
                "shape": [len(texts)],
                "datatype": "BYTES",
                "data": texts,
            }
        ]
    }


def _build_rerank_payload(query: str, documents: list[str]) -> dict[str, Any]:
    """Build the Triton v2 inference request for a rerank model.

    Sends two inputs: QUERY (single string) and DOCUMENTS (list of strings).
    """
    return {
        "inputs": [
            {
                "name": "QUERY",
                "shape": [1],
                "datatype": "BYTES",
                "data": [query],
            },
            {
                "name": "DOCUMENTS",
                "shape": [len(documents)],
                "datatype": "BYTES",
                "data": documents,
            },
        ]
    }


async def infer_embedding(texts: list[str]) -> list[list[float]] | None:
    """Batch embed via Triton ``/v2/models/{model}/infer`` endpoint.

    Automatically chunks the input list into batches of
    ``EMBED_BATCH_SIZE`` to stay within Triton limits.

    Returns a list of embedding vectors (one per input text), or *None*
    if the request fails.
    """
    if not texts:
        return []

    client = await _get_client()
    url = f"{TRITON_URL}/v2/models/{EMBED_MODEL}/infer"
    all_embeddings: list[list[float]] = []

    try:
        # Process in batches
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            payload = _build_text_payload(batch)

            resp = await client.post(url, json=payload)
            resp.raise_for_status()

            data = resp.json()
            outputs = data.get("outputs", [])
            if not outputs:
                logger.error("Triton embedding response missing 'outputs': %s", data)
                return None

            # Triton returns a flat list; reshape by output shape
            raw = outputs[0].get("data", [])
            shape = outputs[0].get("shape", [])

            if len(shape) == 2:
                dim = shape[1]
                embeddings = [
                    raw[i * dim : (i + 1) * dim] for i in range(shape[0])
                ]
            elif len(shape) == 1:
                # Single-vector response (one text)
                embeddings = [raw]
            else:
                logger.error("Unexpected output shape from Triton: %s", shape)
                return None

            all_embeddings.extend(embeddings)

        return all_embeddings

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Triton embedding HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        return None
    except httpx.TimeoutException:
        logger.error("Triton embedding request timed out")
        return None
    except Exception:
        logger.exception("Unexpected error during Triton embedding inference")
        return None


async def infer_rerank(query: str, documents: list[str]) -> list[float] | None:
    """Rerank documents against a query via Triton.

    Returns a list of relevance scores (one per document), or *None*
    if the request fails.
    """
    if not documents:
        return []
    if not query:
        logger.warning("infer_rerank called with empty query")
        return None

    client = await _get_client()
    url = f"{TRITON_URL}/v2/models/{RERANK_MODEL}/infer"

    try:
        payload = _build_rerank_payload(query, documents)
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

        data = resp.json()
        outputs = data.get("outputs", [])
        if not outputs:
            logger.error("Triton rerank response missing 'outputs': %s", data)
            return None

        scores: list[float] = [float(s) for s in outputs[0].get("data", [])]
        return scores

    except httpx.HTTPStatusError as exc:
        logger.error(
            "Triton rerank HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        return None
    except httpx.TimeoutException:
        logger.error("Triton rerank request timed out")
        return None
    except Exception:
        logger.exception("Unexpected error during Triton rerank inference")
        return None


async def close_client() -> None:
    """Shut down the shared HTTP client (call at app shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
