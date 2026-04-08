"""Qdrant REST client for NeuralForge.

Provides synchronous helpers around the Qdrant HTTP API using httpx.
All calls are wrapped in try/except with logging and graceful fallbacks.
"""
import logging
from typing import Optional

import httpx

from forge.config import QDRANT_URL, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 30.0  # seconds
_BATCH_SIZE = 100


def _url(path: str) -> str:
    """Build a full Qdrant REST URL from a path fragment."""
    base = QDRANT_URL.rstrip("/")
    return f"{base}{path}"


def _collection_url(path: str = "") -> str:
    """Build a URL scoped to the configured collection."""
    return _url(f"/collections/{QDRANT_COLLECTION}{path}")


def _get(path: str, **kwargs) -> httpx.Response:
    """Issue an HTTP GET to the Qdrant REST API."""
    return httpx.get(_url(path), timeout=_TIMEOUT, **kwargs)


def _post(path: str, **kwargs) -> httpx.Response:
    """Issue an HTTP POST to the Qdrant REST API."""
    return httpx.post(_url(path), timeout=_TIMEOUT, **kwargs)


def _put(path: str, **kwargs) -> httpx.Response:
    """Issue an HTTP PUT to the Qdrant REST API."""
    return httpx.put(_url(path), timeout=_TIMEOUT, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_collection_info() -> Optional[dict]:
    """Return collection metadata, or ``None`` on any failure.

    Returns the ``result`` object from the Qdrant response which includes
    ``vectors_count``, ``points_count``, ``status``, ``config``, etc.
    """
    try:
        resp = _get(f"/collections/{QDRANT_COLLECTION}")
        resp.raise_for_status()
        data = resp.json()
        return data.get("result")
    except httpx.TimeoutException:
        logger.warning("Timeout fetching collection info from Qdrant")
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning("Qdrant returned %s for collection info", exc.response.status_code)
        return None
    except Exception:
        logger.exception("Unexpected error fetching collection info")
        return None


def get_total_chunks() -> int:
    """Return the total number of points (chunks) in the collection.

    Falls back to ``0`` on any error.
    """
    info = get_collection_info()
    if info is None:
        return 0
    return int(info.get("points_count", 0))


def get_status() -> str:
    """Return collection health as ``'green'``, ``'yellow'``, or ``'red'``.

    * ``green``  -- collection exists and is in status ``'green'``
    * ``yellow`` -- collection exists but status is not ``'green'``
    * ``red``    -- collection unreachable or missing
    """
    info = get_collection_info()
    if info is None:
        return "red"
    status = info.get("status", "").lower()
    if status == "green":
        return "green"
    return "yellow"


def count_chunks_for_expert(creator: str) -> int:
    """Count how many points belong to a specific expert/creator.

    Uses the Qdrant ``/count`` endpoint with a filter on the ``creator``
    payload field.  Returns ``0`` on any error.
    """
    payload = {
        "filter": {
            "must": [
                {"key": "creator", "match": {"value": creator}}
            ]
        },
        "exact": True,
    }
    try:
        resp = _post(
            f"/collections/{QDRANT_COLLECTION}/points/count",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return int(data.get("result", {}).get("count", 0))
    except httpx.TimeoutException:
        logger.warning("Timeout counting chunks for expert %r", creator)
        return 0
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Qdrant returned %s counting chunks for expert %r",
            exc.response.status_code,
            creator,
        )
        return 0
    except Exception:
        logger.exception("Unexpected error counting chunks for expert %r", creator)
        return 0


def get_all_expert_names() -> list[str]:
    """Return a deduplicated, sorted list of all expert/creator names.

    Scrolls through *all* points in the collection, extracting the
    ``creator`` payload field from each.  Returns an empty list on error.
    """
    experts: set[str] = set()
    offset: Optional[str | int] = None  # first call has no offset
    try:
        while True:
            payload: dict = {
                "limit": 100,
                "with_payload": ["creator"],
                "with_vector": False,
            }
            if offset is not None:
                payload["offset"] = offset

            resp = _post(
                f"/collections/{QDRANT_COLLECTION}/points/scroll",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json().get("result", {})
            points = data.get("points", [])
            for pt in points:
                creator = (pt.get("payload") or {}).get("creator")
                if creator:
                    experts.add(creator)

            next_offset = data.get("next_page_offset")
            if next_offset is None or not points:
                break
            offset = next_offset

        return sorted(experts)
    except httpx.TimeoutException:
        logger.warning("Timeout scrolling expert names from Qdrant")
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("Qdrant returned %s scrolling expert names", exc.response.status_code)
        return []
    except Exception:
        logger.exception("Unexpected error scrolling expert names")
        return []


def search_vectors(
    vector: list[float],
    limit: int = 10,
    expert: Optional[str] = None,
    must_filters: Optional[list[dict]] = None,
    should_filters: Optional[list[dict]] = None,
) -> list[dict]:
    """Search the collection by vector similarity.

    Parameters
    ----------
    vector:
        The query embedding.
    limit:
        Maximum number of results to return.
    expert:
        Optional creator/expert name to filter on.
    must_filters:
        Extra ``must`` filter conditions (Qdrant filter format).
    should_filters:
        Extra ``should`` filter conditions (Qdrant filter format).

    Returns
    -------
    list[dict]
        Each dict has keys: ``score``, ``expert``, ``title``, ``text``,
        ``source``, ``chunk_index``.  Empty list on error.
    """
    must: list[dict] = list(must_filters) if must_filters else []
    should: list[dict] = list(should_filters) if should_filters else []

    if expert:
        must.append({"key": "creator", "match": {"value": expert}})

    payload: dict = {
        "vector": vector,
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
    }

    qdrant_filter: dict = {}
    if must:
        qdrant_filter["must"] = must
    if should:
        qdrant_filter["should"] = should
    if qdrant_filter:
        payload["filter"] = qdrant_filter

    try:
        resp = _post(
            f"/collections/{QDRANT_COLLECTION}/points/search",
            json=payload,
        )
        resp.raise_for_status()
        raw_results = resp.json().get("result", [])
        results: list[dict] = []
        for hit in raw_results:
            p = hit.get("payload") or {}
            results.append({
                "score": hit.get("score", 0.0),
                "expert": p.get("creator", ""),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "source": p.get("source", ""),
                "chunk_index": p.get("chunk_index", 0),
            })
        return results
    except httpx.TimeoutException:
        logger.warning("Timeout during vector search")
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning("Qdrant returned %s during vector search", exc.response.status_code)
        return []
    except Exception:
        logger.exception("Unexpected error during vector search")
        return []


def upsert_points(points: list[dict]) -> int:
    """Upsert points into the collection in batches.

    Each point dict must have ``id``, ``vector``, and ``payload`` keys.
    Points are sent in batches of :data:`_BATCH_SIZE` (100).

    Returns the total number of successfully upserted points.
    """
    if not points:
        return 0

    upserted = 0
    for i in range(0, len(points), _BATCH_SIZE):
        batch = points[i : i + _BATCH_SIZE]
        payload = {"points": batch}
        try:
            resp = _put(
                f"/collections/{QDRANT_COLLECTION}/points",
                json=payload,
            )
            resp.raise_for_status()
            upserted += len(batch)
        except httpx.TimeoutException:
            logger.warning(
                "Timeout upserting batch %d-%d", i, i + len(batch)
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Qdrant returned %s upserting batch %d-%d",
                exc.response.status_code,
                i,
                i + len(batch),
            )
        except Exception:
            logger.exception(
                "Unexpected error upserting batch %d-%d", i, i + len(batch)
            )
    return upserted
