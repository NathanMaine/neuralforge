"""Embedding client with TTL cache and batch support.

Uses :mod:`forge.core.triton_client` as the backend for generating
embeddings and provides an in-memory cache with time-to-live eviction
and maximum-size limits.
"""
import logging
import time

from forge.config import EMBED_CACHE_TTL, EMBED_CACHE_MAX
from forge.core.triton_client import infer_embedding

logger = logging.getLogger(__name__)

# Cache: text -> (embedding_vector, timestamp)
_cache: dict[str, tuple[list[float], float]] = {}


def _is_valid(entry: tuple[list[float], float]) -> bool:
    """Return True if the cache entry has not expired."""
    return (time.monotonic() - entry[1]) < EMBED_CACHE_TTL


def _evict_expired() -> None:
    """Remove all expired entries from the cache."""
    now = time.monotonic()
    expired = [k for k, (_, ts) in _cache.items() if (now - ts) >= EMBED_CACHE_TTL]
    for k in expired:
        del _cache[k]


def _evict_oldest(need_space: int = 1) -> None:
    """Evict the oldest entries to make room for *need_space* new items.

    First removes expired entries.  If still over capacity, drops the
    oldest entries by timestamp.
    """
    _evict_expired()
    overflow = len(_cache) + need_space - EMBED_CACHE_MAX
    if overflow <= 0:
        return
    # Sort by timestamp ascending (oldest first), evict overflow count
    by_age = sorted(_cache.items(), key=lambda kv: kv[1][1])
    for key, _ in by_age[:overflow]:
        del _cache[key]


async def get_embedding(text: str) -> list[float] | None:
    """Return the embedding for *text*, serving from cache when possible.

    Returns *None* if Triton is unreachable or returns an error.
    """
    # Cache hit?
    cached = _cache.get(text)
    if cached is not None and _is_valid(cached):
        return cached[0]

    # Cache miss — call Triton
    result = await infer_embedding([text])
    if result is None or len(result) == 0:
        return None

    vec = result[0]
    _evict_oldest(need_space=1)
    _cache[text] = (vec, time.monotonic())
    return vec


async def get_embeddings_batch(texts: list[str]) -> list[list[float] | None]:
    """Batch embed a list of texts.

    Texts with valid cache entries are returned immediately.  Cache
    misses are collected and sent to Triton in a single batched request.
    The results are merged back in the original order.

    Returns a list with one entry per input text.  Individual entries
    are *None* if the embedding could not be obtained.
    """
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    # Phase 1: check cache
    for i, text in enumerate(texts):
        cached = _cache.get(text)
        if cached is not None and _is_valid(cached):
            results[i] = cached[0]
        else:
            miss_indices.append(i)
            miss_texts.append(text)

    if not miss_texts:
        return results

    # Phase 2: batch call Triton for all misses
    embeddings = await infer_embedding(miss_texts)

    if embeddings is None:
        # Triton failed — all misses remain None
        logger.warning(
            "Triton batch embedding failed for %d texts", len(miss_texts)
        )
        return results

    # Phase 3: merge results and update cache
    _evict_oldest(need_space=len(embeddings))
    now = time.monotonic()

    for idx, vec in zip(miss_indices, embeddings):
        results[idx] = vec
        _cache[texts[idx]] = (vec, now)

    return results


def clear_cache() -> None:
    """Remove all entries from the embedding cache."""
    _cache.clear()


def get_cache_stats() -> dict:
    """Return cache statistics.

    Keys:
        size: current number of entries
        max_size: configured maximum
        ttl_seconds: configured TTL
        valid: number of entries that have not expired
        expired: number of entries past TTL but not yet evicted
    """
    now = time.monotonic()
    valid = sum(1 for _, (__, ts) in _cache.items() if (now - ts) < EMBED_CACHE_TTL)
    return {
        "size": len(_cache),
        "max_size": EMBED_CACHE_MAX,
        "ttl_seconds": EMBED_CACHE_TTL,
        "valid": valid,
        "expired": len(_cache) - valid,
    }
