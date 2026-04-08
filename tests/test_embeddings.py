"""Tests for forge.core.embeddings — 25 tests covering cache, batch, TTL, eviction, and stats."""
import time
from unittest.mock import AsyncMock, patch

import pytest

from forge.core import embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure a clean cache for every test."""
    embeddings.clear_cache()
    yield
    embeddings.clear_cache()


def _vec(seed: float, dim: int = 3) -> list[float]:
    """Produce a deterministic fake embedding vector."""
    return [seed + i * 0.01 for i in range(dim)]


# ---------------------------------------------------------------------------
# Single embedding
# ---------------------------------------------------------------------------

class TestGetEmbedding:
    """Tests for get_embedding()."""

    @pytest.mark.asyncio
    async def test_single_embed_success(self):
        vec = _vec(1.0)
        with patch.object(embeddings, "infer_embedding", new_callable=AsyncMock, return_value=[vec]):
            result = await embeddings.get_embedding("hello")
        assert result == vec

    @pytest.mark.asyncio
    async def test_single_embed_caches_result(self):
        vec = _vec(2.0)
        mock = AsyncMock(return_value=[vec])
        with patch.object(embeddings, "infer_embedding", mock):
            await embeddings.get_embedding("hello")
            await embeddings.get_embedding("hello")
        # Triton should be called only once
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_single_embed_triton_failure(self):
        with patch.object(embeddings, "infer_embedding", new_callable=AsyncMock, return_value=None):
            result = await embeddings.get_embedding("fail")
        assert result is None

    @pytest.mark.asyncio
    async def test_single_embed_empty_result(self):
        with patch.object(embeddings, "infer_embedding", new_callable=AsyncMock, return_value=[]):
            result = await embeddings.get_embedding("empty")
        assert result is None

    @pytest.mark.asyncio
    async def test_different_texts_get_different_cache_entries(self):
        mock = AsyncMock(side_effect=[[_vec(1.0)], [_vec(2.0)]])
        with patch.object(embeddings, "infer_embedding", mock):
            r1 = await embeddings.get_embedding("hello")
            r2 = await embeddings.get_embedding("world")
        assert r1 != r2
        assert mock.call_count == 2


# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

class TestCacheTTL:
    """Tests for TTL-based expiry."""

    @pytest.mark.asyncio
    async def test_expired_entry_triggers_refresh(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 1)  # 1 second
        vec1 = _vec(1.0)
        vec2 = _vec(2.0)
        mock = AsyncMock(side_effect=[[vec1], [vec2]])
        with patch.object(embeddings, "infer_embedding", mock):
            r1 = await embeddings.get_embedding("text")
            assert r1 == vec1

            # Manually expire the entry
            embeddings._cache["text"] = (vec1, time.monotonic() - 10)

            r2 = await embeddings.get_embedding("text")
            assert r2 == vec2
        assert mock.call_count == 2

    def test_is_valid_fresh_entry(self):
        entry = (_vec(1.0), time.monotonic())
        assert embeddings._is_valid(entry) is True

    def test_is_valid_expired_entry(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 60)
        entry = (_vec(1.0), time.monotonic() - 120)
        assert embeddings._is_valid(entry) is False

    def test_evict_expired_removes_old_entries(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 5)
        now = time.monotonic()
        embeddings._cache["old"] = (_vec(1.0), now - 100)
        embeddings._cache["fresh"] = (_vec(2.0), now)
        embeddings._evict_expired()
        assert "old" not in embeddings._cache
        assert "fresh" in embeddings._cache


# ---------------------------------------------------------------------------
# Cache eviction (max size)
# ---------------------------------------------------------------------------

class TestCacheEviction:
    """Tests for max-size eviction."""

    def test_evict_oldest_makes_room(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_MAX", 3)
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 9999)
        now = time.monotonic()
        embeddings._cache["a"] = (_vec(1.0), now - 30)
        embeddings._cache["b"] = (_vec(2.0), now - 20)
        embeddings._cache["c"] = (_vec(3.0), now - 10)
        # Need room for 1 more -> should evict "a" (oldest)
        embeddings._evict_oldest(need_space=1)
        assert "a" not in embeddings._cache
        assert "b" in embeddings._cache
        assert "c" in embeddings._cache

    def test_evict_oldest_no_eviction_needed(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_MAX", 10)
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 9999)
        embeddings._cache["x"] = (_vec(1.0), time.monotonic())
        embeddings._evict_oldest(need_space=1)
        assert "x" in embeddings._cache

    @pytest.mark.asyncio
    async def test_cache_respects_max_size(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_MAX", 2)
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 9999)

        call_count = 0

        async def fake_infer(texts):
            nonlocal call_count
            call_count += 1
            return [_vec(call_count)]

        with patch.object(embeddings, "infer_embedding", side_effect=fake_infer):
            await embeddings.get_embedding("a")
            await embeddings.get_embedding("b")
            await embeddings.get_embedding("c")  # should evict "a"

        assert len(embeddings._cache) <= 2
        assert "a" not in embeddings._cache

    def test_evict_oldest_prefers_expired(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_MAX", 2)
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 5)
        now = time.monotonic()
        embeddings._cache["expired"] = (_vec(1.0), now - 100)
        embeddings._cache["fresh"] = (_vec(2.0), now)
        embeddings._evict_oldest(need_space=1)
        assert "expired" not in embeddings._cache
        assert "fresh" in embeddings._cache


# ---------------------------------------------------------------------------
# Batch embeddings
# ---------------------------------------------------------------------------

class TestGetEmbeddingsBatch:
    """Tests for get_embeddings_batch()."""

    @pytest.mark.asyncio
    async def test_empty_input(self):
        result = await embeddings.get_embeddings_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_all_cache_hits(self):
        now = time.monotonic()
        embeddings._cache["a"] = (_vec(1.0), now)
        embeddings._cache["b"] = (_vec(2.0), now)

        mock = AsyncMock()
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(["a", "b"])
        assert result == [_vec(1.0), _vec(2.0)]
        mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_cache_misses(self):
        vecs = [_vec(1.0), _vec(2.0)]
        mock = AsyncMock(return_value=vecs)
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(["a", "b"])
        assert result == vecs
        mock.assert_called_once_with(["a", "b"])

    @pytest.mark.asyncio
    async def test_partial_cache_hits(self):
        now = time.monotonic()
        embeddings._cache["b"] = (_vec(2.0), now)

        mock = AsyncMock(return_value=[_vec(1.0), _vec(3.0)])
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(["a", "b", "c"])
        assert result[0] == _vec(1.0)
        assert result[1] == _vec(2.0)
        assert result[2] == _vec(3.0)
        # Only misses sent to Triton
        mock.assert_called_once_with(["a", "c"])

    @pytest.mark.asyncio
    async def test_batch_populates_cache(self):
        vecs = [_vec(1.0), _vec(2.0)]
        mock = AsyncMock(return_value=vecs)
        with patch.object(embeddings, "infer_embedding", mock):
            await embeddings.get_embeddings_batch(["x", "y"])
        assert "x" in embeddings._cache
        assert "y" in embeddings._cache

    @pytest.mark.asyncio
    async def test_batch_triton_failure_returns_nones(self):
        mock = AsyncMock(return_value=None)
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(["a", "b"])
        assert result == [None, None]

    @pytest.mark.asyncio
    async def test_batch_partial_cache_with_failure(self):
        """Cached entries still returned even if Triton fails on misses."""
        now = time.monotonic()
        embeddings._cache["a"] = (_vec(1.0), now)
        mock = AsyncMock(return_value=None)
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(["a", "b"])
        assert result[0] == _vec(1.0)
        assert result[1] is None

    @pytest.mark.asyncio
    async def test_batch_preserves_order(self):
        vecs = [_vec(float(i)) for i in range(5)]
        mock = AsyncMock(return_value=vecs)
        texts = ["t0", "t1", "t2", "t3", "t4"]
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(texts)
        for i in range(5):
            assert result[i] == _vec(float(i))

    @pytest.mark.asyncio
    async def test_batch_skips_expired_cache(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 1)
        embeddings._cache["a"] = (_vec(1.0), time.monotonic() - 100)
        new_vec = _vec(9.0)
        mock = AsyncMock(return_value=[new_vec])
        with patch.object(embeddings, "infer_embedding", mock):
            result = await embeddings.get_embeddings_batch(["a"])
        assert result[0] == new_vec


# ---------------------------------------------------------------------------
# clear_cache / get_cache_stats
# ---------------------------------------------------------------------------

class TestCacheManagement:
    """Tests for clear_cache() and get_cache_stats()."""

    def test_clear_cache(self):
        embeddings._cache["x"] = (_vec(1.0), time.monotonic())
        embeddings.clear_cache()
        assert len(embeddings._cache) == 0

    def test_stats_empty_cache(self):
        stats = embeddings.get_cache_stats()
        assert stats["size"] == 0
        assert stats["valid"] == 0
        assert stats["expired"] == 0

    def test_stats_with_entries(self):
        now = time.monotonic()
        embeddings._cache["a"] = (_vec(1.0), now)
        embeddings._cache["b"] = (_vec(2.0), now)
        stats = embeddings.get_cache_stats()
        assert stats["size"] == 2
        assert stats["valid"] == 2
        assert stats["expired"] == 0

    def test_stats_with_expired(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 10)
        now = time.monotonic()
        embeddings._cache["fresh"] = (_vec(1.0), now)
        embeddings._cache["stale"] = (_vec(2.0), now - 100)
        stats = embeddings.get_cache_stats()
        assert stats["size"] == 2
        assert stats["valid"] == 1
        assert stats["expired"] == 1

    def test_stats_max_size_from_config(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_MAX", 42)
        stats = embeddings.get_cache_stats()
        assert stats["max_size"] == 42

    def test_stats_ttl_from_config(self, monkeypatch):
        monkeypatch.setattr(embeddings, "EMBED_CACHE_TTL", 600)
        stats = embeddings.get_cache_stats()
        assert stats["ttl_seconds"] == 600
