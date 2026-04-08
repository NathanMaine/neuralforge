"""Comprehensive tests for forge.ingest.upserter — batch ingestion.

All Qdrant and embedding calls are mocked.
"""
import pytest
from unittest.mock import AsyncMock, patch

from forge.ingest.upserter import ingest_chunks


class TestIngestChunks:
    """Tests for ingest_chunks."""

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=3)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_basic_ingest(self, mock_embed, mock_upsert):
        """Chunks are batch-embedded and upserted."""
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        count = await ingest_chunks(
            chunks=["a", "b", "c"],
            creator="alice",
            title="Doc A",
            source="test.pdf",
        )

        assert count == 3
        mock_embed.assert_called_once_with(["a", "b", "c"])
        mock_upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        """Empty chunk list returns 0 immediately."""
        count = await ingest_chunks(
            chunks=[], creator="alice", title="", source=""
        )
        assert count == 0

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=2)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_skips_none_embeddings(self, mock_embed, mock_upsert):
        """Chunks with None embeddings are skipped."""
        mock_embed.return_value = [[0.1] * 384, None, [0.3] * 384]

        count = await ingest_chunks(
            chunks=["a", "b", "c"],
            creator="alice",
            title="Doc A",
            source="test.pdf",
        )

        assert count == 2
        points = mock_upsert.call_args[0][0]
        assert len(points) == 2

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_all_embeddings_none(self, mock_embed):
        """All embeddings None returns 0."""
        mock_embed.return_value = [None, None, None]

        count = await ingest_chunks(
            chunks=["a", "b", "c"],
            creator="alice",
            title="Doc A",
            source="test.pdf",
        )

        assert count == 0

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=1)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_payload_fields(self, mock_embed, mock_upsert):
        """Upserted points have correct payload fields."""
        mock_embed.return_value = [[0.1] * 384]

        await ingest_chunks(
            chunks=["hello"],
            creator="bob",
            title="Test Doc",
            source="test.txt",
            source_type="blog",
        )

        points = mock_upsert.call_args[0][0]
        payload = points[0]["payload"]
        assert payload["text"] == "hello"
        assert payload["creator"] == "bob"
        assert payload["title"] == "Test Doc"
        assert payload["source"] == "test.txt"
        assert payload["source_type"] == "blog"
        assert payload["chunk_index"] == 0
        assert "ingested_at" in payload

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=2)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_chunk_indices(self, mock_embed, mock_upsert):
        """Chunk indices are sequential."""
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        await ingest_chunks(
            chunks=["a", "b"],
            creator="alice",
            title="",
            source="",
        )

        points = mock_upsert.call_args[0][0]
        indices = [p["payload"]["chunk_index"] for p in points]
        assert indices == [0, 1]

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=1)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_default_source_type(self, mock_embed, mock_upsert):
        """Default source_type is 'document'."""
        mock_embed.return_value = [[0.1] * 384]

        await ingest_chunks(
            chunks=["a"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        assert points[0]["payload"]["source_type"] == "document"

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=1)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_uuid_format(self, mock_embed, mock_upsert):
        """Point IDs are UUID format."""
        mock_embed.return_value = [[0.1] * 384]

        await ingest_chunks(
            chunks=["a"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        point_id = points[0]["id"]
        assert len(point_id) == 36
        assert point_id.count("-") == 4

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=5)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_many_chunks(self, mock_embed, mock_upsert):
        """Multiple chunks all get embedded and upserted."""
        n = 5
        mock_embed.return_value = [[float(i)] * 384 for i in range(n)]

        count = await ingest_chunks(
            chunks=[f"chunk {i}" for i in range(n)],
            creator="alice",
            title="Big Doc",
            source="big.pdf",
        )

        assert count == 5
        points = mock_upsert.call_args[0][0]
        assert len(points) == 5

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=3)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_skipped_indices_preserved(self, mock_embed, mock_upsert):
        """Chunk indices reflect original position even when some are skipped."""
        mock_embed.return_value = [[0.1] * 384, None, [0.3] * 384, [0.4] * 384]

        await ingest_chunks(
            chunks=["a", "b", "c", "d"],
            creator="alice",
            title="",
            source="",
        )

        points = mock_upsert.call_args[0][0]
        indices = [p["payload"]["chunk_index"] for p in points]
        assert indices == [0, 2, 3]  # index 1 was skipped

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=1)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_vectors_attached(self, mock_embed, mock_upsert):
        """Each point has a vector from the embedding."""
        vec = [0.42] * 384
        mock_embed.return_value = [vec]

        await ingest_chunks(
            chunks=["test"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        assert points[0]["vector"] == vec

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=0)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_upsert_returns_partial(self, mock_embed, mock_upsert):
        """Returns whatever count upsert_points returns."""
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]
        mock_upsert.return_value = 1  # partial success

        count = await ingest_chunks(
            chunks=["a", "b"], creator="alice", title="", source=""
        )

        assert count == 1

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=1)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_single_chunk(self, mock_embed, mock_upsert):
        """Single chunk ingest works."""
        mock_embed.return_value = [[0.5] * 384]

        count = await ingest_chunks(
            chunks=["only one"],
            creator="alice",
            title="Single",
            source="single.txt",
        )

        assert count == 1

    @pytest.mark.asyncio
    @patch("forge.ingest.upserter.qdrant_client.upsert_points", return_value=1)
    @patch("forge.ingest.upserter.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_timestamps_consistent(self, mock_embed, mock_upsert):
        """All points in a batch have the same ingested_at timestamp."""
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        await ingest_chunks(
            chunks=["a", "b"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        ts0 = points[0]["payload"]["ingested_at"]
        ts1 = points[1]["payload"]["ingested_at"]
        assert ts0 == ts1
