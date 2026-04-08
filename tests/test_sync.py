"""Comprehensive tests for forge.core.sync — DataSync consistency layer.

All Qdrant and embedding calls are mocked so no live services are needed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from forge.core.sync import DataSync
from forge.graph.engine import GraphEngine
from forge.graph.store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path):
    """Create a fresh GraphEngine with a temp-dir backed store."""
    store = GraphStore(data_dir=str(tmp_path / "graph"))
    return GraphEngine(store=store)


@pytest.fixture
def sync(engine):
    """Create a DataSync wired to the test engine."""
    return DataSync(graph_engine=engine)


# ---------------------------------------------------------------------------
# ingest_and_sync
# ---------------------------------------------------------------------------


class TestIngestAndSync:
    """Tests for DataSync.ingest_and_sync."""

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=3)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_basic_ingest(self, mock_embed, mock_upsert, sync):
        """Chunks are embedded, upserted, and expert node created."""
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]

        result = await sync.ingest_and_sync(
            chunks=["a", "b", "c"],
            creator="alice",
            title="Doc A",
            source="test.pdf",
        )

        assert result["chunks_upserted"] == 3
        assert result["expert_created"] is True
        assert result["errors"] == []
        mock_embed.assert_called_once_with(["a", "b", "c"])
        mock_upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_zero(self, sync):
        """Empty chunk list short-circuits with no work done."""
        result = await sync.ingest_and_sync(
            chunks=[], creator="alice", title="", source=""
        )
        assert result["chunks_upserted"] == 0
        assert result["expert_created"] is False

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=2)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_partial_embedding_failures(self, mock_embed, mock_upsert, sync):
        """Chunks with None embeddings are skipped."""
        mock_embed.return_value = [[0.1] * 384, None, [0.3] * 384]

        result = await sync.ingest_and_sync(
            chunks=["a", "b", "c"],
            creator="alice",
            title="Doc A",
            source="test.pdf",
        )

        assert result["chunks_upserted"] == 2
        assert "chunk_1_embedding_none" in result["errors"]
        # Only 2 points should be sent to upsert
        points = mock_upsert.call_args[0][0]
        assert len(points) == 2

    @pytest.mark.asyncio
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_embedding_exception(self, mock_embed, sync):
        """Exception in embedding returns error, no upsert."""
        mock_embed.side_effect = RuntimeError("Triton down")

        result = await sync.ingest_and_sync(
            chunks=["a"], creator="alice", title="", source=""
        )

        assert result["chunks_upserted"] == 0
        assert len(result["errors"]) == 1
        assert "embedding_failed" in result["errors"][0]

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", side_effect=RuntimeError("Qdrant down"))
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_qdrant_upsert_exception(self, mock_embed, mock_upsert, sync):
        """Exception in Qdrant upsert returns error."""
        mock_embed.return_value = [[0.1] * 384]

        result = await sync.ingest_and_sync(
            chunks=["a"], creator="alice", title="", source=""
        )

        assert result["chunks_upserted"] == 0
        assert any("qdrant_upsert_failed" in e for e in result["errors"])

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=1)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_expert_not_recreated(self, mock_embed, mock_upsert, sync, engine):
        """If expert already exists, expert_created is False on second call."""
        mock_embed.return_value = [[0.1] * 384]

        # First ingest creates expert
        r1 = await sync.ingest_and_sync(
            chunks=["a"], creator="alice", title="", source=""
        )
        assert r1["expert_created"] is True

        # Second ingest finds existing expert
        r2 = await sync.ingest_and_sync(
            chunks=["b"], creator="alice", title="", source=""
        )
        assert r2["expert_created"] is False

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=1)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_payload_has_correct_fields(self, mock_embed, mock_upsert, sync):
        """Upserted points have all required payload fields."""
        mock_embed.return_value = [[0.1] * 384]

        await sync.ingest_and_sync(
            chunks=["hello"],
            creator="bob",
            title="Test",
            source="test.txt",
            source_type="blog",
        )

        points = mock_upsert.call_args[0][0]
        payload = points[0]["payload"]
        assert payload["text"] == "hello"
        assert payload["creator"] == "bob"
        assert payload["title"] == "Test"
        assert payload["source"] == "test.txt"
        assert payload["source_type"] == "blog"
        assert payload["chunk_index"] == 0
        assert "ingested_at" in payload

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=2)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_chunk_indices_sequential(self, mock_embed, mock_upsert, sync):
        """chunk_index values are sequential across chunks."""
        mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

        await sync.ingest_and_sync(
            chunks=["a", "b"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        indices = [p["payload"]["chunk_index"] for p in points]
        assert indices == [0, 1]

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=1)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_default_source_type(self, mock_embed, mock_upsert, sync):
        """Default source_type is 'document'."""
        mock_embed.return_value = [[0.1] * 384]

        await sync.ingest_and_sync(
            chunks=["a"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        assert points[0]["payload"]["source_type"] == "document"

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=0)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_all_embeddings_none(self, mock_embed, mock_upsert, sync):
        """All embeddings None means no upsert called, but no crash."""
        mock_embed.return_value = [None, None]

        result = await sync.ingest_and_sync(
            chunks=["a", "b"], creator="alice", title="", source=""
        )

        # No points to upsert -- upsert should not be called or called with []
        assert result["chunks_upserted"] == 0
        assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    @patch("forge.core.sync.qdrant_client.upsert_points", return_value=1)
    @patch("forge.core.sync.embeddings.get_embeddings_batch", new_callable=AsyncMock)
    async def test_point_ids_are_uuids(self, mock_embed, mock_upsert, sync):
        """Each point has a UUID-format ID."""
        mock_embed.return_value = [[0.1] * 384]

        await sync.ingest_and_sync(
            chunks=["a"], creator="alice", title="", source=""
        )

        points = mock_upsert.call_args[0][0]
        point_id = points[0]["id"]
        assert len(point_id) == 36  # UUID format
        assert point_id.count("-") == 4


# ---------------------------------------------------------------------------
# check_consistency
# ---------------------------------------------------------------------------


class TestCheckConsistency:
    """Tests for DataSync.check_consistency."""

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_fully_consistent(self, mock_names, sync, engine):
        """Both stores have the same creators -- consistent=True."""
        mock_names.return_value = ["alice", "bob"]
        engine.add_expert("alice")
        engine.add_expert("bob")

        result = sync.check_consistency()
        assert result["consistent"] is True
        assert result["qdrant_only"] == []
        assert result["graph_only"] == []

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_qdrant_only(self, mock_names, sync, engine):
        """Creator in Qdrant but not graph."""
        mock_names.return_value = ["alice", "bob"]
        engine.add_expert("alice")

        result = sync.check_consistency()
        assert result["consistent"] is False
        assert result["qdrant_only"] == ["bob"]
        assert result["graph_only"] == []

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_graph_only(self, mock_names, sync, engine):
        """Expert in graph but not Qdrant."""
        mock_names.return_value = ["alice"]
        engine.add_expert("alice")
        engine.add_expert("charlie")

        result = sync.check_consistency()
        assert result["consistent"] is False
        assert result["qdrant_only"] == []
        assert result["graph_only"] == ["charlie"]

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_both_empty(self, mock_names, sync):
        """Both stores empty -- consistent."""
        mock_names.return_value = []

        result = sync.check_consistency()
        assert result["consistent"] is True


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------


class TestRepair:
    """Tests for DataSync.repair."""

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_repairs_missing_graph_nodes(self, mock_names, sync, engine):
        """Creates graph nodes for Qdrant-only creators."""
        mock_names.return_value = ["alice", "bob"]
        engine.add_expert("alice")

        result = sync.repair()
        assert result["repaired"] == 1
        assert "bob" in result["names"]
        assert engine.has_expert("bob")

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_nothing_to_repair(self, mock_names, sync, engine):
        """No repair needed when consistent."""
        mock_names.return_value = ["alice"]
        engine.add_expert("alice")

        result = sync.repair()
        assert result["repaired"] == 0
        assert result["names"] == []

    @patch("forge.core.sync.qdrant_client.get_all_expert_names")
    def test_repair_multiple(self, mock_names, sync, engine):
        """Repairs multiple missing nodes."""
        mock_names.return_value = ["alice", "bob", "charlie"]

        result = sync.repair()
        assert result["repaired"] == 3
        assert sorted(result["names"]) == ["alice", "bob", "charlie"]
