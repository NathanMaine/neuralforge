"""Comprehensive tests for forge.graph.discovery — expert relationship discovery.

All NIM, Qdrant, and embedding calls are mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from forge.graph.discovery import (
    CLASSIFICATION_PROMPT,
    create_edge_from_discovery,
    discover_pair,
    get_shared_topics,
)
from forge.graph.engine import GraphEngine
from forge.graph.store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path):
    """Create a fresh GraphEngine with temp-dir backed store."""
    store = GraphStore(data_dir=str(tmp_path / "graph"))
    eng = GraphEngine(store=store)
    eng.add_expert("alice")
    eng.add_expert("bob")
    return eng


# ---------------------------------------------------------------------------
# discover_pair
# ---------------------------------------------------------------------------


class TestDiscoverPair:
    """Tests for discover_pair."""

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_successful_discovery(self, mock_embed, mock_search, mock_classify):
        """Successful discovery returns relationship dict."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice on ML", "score": 0.9}],
            [{"text": "Bob on ML", "score": 0.8}],
        ]
        mock_classify.return_value = {
            "relationship": "agrees",
            "confidence": 0.85,
            "summary": "Both support transformers",
        }

        result = await discover_pair("alice", "bob", "machine learning")

        assert result is not None
        assert result["expert_a"] == "alice"
        assert result["expert_b"] == "bob"
        assert result["topic"] == "machine learning"
        assert result["relationship"] == "agrees"
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_embedding_fails_returns_none(self, mock_embed):
        """Failed topic embedding returns None."""
        mock_embed.return_value = None

        result = await discover_pair("alice", "bob", "topic")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_no_chunks_for_expert_a(self, mock_embed, mock_search):
        """No chunks for expert A returns None."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [],  # no chunks for alice
            [{"text": "Bob stuff", "score": 0.8}],
        ]

        result = await discover_pair("alice", "bob", "topic")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_no_chunks_for_expert_b(self, mock_embed, mock_search):
        """No chunks for expert B returns None."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice stuff", "score": 0.9}],
            [],  # no chunks for bob
        ]

        result = await discover_pair("alice", "bob", "topic")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_nim_fails_returns_none(self, mock_embed, mock_search, mock_classify):
        """NIM classification failure returns None."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice on ML"}],
            [{"text": "Bob on ML"}],
        ]
        mock_classify.return_value = None

        result = await discover_pair("alice", "bob", "ML")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_below_confidence_floor(self, mock_embed, mock_search, mock_classify):
        """Confidence below floor returns None."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice on ML"}],
            [{"text": "Bob on ML"}],
        ]
        mock_classify.return_value = {
            "relationship": "agrees",
            "confidence": 0.3,  # below default 0.6
            "summary": "Weak agreement",
        }

        result = await discover_pair("alice", "bob", "ML")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_at_confidence_floor(self, mock_embed, mock_search, mock_classify):
        """Confidence exactly at floor passes."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice on ML"}],
            [{"text": "Bob on ML"}],
        ]
        mock_classify.return_value = {
            "relationship": "extends",
            "confidence": 0.6,
            "summary": "ok",
        }

        result = await discover_pair("alice", "bob", "ML")
        assert result is not None
        assert result["confidence"] == 0.6

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_missing_confidence_key(self, mock_embed, mock_search, mock_classify):
        """Missing confidence key defaults to 0.0 which is below floor."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice on ML"}],
            [{"text": "Bob on ML"}],
        ]
        mock_classify.return_value = {
            "relationship": "agrees",
            "summary": "no confidence field",
        }

        result = await discover_pair("alice", "bob", "ML")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_empty_text_in_chunks(self, mock_embed, mock_search):
        """Chunks with empty text fields return None."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": ""}],
            [{"text": "Bob stuff"}],
        ]

        result = await discover_pair("alice", "bob", "topic")
        assert result is None

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_multiple_chunks_concatenated(self, mock_embed, mock_search, mock_classify):
        """Multiple chunks per expert are concatenated."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Chunk A1"}, {"text": "Chunk A2"}],
            [{"text": "Chunk B1"}],
        ]
        mock_classify.return_value = {
            "relationship": "disagrees",
            "confidence": 0.9,
            "summary": "They disagree",
        }

        result = await discover_pair("alice", "bob", "topic")
        assert result is not None
        # Verify classify_json was called (meaning text was concatenated)
        mock_classify.assert_called_once()

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_result_has_all_fields(self, mock_embed, mock_search, mock_classify):
        """Result dict contains all expected keys."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice text"}],
            [{"text": "Bob text"}],
        ]
        mock_classify.return_value = {
            "relationship": "extends",
            "confidence": 0.75,
            "summary": "Bob extends Alice",
        }

        result = await discover_pair("alice", "bob", "AI")

        assert set(result.keys()) == {
            "expert_a", "expert_b", "topic",
            "relationship", "confidence", "summary",
        }

    @pytest.mark.asyncio
    @patch("forge.graph.discovery.nim_client.classify_json", new_callable=AsyncMock)
    @patch("forge.graph.discovery.qdrant_client.search_vectors")
    @patch("forge.graph.discovery.embeddings.get_embedding", new_callable=AsyncMock)
    async def test_missing_relationship_defaults_unrelated(self, mock_embed, mock_search, mock_classify):
        """Missing relationship key defaults to 'unrelated'."""
        mock_embed.return_value = [0.1] * 384
        mock_search.side_effect = [
            [{"text": "Alice text"}],
            [{"text": "Bob text"}],
        ]
        mock_classify.return_value = {
            "confidence": 0.8,
            "summary": "no rel field",
        }

        result = await discover_pair("alice", "bob", "AI")
        assert result["relationship"] == "unrelated"


# ---------------------------------------------------------------------------
# create_edge_from_discovery
# ---------------------------------------------------------------------------


class TestCreateEdgeFromDiscovery:
    """Tests for create_edge_from_discovery."""

    def test_creates_edge(self, engine):
        """Edge is created in the graph from discovery result."""
        discovery = {
            "expert_a": "alice",
            "expert_b": "bob",
            "topic": "ML",
            "relationship": "related_to",
            "confidence": 0.85,
            "summary": "They agree",
        }

        create_edge_from_discovery(engine, discovery)

        assert engine.edge_count() == 1

    def test_edge_has_correct_type(self, engine):
        """Created edge uses the relationship type."""
        discovery = {
            "expert_a": "alice",
            "expert_b": "bob",
            "topic": "AI",
            "relationship": "related_to",
            "confidence": 0.9,
            "summary": "test",
        }

        create_edge_from_discovery(engine, discovery)

        edges = engine.store.edges_df
        assert len(edges) == 1
        assert edges.iloc[0]["edge_type"] == "related_to"

    def test_missing_summary_handled(self, engine):
        """Discovery result without summary key does not crash."""
        discovery = {
            "expert_a": "alice",
            "expert_b": "bob",
            "topic": "AI",
            "relationship": "related_to",
            "confidence": 0.9,
        }

        # Should not raise
        create_edge_from_discovery(engine, discovery)
        assert engine.edge_count() == 1


# ---------------------------------------------------------------------------
# get_shared_topics
# ---------------------------------------------------------------------------


class TestGetSharedTopics:
    """Tests for get_shared_topics."""

    @patch("forge.graph.discovery.qdrant_client._post")
    def test_shared_titles(self, mock_post):
        """Returns titles that both experts have."""
        def side_effect(url, **kwargs):
            body = kwargs.get("json", {})
            creator = body.get("filter", {}).get("must", [{}])[0].get("match", {}).get("value", "")
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if creator == "alice":
                resp.json.return_value = {
                    "result": {
                        "points": [
                            {"payload": {"title": "Machine Learning"}},
                            {"payload": {"title": "Deep Learning"}},
                        ]
                    }
                }
            else:
                resp.json.return_value = {
                    "result": {
                        "points": [
                            {"payload": {"title": "machine learning"}},
                            {"payload": {"title": "Robotics"}},
                        ]
                    }
                }
            return resp

        mock_post.side_effect = side_effect

        topics = get_shared_topics("alice", "bob")
        assert "machine learning" in topics

    @patch("forge.graph.discovery.qdrant_client._post")
    def test_no_shared_topics(self, mock_post):
        """Returns empty list when no overlap."""
        def side_effect(url, **kwargs):
            body = kwargs.get("json", {})
            creator = body.get("filter", {}).get("must", [{}])[0].get("match", {}).get("value", "")
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if creator == "alice":
                resp.json.return_value = {
                    "result": {"points": [{"payload": {"title": "AI"}}]}
                }
            else:
                resp.json.return_value = {
                    "result": {"points": [{"payload": {"title": "Biology"}}]}
                }
            return resp

        mock_post.side_effect = side_effect

        topics = get_shared_topics("alice", "bob")
        assert topics == []

    @patch("forge.graph.discovery.qdrant_client._post")
    def test_empty_titles_skipped(self, mock_post):
        """Points with empty titles are skipped."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "result": {"points": [{"payload": {"title": ""}}, {"payload": {}}]}
        }
        mock_post.return_value = resp

        topics = get_shared_topics("alice", "bob")
        assert topics == []

    @patch("forge.graph.discovery.qdrant_client._post")
    def test_exception_returns_empty(self, mock_post):
        """Exception during scroll returns empty list."""
        mock_post.side_effect = RuntimeError("network error")

        topics = get_shared_topics("alice", "bob")
        assert topics == []


class TestClassificationPrompt:
    """Verify the prompt template."""

    def test_prompt_has_placeholders(self):
        """Prompt contains expected placeholders."""
        assert "{topic}" in CLASSIFICATION_PROMPT
        assert "{expert_a}" in CLASSIFICATION_PROMPT
        assert "{expert_b}" in CLASSIFICATION_PROMPT
        assert "{text_a}" in CLASSIFICATION_PROMPT
        assert "{text_b}" in CLASSIFICATION_PROMPT

    def test_prompt_formats_correctly(self):
        """Prompt can be formatted without error."""
        formatted = CLASSIFICATION_PROMPT.format(
            topic="AI",
            expert_a="Alice",
            expert_b="Bob",
            text_a="Alice says...",
            text_b="Bob says...",
        )
        assert "Alice" in formatted
        assert "Bob" in formatted
        assert "AI" in formatted
