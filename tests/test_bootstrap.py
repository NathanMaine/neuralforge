"""Tests for forge.graph.bootstrap — graph bootstrapping on first startup.

All Qdrant calls are mocked so no live services are needed.
"""
import pytest
from unittest.mock import patch

from forge.graph.bootstrap import bootstrap_graph, DEFAULT_CONCEPTS
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


# ---------------------------------------------------------------------------
# bootstrap_graph
# ---------------------------------------------------------------------------


class TestBootstrapGraph:
    """Tests for bootstrap_graph."""

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_creates_expert_nodes(self, mock_names, engine):
        """Expert nodes are created from Qdrant creator names."""
        mock_names.return_value = ["alice", "bob", "charlie"]

        result = bootstrap_graph(engine)

        assert result["experts_created"] == 3
        assert result["expert_names"] == ["alice", "bob", "charlie"]
        assert engine.has_expert("alice")
        assert engine.has_expert("bob")
        assert engine.has_expert("charlie")

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_creates_concept_nodes(self, mock_names, engine):
        """20 default concept nodes are created."""
        mock_names.return_value = []

        result = bootstrap_graph(engine)

        assert result["concepts_created"] == 20
        # Verify a few known concepts
        assert engine.get_concept("machine learning") is not None
        assert engine.get_concept("cybersecurity") is not None
        assert engine.get_concept("data privacy") is not None

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_skips_non_empty_graph(self, mock_names, engine):
        """Bootstrap is skipped if graph already has nodes."""
        engine.add_expert("existing")
        mock_names.return_value = ["alice"]

        result = bootstrap_graph(engine)

        assert result["skipped"] is True
        assert result["experts_created"] == 0
        assert result["concepts_created"] == 0

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_empty_qdrant_still_creates_concepts(self, mock_names, engine):
        """Even with no Qdrant data, concepts are seeded."""
        mock_names.return_value = []

        result = bootstrap_graph(engine)

        assert result["experts_created"] == 0
        assert result["concepts_created"] == 20
        assert not result["skipped"]

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_returns_summary_dict(self, mock_names, engine):
        """Result dict has all expected keys."""
        mock_names.return_value = ["alice"]

        result = bootstrap_graph(engine)

        assert "experts_created" in result
        assert "concepts_created" in result
        assert "skipped" in result
        assert "expert_names" in result

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_default_concepts_count(self, mock_names, engine):
        """DEFAULT_CONCEPTS list has exactly 20 entries."""
        assert len(DEFAULT_CONCEPTS) == 20

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_node_count_after_bootstrap(self, mock_names, engine):
        """Total nodes = experts + concepts."""
        mock_names.return_value = ["alice", "bob"]

        bootstrap_graph(engine)

        assert engine.node_count() == 22  # 2 experts + 20 concepts

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_idempotent_second_call(self, mock_names, engine):
        """Second call is a no-op because graph is no longer empty."""
        mock_names.return_value = ["alice"]

        result1 = bootstrap_graph(engine)
        assert result1["skipped"] is False

        result2 = bootstrap_graph(engine)
        assert result2["skipped"] is True
        assert engine.node_count() == 21  # still same count

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_expert_names_list_matches(self, mock_names, engine):
        """expert_names in result matches what was created."""
        mock_names.return_value = ["zara", "mike", "alice"]

        result = bootstrap_graph(engine)

        assert sorted(result["expert_names"]) == ["alice", "mike", "zara"]

    @patch("forge.graph.bootstrap.qdrant_client.get_all_expert_names")
    def test_all_concepts_created(self, mock_names, engine):
        """Every concept in DEFAULT_CONCEPTS gets a node."""
        mock_names.return_value = []

        bootstrap_graph(engine)

        for concept_name in DEFAULT_CONCEPTS:
            assert engine.get_concept(concept_name) is not None, (
                f"Missing concept: {concept_name}"
            )
