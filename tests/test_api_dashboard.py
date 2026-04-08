"""Tests for forge.api.routes.dashboard -- dashboard, experts, expert detail."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_engine():
    """Create a mock graph engine with test data."""
    from forge.graph.models import Node, NodeType, Edge, EdgeType, EdgeSource, GraphStats

    engine = MagicMock()
    store = MagicMock()
    engine.store = store

    expert_a = Node(id="exp-1", name="Alice Expert", node_type=NodeType.expert)
    expert_b = Node(id="exp-2", name="Bob Expert", node_type=NodeType.expert)

    engine.get_all_experts.return_value = [expert_a, expert_b]
    engine.node_count.return_value = 10
    engine.edge_count.return_value = 5
    engine.expert_authority.return_value = []

    edge = Edge(
        id="edge-1", source_id="exp-1", target_id="exp-2",
        edge_type=EdgeType.agrees_with, weight=0.9, confidence=0.85,
        source=EdgeSource.auto_discovered,
    )
    store.get_edges_for_node.return_value = [edge]
    store.get_node.side_effect = lambda nid: expert_a if nid == "exp-1" else expert_b
    store.get_stats.return_value = GraphStats(
        total_nodes=10, total_edges=5, active_edges=4, expired_edges=1,
        node_type_counts={"expert": 2, "concept": 8},
        edge_type_counts={"agrees_with": 3, "contradicts": 2},
    )

    return engine


@pytest.fixture
def client(mock_engine):
    with patch("forge.api.main.GraphStore"), \
         patch("forge.api.main.GraphEngine", return_value=mock_engine), \
         patch("forge.api.main.bootstrap_graph", return_value={"skipped": True}), \
         patch("forge.api.main.GuardrailsEngine"), \
         patch("forge.api.main.start_scheduler"), \
         patch("forge.api.main.stop_scheduler"):
        mock_engine.load = MagicMock()
        from forge.api.main import app
        import forge.api.main as main_mod
        with TestClient(app) as c:
            main_mod.graph_engine = mock_engine
            yield c


class TestDashboardEndpoint:
    def test_dashboard_returns_200(self, client):
        with patch("forge.core.qdrant_client.get_total_chunks", return_value=100), \
             patch("forge.core.qdrant_client.get_status", return_value="green"), \
             patch("forge.workers.scheduler.is_running", return_value=True), \
             patch("forge.workers.scheduler.get_jobs", return_value=[]):
            resp = client.get("/api/dashboard")
        assert resp.status_code == 200

    def test_dashboard_has_graph_stats(self, client):
        with patch("forge.core.qdrant_client.get_total_chunks", return_value=100), \
             patch("forge.core.qdrant_client.get_status", return_value="green"), \
             patch("forge.workers.scheduler.is_running", return_value=False), \
             patch("forge.workers.scheduler.get_jobs", return_value=[]):
            data = client.get("/api/dashboard").json()
        assert "graph" in data
        assert data["graph"]["total_nodes"] == 10
        assert data["graph"]["total_edges"] == 5

    def test_dashboard_has_knowledge_base(self, client):
        with patch("forge.core.qdrant_client.get_total_chunks", return_value=42), \
             patch("forge.core.qdrant_client.get_status", return_value="green"), \
             patch("forge.workers.scheduler.is_running", return_value=False), \
             patch("forge.workers.scheduler.get_jobs", return_value=[]):
            data = client.get("/api/dashboard").json()
        assert data["knowledge_base"]["total_chunks"] == 42
        assert data["knowledge_base"]["qdrant_status"] == "green"

    def test_dashboard_has_services(self, client):
        with patch("forge.core.qdrant_client.get_total_chunks", return_value=0), \
             patch("forge.core.qdrant_client.get_status", return_value="red"), \
             patch("forge.workers.scheduler.is_running", return_value=False), \
             patch("forge.workers.scheduler.get_jobs", return_value=[]):
            data = client.get("/api/dashboard").json()
        assert "services" in data

    def test_dashboard_qdrant_error(self, client):
        with patch("forge.core.qdrant_client.get_total_chunks", side_effect=Exception), \
             patch("forge.core.qdrant_client.get_status", side_effect=Exception), \
             patch("forge.workers.scheduler.is_running", return_value=False), \
             patch("forge.workers.scheduler.get_jobs", return_value=[]):
            data = client.get("/api/dashboard").json()
        assert data["knowledge_base"]["total_chunks"] == 0


class TestExpertsEndpoint:
    def test_list_experts(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=10):
            data = client.get("/api/experts").json()
        assert len(data["experts"]) == 2

    def test_expert_has_slug(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=5):
            data = client.get("/api/experts").json()
        assert data["experts"][0]["slug"] == "alice-expert"

    def test_expert_has_chunk_count(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=15):
            data = client.get("/api/experts").json()
        assert data["experts"][0]["chunk_count"] == 15

    def test_expert_has_edge_count(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=0):
            data = client.get("/api/experts").json()
        assert data["experts"][0]["edge_count"] == 1

    def test_experts_empty_engine(self, client, mock_engine):
        mock_engine.get_all_experts.return_value = []
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=0):
            data = client.get("/api/experts").json()
        assert data["experts"] == []


class TestExpertDetailEndpoint:
    def test_expert_detail_found(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=10):
            resp = client.get("/api/expert/alice-expert")
        assert resp.status_code == 200

    def test_expert_detail_has_connections(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=5):
            data = client.get("/api/expert/alice-expert").json()
        assert "connections" in data
        assert len(data["connections"]) == 1

    def test_expert_detail_not_found(self, client):
        resp = client.get("/api/expert/nobody")
        assert resp.status_code == 404

    def test_expert_detail_has_name(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=0):
            data = client.get("/api/expert/alice-expert").json()
        assert data["name"] == "Alice Expert"

    def test_expert_detail_has_rankings(self, client):
        with patch("forge.core.qdrant_client.count_chunks_for_expert", return_value=0):
            data = client.get("/api/expert/alice-expert").json()
        assert "rankings" in data
