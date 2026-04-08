"""Tests for forge.api.routes.graph -- full CRUD + cuGraph queries."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from forge.graph.models import (
    Node, Edge, NodeType, EdgeType, EdgeSource,
    GraphStats, TraversalResult, Contradiction, ExpertRanking,
)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    store = MagicMock()
    engine.store = store
    engine.load = MagicMock()
    engine.node_count.return_value = 5
    engine.edge_count.return_value = 3

    # Default empty stats
    store.get_stats.return_value = GraphStats(
        total_nodes=5, total_edges=3, active_edges=3, expired_edges=0,
        node_type_counts={"expert": 2, "concept": 3},
        edge_type_counts={"agrees_with": 2, "related_to": 1},
    )
    store.nodes_df = MagicMock()
    store.nodes_df.empty = True
    store.nodes_df.head.return_value = MagicMock(iterrows=lambda: iter([]))

    return engine


@pytest.fixture
def client(mock_engine):
    with patch("forge.api.main.GraphStore"), \
         patch("forge.api.main.GraphEngine", return_value=mock_engine), \
         patch("forge.api.main.bootstrap_graph", return_value={"skipped": True}), \
         patch("forge.api.main.GuardrailsEngine"), \
         patch("forge.api.main.start_scheduler"), \
         patch("forge.api.main.stop_scheduler"):
        from forge.api.main import app
        import forge.api.main as m
        with TestClient(app) as c:
            m.graph_engine = mock_engine
            yield c


class TestNodeCRUD:
    def test_create_node(self, client, mock_engine):
        node = Node(id="n1", name="Test", node_type=NodeType.concept)
        mock_engine.add_node.return_value = node
        resp = client.post("/api/graph/nodes", json={
            "name": "Test", "node_type": "concept",
        })
        assert resp.status_code == 201

    def test_create_node_invalid_type(self, client):
        resp = client.post("/api/graph/nodes", json={
            "name": "Test", "node_type": "invalid_type",
        })
        assert resp.status_code == 400

    def test_get_node(self, client, mock_engine):
        node = Node(id="n1", name="Test", node_type=NodeType.concept)
        mock_engine.store.get_node.return_value = node
        resp = client.get("/api/graph/nodes/n1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test"

    def test_get_node_not_found(self, client, mock_engine):
        mock_engine.store.get_node.return_value = None
        resp = client.get("/api/graph/nodes/missing")
        assert resp.status_code == 404

    def test_list_nodes_empty(self, client, mock_engine):
        resp = client.get("/api/graph/nodes")
        assert resp.status_code == 200

    def test_list_nodes_with_type_filter(self, client, mock_engine):
        mock_engine.store.get_nodes_by_type.return_value = [
            Node(id="n1", name="Test", node_type=NodeType.expert)
        ]
        resp = client.get("/api/graph/nodes?node_type=expert")
        assert resp.status_code == 200

    def test_list_nodes_with_search(self, client, mock_engine):
        mock_engine.store.search_nodes.return_value = [
            Node(id="n1", name="ML Expert", node_type=NodeType.expert)
        ]
        resp = client.get("/api/graph/nodes?q=ML")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1

    def test_update_node(self, client, mock_engine):
        import pandas as pd
        # Use a real DataFrame to avoid mask issues
        mock_engine.store.nodes_df = pd.DataFrame([{
            "id": "n1", "name": "Test", "node_type": "concept",
            "description": "old", "metadata": "{}", "created_at": "2024-01-01",
        }])
        node = Node(id="n1", name="Test", node_type=NodeType.concept, description="updated")
        mock_engine.store.get_node.return_value = node
        resp = client.patch("/api/graph/nodes/n1", json={"description": "updated"})
        assert resp.status_code == 200

    def test_update_node_not_found(self, client, mock_engine):
        import pandas as pd
        mock_engine.store.nodes_df = pd.DataFrame(columns=["id", "name", "node_type", "description", "metadata", "created_at"])
        resp = client.patch("/api/graph/nodes/missing", json={"description": "x"})
        assert resp.status_code == 404


class TestEdgeCRUD:
    def test_create_edge(self, client, mock_engine):
        node = Node(id="n1", name="A", node_type=NodeType.concept)
        mock_engine.store.get_node.return_value = node
        edge = Edge(id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.related_to)
        mock_engine.add_edge.return_value = edge
        resp = client.post("/api/graph/edges", json={
            "source_id": "n1", "target_id": "n2", "edge_type": "related_to",
        })
        assert resp.status_code == 201

    def test_create_edge_invalid_type(self, client, mock_engine):
        resp = client.post("/api/graph/edges", json={
            "source_id": "n1", "target_id": "n2", "edge_type": "invalid",
        })
        assert resp.status_code == 400

    def test_create_edge_missing_source(self, client, mock_engine):
        mock_engine.store.get_node.return_value = None
        resp = client.post("/api/graph/edges", json={
            "source_id": "missing", "target_id": "n2", "edge_type": "related_to",
        })
        assert resp.status_code == 404

    def test_update_edge(self, client, mock_engine):
        edge = Edge(id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.related_to, weight=0.8)
        mock_engine.store.update_edge.return_value = edge
        resp = client.patch("/api/graph/edges/e1", json={"weight": 0.8})
        assert resp.status_code == 200

    def test_update_edge_not_found(self, client, mock_engine):
        mock_engine.store.update_edge.return_value = None
        resp = client.patch("/api/graph/edges/missing", json={"weight": 0.5})
        assert resp.status_code == 404

    def test_update_edge_no_fields(self, client):
        resp = client.patch("/api/graph/edges/e1", json={})
        assert resp.status_code == 400

    def test_delete_edge(self, client, mock_engine):
        edge = Edge(id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.related_to)
        mock_engine.store.get_edge.return_value = edge
        resp = client.delete("/api/graph/edges/e1")
        assert resp.status_code == 204

    def test_delete_edge_not_found(self, client, mock_engine):
        mock_engine.store.get_edge.return_value = None
        resp = client.delete("/api/graph/edges/missing")
        assert resp.status_code == 404


class TestGraphQueries:
    def test_traverse(self, client, mock_engine):
        result = TraversalResult(root_id="n1", nodes=[], edges=[], depth=2)
        mock_engine.traverse.return_value = result
        resp = client.get("/api/graph/traverse?node_id=n1")
        assert resp.status_code == 200
        assert resp.json()["root_id"] == "n1"

    def test_traverse_with_edge_types(self, client, mock_engine):
        result = TraversalResult(root_id="n1")
        mock_engine.traverse.return_value = result
        resp = client.get("/api/graph/traverse?node_id=n1&edge_types=agrees_with,contradicts")
        assert resp.status_code == 200

    def test_contradictions(self, client, mock_engine):
        mock_engine.find_contradictions.return_value = []
        resp = client.get("/api/graph/contradictions")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_contradictions_with_topic(self, client, mock_engine):
        mock_engine.find_contradictions.return_value = []
        resp = client.get("/api/graph/contradictions?topic=ML")
        assert resp.status_code == 200

    def test_experts_for_topic(self, client, mock_engine):
        ranking = ExpertRanking(expert_id="e1", expert_name="Alice", topic="ML", score=0.9, edge_count=5)
        mock_engine.expert_authority.return_value = [ranking]
        resp = client.get("/api/graph/experts-for?topic=ML")
        assert resp.status_code == 200
        assert len(resp.json()["rankings"]) == 1

    def test_rankings(self, client, mock_engine):
        mock_engine.expert_authority.return_value = []
        resp = client.get("/api/graph/rankings?topic=NLP")
        assert resp.status_code == 200


class TestCuGraphQueries:
    def test_pagerank(self, client, mock_engine):
        mock_engine.pagerank.return_value = {"n1": 0.5, "n2": 0.3}
        mock_engine.store.get_node.return_value = Node(id="n1", name="A", node_type=NodeType.concept)
        resp = client.get("/api/graph/pagerank")
        assert resp.status_code == 200
        assert "pagerank" in resp.json()

    def test_communities(self, client, mock_engine):
        mock_engine.find_communities.return_value = {"n1": 0, "n2": 0, "n3": 1}
        mock_engine.store.get_node.return_value = Node(id="n1", name="A", node_type=NodeType.concept)
        resp = client.get("/api/graph/communities")
        assert resp.status_code == 200
        assert resp.json()["num_communities"] == 2

    def test_shortest_path_found(self, client, mock_engine):
        mock_engine.shortest_path.return_value = ["n1", "n2", "n3"]
        mock_engine.store.get_node.return_value = Node(id="n1", name="A", node_type=NodeType.concept)
        resp = client.get("/api/graph/shortest-path?source=n1&target=n3")
        assert resp.status_code == 200
        assert resp.json()["found"] is True
        assert resp.json()["length"] == 2

    def test_shortest_path_not_found(self, client, mock_engine):
        mock_engine.shortest_path.return_value = []
        resp = client.get("/api/graph/shortest-path?source=n1&target=n9")
        assert resp.status_code == 200
        assert resp.json()["found"] is False


class TestTimelineAndStats:
    def test_timeline(self, client, mock_engine):
        mock_engine.find_changes_since.return_value = []
        resp = client.get("/api/graph/timeline?since=2024-01-01")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_timeline_with_topic(self, client, mock_engine):
        mock_engine.find_changes_since.return_value = []
        resp = client.get("/api/graph/timeline?since=2024-01-01&topic=ML")
        assert resp.status_code == 200

    def test_stats(self, client, mock_engine):
        resp = client.get("/api/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] == 5
        assert data["total_edges"] == 3
