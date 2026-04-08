"""Tests for forge.mcp -- server handler, transport, and all 17 tools."""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from forge.mcp.server import handle_request, handle_batch, PARSE_ERROR, METHOD_NOT_FOUND


# --- Server handler tests ---

class TestMCPServer:
    @pytest.mark.asyncio
    async def test_initialize(self):
        resp = await handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        })
        assert resp["result"]["serverInfo"]["name"] == "neuralforge-mcp"
        assert resp["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_initialized_notification(self):
        resp = await handle_request({
            "jsonrpc": "2.0", "id": 2, "method": "notifications/initialized",
        })
        assert resp["result"]["acknowledged"] is True

    @pytest.mark.asyncio
    async def test_tools_list(self):
        resp = await handle_request({
            "jsonrpc": "2.0", "id": 3, "method": "tools/list",
        })
        tools = resp["result"]["tools"]
        assert len(tools) >= 17
        tool_names = [t["name"] for t in tools]
        assert "search_experts" in tool_names
        assert "pagerank" in tool_names
        assert "auto_capture_status" in tool_names

    @pytest.mark.asyncio
    async def test_tools_call_missing_name(self):
        resp = await handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {},
        })
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_tools_call_unknown_tool(self):
        resp = await handle_request({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "nonexistent_tool"},
        })
        assert "error" in resp
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    @pytest.mark.asyncio
    async def test_unknown_method(self):
        resp = await handle_request({
            "jsonrpc": "2.0", "id": 6, "method": "unknown/method",
        })
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_missing_method(self):
        resp = await handle_request({"jsonrpc": "2.0", "id": 7})
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_batch_request(self):
        responses = await handle_batch([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ])
        assert len(responses) == 2
        assert responses[0]["result"]["serverInfo"]["name"] == "neuralforge-mcp"

    @pytest.mark.asyncio
    async def test_tool_call_success(self):
        with patch("forge.mcp.tools.auto_capture_status", new_callable=AsyncMock, return_value='{"enabled": false}'):
            resp = await handle_request({
                "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                "params": {"name": "auto_capture_status", "arguments": {}},
            })
        assert "result" in resp
        assert resp["result"]["isError"] is False

    @pytest.mark.asyncio
    async def test_tool_call_invalid_params(self):
        # search_experts requires 'query' but we provide nothing
        with patch("forge.mcp.tools.search_experts", new_callable=AsyncMock, side_effect=TypeError("missing")):
            resp = await handle_request({
                "jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {"name": "search_experts", "arguments": {}},
            })
        assert "error" in resp

    @pytest.mark.asyncio
    async def test_tool_call_runtime_error(self):
        from forge.mcp.tools import TOOL_DISPATCH
        original = TOOL_DISPATCH["stats"]
        async def boom_stats(**kwargs):
            raise RuntimeError("boom")
        TOOL_DISPATCH["stats"] = boom_stats
        try:
            resp = await handle_request({
                "jsonrpc": "2.0", "id": 10, "method": "tools/call",
                "params": {"name": "stats", "arguments": {}},
            })
            assert resp["result"]["isError"] is True
        finally:
            TOOL_DISPATCH["stats"] = original


# --- Transport tests ---

class TestMCPTransport:
    @pytest.fixture
    def client(self):
        mock_engine = MagicMock()
        mock_engine.load = MagicMock()
        mock_engine.node_count.return_value = 0
        mock_engine.edge_count.return_value = 0
        mock_engine.store = MagicMock()
        mock_engine.store.get_stats.return_value = MagicMock(
            total_nodes=0, total_edges=0, active_edges=0, expired_edges=0,
            node_type_counts={}, edge_type_counts={},
        )

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

    def test_single_request(self, client):
        resp = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
        })
        assert resp.status_code == 200
        assert resp.json()["result"]["serverInfo"]["name"] == "neuralforge-mcp"

    def test_batch_request(self, client):
        resp = client.post("/mcp", json=[
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ])
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_empty_batch(self, client):
        resp = client.post("/mcp", json=[])
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_invalid_json(self, client):
        resp = client.post("/mcp", content=b"not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 200

    def test_invalid_type(self, client):
        resp = client.post("/mcp", json="string")
        assert resp.status_code == 200


# --- Tool tests ---

class TestMCPTools:
    @pytest.mark.asyncio
    async def test_list_experts(self):
        mock_engine = MagicMock()
        mock_engine.get_all_experts.return_value = []
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import list_experts
            result = json.loads(await list_experts())
        assert "experts" in result

    @pytest.mark.asyncio
    async def test_search_experts(self):
        with patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            from forge.mcp.tools import search_experts
            result = json.loads(await search_experts(query="test"))
        assert "results" in result

    @pytest.mark.asyncio
    async def test_stats(self):
        from forge.graph.models import GraphStats
        mock_engine = MagicMock()
        mock_engine.store.get_stats.return_value = GraphStats(total_nodes=5, total_edges=3)
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import stats
            result = json.loads(await stats())
        assert result["total_nodes"] == 5

    @pytest.mark.asyncio
    async def test_add_node(self):
        from forge.graph.models import Node, NodeType
        mock_engine = MagicMock()
        mock_engine.add_node.return_value = Node(id="n1", name="Test", node_type=NodeType.concept)
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import add_node
            result = json.loads(await add_node(name="Test", node_type="concept"))
        assert result["node"]["name"] == "Test"

    @pytest.mark.asyncio
    async def test_add_node_invalid_type(self):
        mock_engine = MagicMock()
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import add_node
            result = json.loads(await add_node(name="Test", node_type="invalid"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_add_edge(self):
        from forge.graph.models import Edge, EdgeType
        mock_engine = MagicMock()
        mock_engine.add_edge.return_value = Edge(
            id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.related_to
        )
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import add_edge
            result = json.loads(await add_edge(source_id="n1", target_id="n2", edge_type="related_to"))
        assert "edge" in result

    @pytest.mark.asyncio
    async def test_update_edge(self):
        from forge.graph.models import Edge, EdgeType
        mock_engine = MagicMock()
        mock_engine.store.update_edge.return_value = Edge(
            id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.related_to, weight=0.5
        )
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import update_edge
            result = json.loads(await update_edge(edge_id="e1", weight=0.5))
        assert "edge" in result

    @pytest.mark.asyncio
    async def test_expire_edge(self):
        from forge.graph.models import Edge, EdgeType
        mock_engine = MagicMock()
        mock_engine.store.expire_edge.return_value = Edge(
            id="e1", source_id="n1", target_id="n2", edge_type=EdgeType.related_to
        )
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import expire_edge
            result = json.loads(await expire_edge(edge_id="e1"))
        assert "edge" in result

    @pytest.mark.asyncio
    async def test_query_graph(self):
        mock_engine = MagicMock()
        mock_engine.store.search_nodes.return_value = []
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import query_graph
            result = json.loads(await query_graph(query="ML"))
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_traverse_graph(self):
        from forge.graph.models import TraversalResult
        mock_engine = MagicMock()
        mock_engine.traverse.return_value = TraversalResult(root_id="n1")
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import traverse_graph
            result = json.loads(await traverse_graph(node_id="n1"))
        assert result["root_id"] == "n1"

    @pytest.mark.asyncio
    async def test_find_contradictions(self):
        mock_engine = MagicMock()
        mock_engine.find_contradictions.return_value = []
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import find_contradictions
            result = json.loads(await find_contradictions())
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_find_experts_for(self):
        mock_engine = MagicMock()
        mock_engine.expert_authority.return_value = []
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import find_experts_for
            result = json.loads(await find_experts_for(topic="ML"))
        assert result["topic"] == "ML"

    @pytest.mark.asyncio
    async def test_graph_timeline(self):
        mock_engine = MagicMock()
        mock_engine.find_changes_since.return_value = []
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import graph_timeline
            result = json.loads(await graph_timeline(since="2024-01-01"))
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_pagerank(self):
        mock_engine = MagicMock()
        mock_engine.pagerank.return_value = {}
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import pagerank
            result = json.loads(await pagerank())
        assert "pagerank" in result

    @pytest.mark.asyncio
    async def test_communities(self):
        mock_engine = MagicMock()
        mock_engine.find_communities.return_value = {}
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import communities
            result = json.loads(await communities())
        assert result["num_communities"] == 0

    @pytest.mark.asyncio
    async def test_shortest_path(self):
        mock_engine = MagicMock()
        mock_engine.shortest_path.return_value = ["n1", "n2"]
        with patch("forge.mcp.tools._get_engine", return_value=mock_engine):
            from forge.mcp.tools import shortest_path
            result = json.loads(await shortest_path(source_id="n1", target_id="n2"))
        assert result["found"] is True

    @pytest.mark.asyncio
    async def test_auto_capture_status(self):
        from forge.mcp.tools import auto_capture_status
        result = json.loads(await auto_capture_status())
        assert "messages_captured" in result

    @pytest.mark.asyncio
    async def test_get_context(self):
        from forge.layers.engine import LayeredContext
        ctx = LayeredContext(query="test", layer_0="id", total_tokens=10, layers_used=[0])
        with patch("forge.mcp.tools._get_engine", return_value=MagicMock()), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=ctx), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            from forge.mcp.tools import get_context
            result = json.loads(await get_context(query="test"))
        assert "context" in result

    @pytest.mark.asyncio
    async def test_engine_not_initialized(self):
        with patch("forge.mcp.tools._get_engine", return_value=None):
            from forge.mcp.tools import list_experts
            result = json.loads(await list_experts())
        assert "error" in result
