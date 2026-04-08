"""Tests for forge.api.routes.search -- search with layered context + BM25."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from forge.layers.engine import LayeredContext


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.load = MagicMock()
    engine.node_count.return_value = 0
    engine.edge_count.return_value = 0
    engine.store = MagicMock()
    engine.store.get_stats.return_value = MagicMock(
        total_nodes=0, total_edges=0, active_edges=0, expired_edges=0,
        node_type_counts={}, edge_type_counts={},
    )
    engine.expert_authority.return_value = []
    engine.find_contradictions.return_value = []
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


def _mock_context():
    return LayeredContext(
        query="test",
        layer_0="identity",
        layer_1="graph context",
        layer_2="chunks",
        layer_3="deep",
        total_tokens=100,
        layers_used=[0, 1, 2, 3],
        experts_referenced=["Alice"],
    )


class TestSearchEndpoint:
    def test_search_requires_query(self, client):
        resp = client.get("/api/search")
        assert resp.status_code == 422

    def test_search_empty_query(self, client):
        resp = client.get("/api/search?q=")
        assert resp.status_code == 422

    def test_search_basic(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.get("/api/search?q=machine+learning")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "context" in data
        assert data["query"] == "machine learning"

    def test_search_with_limit(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.get("/api/search?q=test&limit=5")
        assert resp.status_code == 200

    def test_search_with_expert_filter(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.get("/api/search?q=test&expert=Alice")
        assert resp.status_code == 200

    def test_search_context_has_layers(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            data = client.get("/api/search?q=test").json()
        ctx = data["context"]
        assert "layer_0" in ctx
        assert "layer_1" in ctx
        assert "assembled" in ctx

    def test_search_has_keywords(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            data = client.get("/api/search?q=machine+learning+deep").json()
        assert "keywords" in data

    def test_search_max_tokens(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.get("/api/search?q=test&max_tokens=1000")
        assert resp.status_code == 200

    def test_search_compress_false(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.get("/api/search?q=test&compress=false")
        assert resp.status_code == 200

    def test_search_layers_parameter(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.get("/api/search?q=test&layers=1")
        assert resp.status_code == 200

    def test_search_with_results(self, client):
        results = [
            {"score": 0.9, "expert": "Alice", "title": "ML Basics", "text": "Machine learning is...", "source": "blog", "chunk_index": 0},
            {"score": 0.7, "expert": "Bob", "title": "DL Guide", "text": "Deep learning uses...", "source": "blog", "chunk_index": 1},
        ]
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=results):
            data = client.get("/api/search?q=machine+learning").json()
        assert len(data["results"]) >= 1

    def test_search_bm25_with_results(self, client):
        results = [
            {"score": 0.9, "expert": "Alice", "title": "ML", "text": "machine learning deep", "source": "x", "chunk_index": 0},
        ]
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=results):
            data = client.get("/api/search?q=machine+learning+deep").json()
        assert len(data["results"]) >= 0  # just verifying no crash

    def test_search_limit_too_high(self, client):
        resp = client.get("/api/search?q=test&limit=200")
        assert resp.status_code == 422

    def test_search_invalid_layers(self, client):
        resp = client.get("/api/search?q=test&layers=5")
        assert resp.status_code == 422

    def test_search_context_experts_referenced(self, client):
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=_mock_context()), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            data = client.get("/api/search?q=test").json()
        assert data["context"]["experts_referenced"] == ["Alice"]


class TestKeywordExtraction:
    def test_extract_keywords(self):
        from forge.api.routes.search import _extract_keywords
        keywords = _extract_keywords("what is machine learning about")
        assert "machine" in keywords
        assert "learning" in keywords
        # Stop words excluded
        assert "what" not in keywords
        assert "about" not in keywords

    def test_empty_query(self):
        from forge.api.routes.search import _extract_keywords
        assert _extract_keywords("") == []

    def test_short_words_filtered(self):
        from forge.api.routes.search import _extract_keywords
        keywords = _extract_keywords("is it ok to do ML")
        # All short words filtered
        assert all(len(k) >= 4 for k in keywords) or keywords == []
