"""Tests for forge.api.main -- FastAPI app, health, lifespan, index."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient


# --- Fixtures ---

@pytest.fixture
def _patch_lifespan():
    """Patch lifespan dependencies so the app starts without real services."""
    with patch("forge.api.main.GraphStore") as mock_store_cls, \
         patch("forge.api.main.GraphEngine") as mock_engine_cls, \
         patch("forge.api.main.bootstrap_graph", return_value={"skipped": True}), \
         patch("forge.api.main.GuardrailsEngine") as mock_gr_cls, \
         patch("forge.api.main.start_scheduler"), \
         patch("forge.api.main.stop_scheduler"):

        mock_store = MagicMock()
        mock_store.save = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine.node_count.return_value = 5
        mock_engine.edge_count.return_value = 3
        mock_engine.load = MagicMock()
        mock_engine.store = mock_store
        mock_engine_cls.return_value = mock_engine

        mock_gr = MagicMock()
        mock_gr.enabled = False
        mock_gr_cls.return_value = mock_gr

        yield {"store": mock_store, "engine": mock_engine, "guardrails": mock_gr}


@pytest.fixture
def client(_patch_lifespan):
    """Create a TestClient with patched lifespan."""
    from forge.api.main import app
    with TestClient(app) as c:
        yield c


# --- Tests ---

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with patch("forge.core.qdrant_client.get_status", return_value="green"):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_version(self, client):
        with patch("forge.core.qdrant_client.get_status", return_value="green"):
            data = client.get("/health").json()
        assert data["version"] == "1.0.0"

    def test_health_has_status(self, client):
        with patch("forge.core.qdrant_client.get_status", return_value="green"):
            data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_services(self, client):
        with patch("forge.core.qdrant_client.get_status", return_value="green"):
            data = client.get("/health").json()
        assert "services" in data
        assert "qdrant" in data["services"]
        assert "graph" in data["services"]
        assert "guardrails" in data["services"]
        assert "scheduler" in data["services"]

    def test_health_qdrant_down(self, client):
        with patch("forge.core.qdrant_client.get_status", side_effect=Exception("down")):
            data = client.get("/health").json()
        assert data["services"]["qdrant"] == "red"


class TestIndexRoute:
    def test_index_returns_response(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_without_web_dir(self, client):
        with patch("os.path.isfile", return_value=False):
            resp = client.get("/")
        # Should still return 200 (either HTML or JSON fallback)
        assert resp.status_code == 200


class TestLifespan:
    def test_app_creates_engine(self, _patch_lifespan):
        from forge.api.main import app
        with TestClient(app):
            from forge.api.main import get_graph_engine
            engine = get_graph_engine()
            assert engine is not None

    def test_app_creates_guardrails(self, _patch_lifespan):
        from forge.api.main import app
        with TestClient(app):
            from forge.api.main import get_guardrails_engine
            gr = get_guardrails_engine()
            assert gr is not None

    def test_app_title(self, _patch_lifespan):
        from forge.api.main import app
        assert app.title == "NeuralForge"

    def test_app_version(self, _patch_lifespan):
        from forge.api.main import app
        assert app.version == "1.0.0"
