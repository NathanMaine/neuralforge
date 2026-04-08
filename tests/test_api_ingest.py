"""Tests for forge.api.routes.ingest -- upload, scrape, auto-capture."""
import io
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.load = MagicMock()
    engine.node_count.return_value = 0
    engine.edge_count.return_value = 0
    engine.store = MagicMock()
    engine.store.save = MagicMock()
    engine.store.get_stats.return_value = MagicMock(
        total_nodes=0, total_edges=0, active_edges=0, expired_edges=0,
        node_type_counts={}, edge_type_counts={},
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
        from forge.api.main import app
        import forge.api.main as m
        with TestClient(app) as c:
            m.graph_engine = mock_engine
            yield c


class TestDocumentUpload:
    def test_upload_txt(self, client):
        with patch("forge.api.routes.ingest.ingest_chunks", new_callable=AsyncMock, return_value=3):
            resp = client.post(
                "/api/ingest/documents",
                files={"file": ("test.txt", b"This is a test document with enough text for chunking. " * 20, "text/plain")},
                data={"creator": "tester", "title": "Test Doc"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["creator"] == "tester"
        assert data["ingested"] == 3

    def test_upload_unsupported_format(self, client):
        resp = client.post(
            "/api/ingest/documents",
            files={"file": ("test.xyz", b"data", "application/octet-stream")},
            data={"creator": "tester"},
        )
        assert resp.status_code == 400

    def test_upload_empty_document(self, client):
        resp = client.post(
            "/api/ingest/documents",
            files={"file": ("test.txt", b"", "text/plain")},
            data={"creator": "tester"},
        )
        assert resp.status_code == 400

    def test_upload_md(self, client):
        content = b"# Test\n\nThis is markdown content with enough text. " * 20
        with patch("forge.api.routes.ingest.ingest_chunks", new_callable=AsyncMock, return_value=2):
            resp = client.post(
                "/api/ingest/documents",
                files={"file": ("doc.md", content, "text/markdown")},
                data={"creator": "md-tester"},
            )
        assert resp.status_code == 200

    def test_upload_pii_scrubbing(self, client):
        content = b"Call me at 555-123-4567 or email test@example.com. " * 20
        with patch("forge.api.routes.ingest.ingest_chunks", new_callable=AsyncMock, return_value=1):
            resp = client.post(
                "/api/ingest/documents",
                files={"file": ("pii.txt", content, "text/plain")},
                data={"creator": "tester"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "pii_scrubbed" in data


class TestBlogScraping:
    def test_trigger_scrape(self, client):
        with patch("forge.api.routes.ingest.scrape_blog", new_callable=AsyncMock, return_value={
            "source": "Test Blog", "discovered": 10, "extracted": 5, "ingested": 15, "skipped": 5,
        }):
            resp = client.post("/api/ingest/blog", json={
                "url": "https://example.com/blog",
                "name": "Test Blog",
                "creator": "blogger",
            })
        assert resp.status_code == 200
        assert resp.json()["discovered"] == 10


class TestSourceManagement:
    def test_list_sources(self, client):
        with patch("forge.api.routes.ingest.load_sources", return_value=[]):
            resp = client.get("/api/ingest/sources")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_add_source(self, client):
        with patch("forge.api.routes.ingest.add_source", return_value={
            "url": "https://example.com", "name": "Example", "creator": "test",
        }):
            resp = client.post("/api/ingest/sources", json={
                "url": "https://example.com",
                "name": "Example",
                "creator": "test",
            })
        assert resp.status_code == 200
        assert "source" in resp.json()


class TestConversationUpload:
    def test_upload_conversation(self, client):
        text = "Human: What is ML?\nAssistant: Machine learning is a branch of AI."
        with patch("forge.api.routes.ingest.mine_conversation", new_callable=AsyncMock, return_value={
            "format": "claude", "messages": 2, "chunks": 1, "ingested": 1,
            "entities": {}, "edges": 0, "classifications": {"factual": 1},
        }):
            resp = client.post(
                "/api/ingest/conversations",
                files={"file": ("chat.txt", text.encode(), "text/plain")},
                data={"creator": "conv-tester"},
            )
        assert resp.status_code == 200
        assert resp.json()["messages"] == 2

    def test_upload_empty_conversation(self, client):
        resp = client.post(
            "/api/ingest/conversations",
            files={"file": ("empty.txt", b"", "text/plain")},
            data={"creator": "tester"},
        )
        assert resp.status_code == 400


class TestAutoCapture:
    def test_auto_capture(self, client):
        with patch("forge.api.routes.ingest.ingest_chunks", new_callable=AsyncMock, return_value=2):
            resp = client.post("/api/ingest/auto-capture", json={
                "messages": [
                    {"role": "user", "content": "What is deep learning? " * 20},
                    {"role": "assistant", "content": "Deep learning is a subset of ML. " * 20},
                ],
                "creator": "auto",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["captured"] == 2

    def test_auto_capture_empty(self, client):
        resp = client.post("/api/ingest/auto-capture", json={
            "messages": [],
        })
        assert resp.status_code == 200
        assert resp.json()["captured"] == 0

    def test_auto_capture_status(self, client):
        resp = client.get("/api/ingest/auto-capture/status")
        assert resp.status_code == 200
        assert "messages_captured" in resp.json()

    def test_auto_capture_pii(self, client):
        with patch("forge.api.routes.ingest.ingest_chunks", new_callable=AsyncMock, return_value=1):
            resp = client.post("/api/ingest/auto-capture", json={
                "messages": [
                    {"role": "user", "content": "My SSN is 123-45-6789 and email is test@test.com. " * 20},
                ],
            })
        assert resp.status_code == 200
        assert "pii_scrubbed" in resp.json()
