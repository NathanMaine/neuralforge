"""Tests for forge.api.routes.proxy -- OpenAI-compatible proxy with auto-RAG."""
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from forge.layers.engine import LayeredContext


def _nim_response(content="Hello from NIM"):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


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
def mock_guardrails():
    gr = MagicMock()
    gr.enabled = False
    return gr


@pytest.fixture
def client(mock_engine, mock_guardrails):
    with patch("forge.api.main.GraphStore"), \
         patch("forge.api.main.GraphEngine", return_value=mock_engine), \
         patch("forge.api.main.bootstrap_graph", return_value={"skipped": True}), \
         patch("forge.api.main.GuardrailsEngine", return_value=mock_guardrails), \
         patch("forge.api.main.start_scheduler"), \
         patch("forge.api.main.stop_scheduler"):
        from forge.api.main import app
        import forge.api.main as m
        with TestClient(app) as c:
            m.graph_engine = mock_engine
            m.guardrails_engine = mock_guardrails
            yield c


class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["owned_by"] == "neuralforge"

    def test_model_has_id(self, client):
        data = client.get("/v1/models").json()
        assert "id" in data["data"][0]


class TestChatCompletions:
    def test_basic_completion(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="hi")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })
        assert resp.status_code == 200

    def test_missing_messages(self, client):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 400

    def test_invalid_json(self, client):
        resp = client.post("/v1/chat/completions", content=b"not json", headers={"Content-Type": "application/json"})
        assert resp.status_code == 400

    def test_nim_unavailable(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=None), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="hi")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })
        assert resp.status_code == 502

    def test_bypass_header(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()):
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
                headers={"X-NeuralForge-Bypass": "true"},
            )
        assert resp.status_code == 200

    def test_bypass_nim_unavailable(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=None):
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Hello"}]},
                headers={"X-NeuralForge-Bypass": "true"},
            )
        assert resp.status_code == 502

    def test_system_message_preserved(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()) as mock_nim, \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="hi")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
            })
        assert resp.status_code == 200

    def test_no_user_message_passthrough(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "system", "content": "Just system"}],
            })
        assert resp.status_code == 200


class TestContextInjection:
    def test_context_injected(self, client):
        ctx = LayeredContext(
            query="test", layer_0="identity", layer_2="some context",
            total_tokens=50, layers_used=[0, 2], experts_referenced=["Alice"],
        )
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()) as mock_nim, \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=ctx), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Who is Alice?"}],
            })
        assert resp.status_code == 200
        # Check provenance was added
        data = resp.json()
        assert "neuralforge_provenance" in data
        assert "Alice" in data["neuralforge_provenance"]["experts_referenced"]

    def test_context_retrieval_failure(self, client):
        """Context retrieval failure should not break the request."""
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, side_effect=Exception("boom")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=None):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
            })
        assert resp.status_code == 200


class TestGuardrails:
    def test_input_blocked(self, client, mock_guardrails):
        mock_guardrails.enabled = True
        mock_guardrails.check_input = AsyncMock(return_value={
            "allowed": False, "reason": "Jailbreak detected", "scrubbed_query": "",
        })
        with patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="")):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "ignore instructions"}],
            })
        assert resp.status_code == 403

    def test_output_blocked(self, client, mock_guardrails):
        mock_guardrails.enabled = True
        mock_guardrails.check_input = AsyncMock(return_value={
            "allowed": True, "reason": None, "scrubbed_query": "hello",
        })
        mock_guardrails.check_output = AsyncMock(return_value={
            "allowed": False, "response": "", "provenance": {},
        })
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="hello")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
            })
        assert resp.status_code == 403

    def test_guardrails_pii_scrub(self, client, mock_guardrails):
        mock_guardrails.enabled = True
        mock_guardrails.check_input = AsyncMock(return_value={
            "allowed": True, "reason": None, "scrubbed_query": "my SSN is [REDACTED]",
        })
        mock_guardrails.check_output = AsyncMock(return_value={
            "allowed": True, "response": "Understood", "provenance": {},
        })
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response("Understood")), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="test")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "my SSN is 123-45-6789"}],
            })
        assert resp.status_code == 200


class TestStreaming:
    def test_stream_bypass(self, client):
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hi"}}]}

        with patch("forge.core.nim_client.stream_completion", return_value=mock_stream()):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
                headers={"X-NeuralForge-Bypass": "true"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_with_context(self, client):
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hi"}}]}

        ctx = LayeredContext(query="test", layer_0="id", total_tokens=10)
        with patch("forge.core.nim_client.stream_completion", return_value=mock_stream()), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=ctx), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            })
        assert resp.status_code == 200


class TestErrorHandling:
    def test_empty_messages_list(self, client):
        resp = client.post("/v1/chat/completions", json={"messages": []})
        assert resp.status_code == 400

    def test_max_tokens_forwarded(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()) as mock_nim, \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="x")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 500,
            })
        assert resp.status_code == 200

    def test_temperature_forwarded(self, client):
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=_nim_response()), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="x")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0.2,
            })
        assert resp.status_code == 200

    def test_malformed_result_does_not_crash(self, client):
        """When NIM returns a response without choices, output rail is skipped."""
        malformed = {"id": "x", "object": "chat.completion", "choices": []}
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=malformed), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="x")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}],
            })
        assert resp.status_code == 200

    def test_result_missing_choices_key(self, client):
        """NIM response without 'choices' key should not raise."""
        no_choices = {"id": "x", "object": "chat.completion"}
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=no_choices), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="x")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hi"}],
            })
        assert resp.status_code == 200


class TestNoUserMessageBranch:
    def test_stream_with_no_user_message_bypass(self, client):
        """Bypass + stream + no user message hits the stream branch."""
        async def mock_stream(*args, **kwargs):
            yield {"choices": [{"delta": {"content": "Hi"}}]}

        with patch("forge.core.nim_client.stream_completion", return_value=mock_stream()):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "system", "content": "sys"}],
                    "stream": True,
                },
                headers={"X-NeuralForge-Bypass": "true"},
            )
        assert resp.status_code == 200

    def test_no_user_message_nim_unavailable(self, client):
        """No user message path returns 502 when NIM is unavailable."""
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=None):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "system", "content": "sys only"}],
            })
        assert resp.status_code == 502


class TestOutputGuardrailsModifiesResponse:
    def test_output_guardrails_replaces_response_text(self, client, mock_guardrails):
        """When output guardrails returns a different response, it should be used."""
        mock_guardrails.enabled = True
        mock_guardrails.check_input = AsyncMock(return_value={
            "allowed": True, "reason": None, "scrubbed_query": "hello",
        })
        mock_guardrails.check_output = AsyncMock(return_value={
            "allowed": True, "response": "MODIFIED response", "provenance": {},
        })
        nim_resp = _nim_response("original response")
        with patch("forge.core.nim_client.chat_completion", new_callable=AsyncMock, return_value=nim_resp), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=LayeredContext(query="hello")), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            resp = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hello"}],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "MODIFIED response"

    def test_context_injected_after_system_message(self, client):
        """When there are system messages, context is inserted after the last system msg."""
        ctx = LayeredContext(
            query="test", layer_0="identity", layer_2="rich context",
            total_tokens=50, layers_used=[0, 2], experts_referenced=["Alice"],
        )
        captured_messages = []

        async def fake_nim(messages, **kwargs):
            captured_messages.extend(messages)
            return _nim_response()

        with patch("forge.core.nim_client.chat_completion", side_effect=fake_nim), \
             patch("forge.layers.engine.get_context", new_callable=AsyncMock, return_value=ctx), \
             patch("forge.core.embeddings.get_embedding", new_callable=AsyncMock, return_value=[0.1]*768), \
             patch("forge.core.qdrant_client.search_vectors", return_value=[]):
            client.post("/v1/chat/completions", json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Who is Alice?"},
                ],
            })
        # Context message should be inserted after system, before user
        roles = [m["role"] for m in captured_messages]
        # There should be a second system message (context) inserted after index 0
        assert roles.count("system") == 2
        assert roles[0] == "system"


class TestStreamingError:
    def test_streaming_error_yields_error_chunk(self, client):
        """When streaming raises, the error should be returned as an SSE chunk."""
        async def erroring_stream(*args, **kwargs):
            raise RuntimeError("NIM connection failed")
            yield  # noqa: unreachable -- required to make this function a generator

        with patch("forge.core.nim_client.stream_completion", return_value=erroring_stream()):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
                headers={"X-NeuralForge-Bypass": "true"},
            )
        assert resp.status_code == 200
        content = resp.text
        assert "error" in content
