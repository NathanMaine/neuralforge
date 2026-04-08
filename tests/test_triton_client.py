"""Tests for forge.core.triton_client — 18 tests covering embeddings, reranking, and error paths."""
import json

import httpx
import pytest
import pytest_asyncio

from forge.core import triton_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_response(embeddings: list[list[float]]) -> dict:
    """Build a Triton-style embedding response."""
    n = len(embeddings)
    dim = len(embeddings[0]) if embeddings else 0
    flat = [v for vec in embeddings for v in vec]
    return {
        "outputs": [
            {
                "name": "EMBEDDING",
                "shape": [n, dim],
                "datatype": "FP32",
                "data": flat,
            }
        ]
    }


def _make_rerank_response(scores: list[float]) -> dict:
    """Build a Triton-style rerank response."""
    return {
        "outputs": [
            {
                "name": "SCORE",
                "shape": [len(scores)],
                "datatype": "FP32",
                "data": scores,
            }
        ]
    }


class FakeResponse:
    """Minimal fake httpx.Response."""

    def __init__(self, status_code: int = 200, data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text or json.dumps(self._data)

    def raise_for_status(self):
        if self.status_code >= 400:
            resp = httpx.Response(self.status_code, request=httpx.Request("POST", "http://fake"))
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=resp.request, response=resp
            )

    def json(self):
        return self._data


class FakeAsyncClient:
    """Fake httpx.AsyncClient that records calls and returns canned responses."""

    def __init__(self, responses: list[FakeResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict]] = []
        self.is_closed = False
        self._call_idx = 0

    async def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append((url, kwargs))
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return FakeResponse(500, text="No canned response")

    async def aclose(self):
        self.is_closed = True


@pytest.fixture(autouse=True)
def _reset_client():
    """Ensure the module-level client is reset between tests."""
    triton_client._client = None
    yield
    triton_client._client = None


@pytest.fixture()
def fake_client(monkeypatch):
    """Inject a FakeAsyncClient and return it for assertions."""
    client = FakeAsyncClient()
    monkeypatch.setattr(triton_client, "_client", client)

    # Also prevent _get_client from creating a real client
    async def _patched():
        return client

    monkeypatch.setattr(triton_client, "_get_client", _patched)
    return client


# ===========================================================================
# Embedding tests
# ===========================================================================

class TestInferEmbedding:
    """Tests for infer_embedding()."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, fake_client):
        result = await triton_client.infer_embedding([])
        assert result == []
        assert len(fake_client.calls) == 0

    @pytest.mark.asyncio
    async def test_single_text_embedding(self, fake_client):
        fake_client.responses = [
            FakeResponse(200, _make_embed_response([[0.1, 0.2, 0.3]]))
        ]
        result = await triton_client.infer_embedding(["hello"])
        assert result == [[0.1, 0.2, 0.3]]
        assert len(fake_client.calls) == 1

    @pytest.mark.asyncio
    async def test_batch_embedding(self, fake_client):
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        fake_client.responses = [FakeResponse(200, _make_embed_response(vecs))]
        result = await triton_client.infer_embedding(["a", "b", "c"])
        assert result == vecs

    @pytest.mark.asyncio
    async def test_batch_chunking(self, monkeypatch, fake_client):
        """Texts exceeding EMBED_BATCH_SIZE are sent in multiple requests."""
        monkeypatch.setattr(triton_client, "EMBED_BATCH_SIZE", 2)
        v1 = [[1.0, 2.0], [3.0, 4.0]]
        v2 = [[5.0, 6.0]]
        fake_client.responses = [
            FakeResponse(200, _make_embed_response(v1)),
            FakeResponse(200, _make_embed_response(v2)),
        ]
        result = await triton_client.infer_embedding(["a", "b", "c"])
        assert result == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        assert len(fake_client.calls) == 2

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self, fake_client):
        fake_client.responses = [FakeResponse(500, text="Internal Server Error")]
        result = await triton_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, monkeypatch):
        async def _raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        client = FakeAsyncClient()
        client.post = _raise_timeout
        monkeypatch.setattr(triton_client, "_client", client)

        async def _patched():
            return client

        monkeypatch.setattr(triton_client, "_get_client", _patched)

        result = await triton_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_outputs_returns_none(self, fake_client):
        fake_client.responses = [FakeResponse(200, {"outputs": []})]
        result = await triton_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_unexpected_shape_returns_none(self, fake_client):
        fake_client.responses = [
            FakeResponse(200, {
                "outputs": [{"name": "EMBEDDING", "shape": [2, 3, 4], "data": [0.0] * 24}]
            })
        ]
        result = await triton_client.infer_embedding(["hello"])
        assert result is None

    @pytest.mark.asyncio
    async def test_single_dim_shape(self, fake_client):
        """Shape [dim] for a single vector is handled."""
        fake_client.responses = [
            FakeResponse(200, {
                "outputs": [{"name": "EMBEDDING", "shape": [3], "data": [0.1, 0.2, 0.3]}]
            })
        ]
        result = await triton_client.infer_embedding(["hello"])
        assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_url_uses_config(self, fake_client, monkeypatch):
        monkeypatch.setattr(triton_client, "TRITON_URL", "http://gpu:9000")
        monkeypatch.setattr(triton_client, "EMBED_MODEL", "my-embed")
        fake_client.responses = [
            FakeResponse(200, _make_embed_response([[1.0]]))
        ]
        await triton_client.infer_embedding(["x"])
        called_url = fake_client.calls[0][0]
        assert called_url == "http://gpu:9000/v2/models/my-embed/infer"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_none(self, monkeypatch):
        async def _raise(*args, **kwargs):
            raise RuntimeError("kaboom")

        client = FakeAsyncClient()
        client.post = _raise
        monkeypatch.setattr(triton_client, "_client", client)

        async def _patched():
            return client

        monkeypatch.setattr(triton_client, "_get_client", _patched)

        result = await triton_client.infer_embedding(["hello"])
        assert result is None


# ===========================================================================
# Rerank tests
# ===========================================================================

class TestInferRerank:
    """Tests for infer_rerank()."""

    @pytest.mark.asyncio
    async def test_empty_documents_returns_empty_list(self, fake_client):
        result = await triton_client.infer_rerank("query", [])
        assert result == []
        assert len(fake_client.calls) == 0

    @pytest.mark.asyncio
    async def test_empty_query_returns_none(self, fake_client):
        result = await triton_client.infer_rerank("", ["doc1"])
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_rerank(self, fake_client):
        scores = [0.95, 0.42, 0.73]
        fake_client.responses = [FakeResponse(200, _make_rerank_response(scores))]
        result = await triton_client.infer_rerank("query", ["a", "b", "c"])
        assert result == scores

    @pytest.mark.asyncio
    async def test_rerank_http_error(self, fake_client):
        fake_client.responses = [FakeResponse(503, text="Service Unavailable")]
        result = await triton_client.infer_rerank("query", ["a"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rerank_timeout(self, monkeypatch):
        async def _raise_timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        client = FakeAsyncClient()
        client.post = _raise_timeout
        monkeypatch.setattr(triton_client, "_client", client)

        async def _patched():
            return client

        monkeypatch.setattr(triton_client, "_get_client", _patched)

        result = await triton_client.infer_rerank("query", ["doc"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rerank_missing_outputs(self, fake_client):
        fake_client.responses = [FakeResponse(200, {"outputs": []})]
        result = await triton_client.infer_rerank("query", ["doc"])
        assert result is None

    @pytest.mark.asyncio
    async def test_rerank_url_uses_config(self, fake_client, monkeypatch):
        monkeypatch.setattr(triton_client, "TRITON_URL", "http://rerank-box:7000")
        monkeypatch.setattr(triton_client, "RERANK_MODEL", "my-reranker")
        fake_client.responses = [FakeResponse(200, _make_rerank_response([0.5]))]
        await triton_client.infer_rerank("q", ["d"])
        called_url = fake_client.calls[0][0]
        assert called_url == "http://rerank-box:7000/v2/models/my-reranker/infer"


# ===========================================================================
# Client lifecycle
# ===========================================================================

class TestClientLifecycle:
    """Tests for _get_client() and close_client()."""

    @pytest.mark.asyncio
    async def test_close_client(self):
        # Force client creation
        triton_client._client = FakeAsyncClient()
        await triton_client.close_client()
        assert triton_client._client is None
