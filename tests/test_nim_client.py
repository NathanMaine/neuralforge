"""Tests for forge.core.nim_client -- 25 tests covering chat, JSON classification, streaming."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio  # noqa: F401 — ensures the plugin is importable

from forge.core import nim_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nim_response(content: str, model: str = "meta/llama-3.1-8b-instruct") -> dict:
    """Build a minimal OpenAI-style chat completion response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def _mock_httpx_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# =========================================================================
# chat_completion tests
# =========================================================================


class TestChatCompletion:
    """Tests for nim_client.chat_completion."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Successful chat completion returns full response dict."""
        expected = _nim_response("Hello, world!")
        mock_resp = _mock_httpx_response(200, expected)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        assert result == expected
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_http_500_returns_none(self):
        """HTTP 500 from NIM returns None."""
        mock_resp = _mock_httpx_response(500, text="Internal Server Error")
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        """Timeout returns None."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self):
        """Connection refused returns None."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self):
        """Empty response body returns None."""
        mock_resp = _mock_httpx_response(200, json_data={})
        # json() returns falsy {} -- but dict is truthy only when non-empty
        # We need to test with actually empty body
        mock_resp.json.return_value = {}
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_model_from_config(self):
        """Payload includes model from forge.config."""
        expected = _nim_response("ok")
        mock_resp = _mock_httpx_response(200, expected)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "meta/llama-3.1-8b-instruct"

    @pytest.mark.asyncio
    async def test_custom_temperature_and_max_tokens(self):
        """Custom temperature and max_tokens are forwarded."""
        expected = _nim_response("ok")
        mock_resp = _mock_httpx_response(200, expected)
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}],
                max_tokens=500,
                temperature=0.1,
            )

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["max_tokens"] == 500
        assert payload["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_unexpected_exception_returns_none(self):
        """Any unexpected exception is caught and returns None."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = RuntimeError("unexpected")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await nim_client.chat_completion(
                [{"role": "user", "content": "Hi"}]
            )

        assert result is None


# =========================================================================
# _extract_json tests
# =========================================================================


class TestExtractJson:
    """Tests for the internal JSON extraction helper."""

    def test_plain_json(self):
        assert nim_client._extract_json('{"key": "value"}') == {"key": "value"}

    def test_markdown_wrapped_json(self):
        text = '```json\n{"category": "tech", "confidence": 0.95}\n```'
        result = nim_client._extract_json(text)
        assert result == {"category": "tech", "confidence": 0.95}

    def test_markdown_no_language_tag(self):
        text = '```\n{"a": 1}\n```'
        result = nim_client._extract_json(text)
        assert result == {"a": 1}

    def test_malformed_json_returns_none(self):
        assert nim_client._extract_json("not json at all") is None

    def test_whitespace_padding(self):
        assert nim_client._extract_json('   {"ok": true}   ') == {"ok": True}


# =========================================================================
# classify_json tests
# =========================================================================


class TestClassifyJson:
    """Tests for nim_client.classify_json."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self):
        """Model returns clean JSON -- parsed and returned."""
        nim_resp = _nim_response('{"category": "tech", "confidence": 0.92}')
        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = nim_resp
            result = await nim_client.classify_json("Classify this text")

        assert result == {"category": "tech", "confidence": 0.92}

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json_response(self):
        """Model wraps JSON in markdown fences -- still parsed."""
        content = '```json\n{"label": "finance", "confidence": 0.88}\n```'
        nim_resp = _nim_response(content)
        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = nim_resp
            result = await nim_client.classify_json("Classify this text")

        assert result == {"label": "finance", "confidence": 0.88}

    @pytest.mark.asyncio
    async def test_malformed_json_retries(self):
        """Malformed JSON triggers retry; succeeds on second attempt."""
        bad = _nim_response("This is not JSON, sorry")
        good = _nim_response('{"label": "sports"}')

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad
            return good

        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.side_effect = side_effect
            result = await nim_client.classify_json("Classify this", max_retries=2)

        assert result == {"label": "sports"}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """All retries fail -- returns None."""
        bad = _nim_response("nope, no json here")
        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = bad
            result = await nim_client.classify_json("Classify this", max_retries=3)

        assert result is None
        assert mock_cc.call_count == 3

    @pytest.mark.asyncio
    async def test_confidence_field_present(self):
        """Returned dict includes a confidence field."""
        nim_resp = _nim_response('{"category": "health", "confidence": 0.76}')
        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = nim_resp
            result = await nim_client.classify_json("Health article classification")

        assert "confidence" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_chat_completion_returns_none_propagates(self):
        """If chat_completion returns None (network error), classify_json returns None."""
        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = None
            result = await nim_client.classify_json("Classify this")

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_response_structure(self):
        """Response missing choices key triggers retry."""
        bad_structure = {"id": "test", "object": "chat.completion"}
        good = _nim_response('{"label": "ok"}')

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_structure
            return good

        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.side_effect = side_effect
            result = await nim_client.classify_json("test", max_retries=2)

        assert result == {"label": "ok"}

    @pytest.mark.asyncio
    async def test_uses_low_temperature(self):
        """classify_json passes temperature=0.2 for deterministic output."""
        nim_resp = _nim_response('{"label": "test"}')
        with patch.object(nim_client, "chat_completion", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = nim_resp
            await nim_client.classify_json("test")

        call_kwargs = mock_cc.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.2


# =========================================================================
# stream_completion tests
# =========================================================================


class TestStreamCompletion:
    """Tests for nim_client.stream_completion."""

    @staticmethod
    def _make_stream_mocks(chunks):
        """Build mock client + response that yields *chunks* as SSE lines."""
        async def mock_aiter_lines():
            for line in chunks:
                yield line

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.aiter_lines = mock_aiter_lines
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        # client.stream() must return a sync context-manager-like object,
        # not a coroutine, because the production code does:
        #   async with client.stream(...) as resp:
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        return mock_client

    @pytest.mark.asyncio
    async def test_yields_parsed_sse_lines(self):
        """Valid SSE stream yields parsed JSON dicts."""
        chunks = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" world"}}]}',
            "data: [DONE]",
        ]
        mock_client = self._make_stream_mocks(chunks)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = []
            async for chunk in nim_client.stream_completion(
                [{"role": "user", "content": "Hi"}]
            ):
                results.append(chunk)

        assert len(results) == 2
        assert results[0]["choices"][0]["delta"]["content"] == "Hello"
        assert results[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    async def test_skips_empty_and_non_data_lines(self):
        """Empty lines and non-data lines are skipped."""
        chunks = [
            "",
            ": keep-alive",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "",
            "data: [DONE]",
        ]
        mock_client = self._make_stream_mocks(chunks)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = []
            async for chunk in nim_client.stream_completion(
                [{"role": "user", "content": "Hi"}]
            ):
                results.append(chunk)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_stream_timeout_yields_nothing(self):
        """Timeout during streaming yields nothing (no exception raised)."""
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = []
            async for chunk in nim_client.stream_completion(
                [{"role": "user", "content": "Hi"}]
            ):
                results.append(chunk)

        assert results == []

    @pytest.mark.asyncio
    async def test_stream_connection_error(self):
        """Connection error during streaming yields nothing."""
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = []
            async for chunk in nim_client.stream_completion(
                [{"role": "user", "content": "Hi"}]
            ):
                results.append(chunk)

        assert results == []

    @pytest.mark.asyncio
    async def test_stream_http_500(self):
        """HTTP 500 during streaming yields nothing."""
        mock_resp = MagicMock()
        err_resp = MagicMock(spec=httpx.Response)
        err_resp.status_code = 500
        err_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="HTTP 500", request=MagicMock(), response=err_resp
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = []
            async for chunk in nim_client.stream_completion(
                [{"role": "user", "content": "Hi"}]
            ):
                results.append(chunk)

        assert results == []

    @pytest.mark.asyncio
    async def test_stream_malformed_sse_chunk_skipped(self):
        """Malformed JSON in an SSE chunk is skipped, valid chunks still yielded."""
        chunks = [
            'data: {"choices":[{"delta":{"content":"a"}}]}',
            "data: {broken json",
            'data: {"choices":[{"delta":{"content":"b"}}]}',
            "data: [DONE]",
        ]
        mock_client = self._make_stream_mocks(chunks)

        with patch("httpx.AsyncClient", return_value=mock_client):
            results = []
            async for chunk in nim_client.stream_completion(
                [{"role": "user", "content": "Hi"}]
            ):
                results.append(chunk)

        assert len(results) == 2
        assert results[0]["choices"][0]["delta"]["content"] == "a"
        assert results[1]["choices"][0]["delta"]["content"] == "b"
