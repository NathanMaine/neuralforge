"""Tests for forge.api.middleware.guardrails -- guarded query pipeline."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from forge.api.middleware.guardrails import (
    check_query_input,
    check_response_output,
    guarded_pipeline,
)


class TestCheckQueryInput:
    @pytest.mark.asyncio
    async def test_no_engine_passthrough(self):
        result = await check_query_input("test query")
        assert result["allowed"] is True
        assert result["scrubbed_query"] == "test query"
        assert result["reason"] is None

    @pytest.mark.asyncio
    async def test_disabled_engine_passthrough(self):
        engine = MagicMock()
        engine.enabled = False
        result = await check_query_input("test query", engine)
        assert result["allowed"] is True
        assert result["scrubbed_query"] == "test query"
        assert result["reason"] is None

    @pytest.mark.asyncio
    async def test_enabled_engine_calls_check_input(self):
        engine = MagicMock()
        engine.enabled = True
        engine.check_input = AsyncMock(return_value={
            "allowed": True, "reason": None, "scrubbed_query": "clean query",
        })
        result = await check_query_input("test query", engine)
        engine.check_input.assert_called_once_with("test query")
        assert result["scrubbed_query"] == "clean query"

    @pytest.mark.asyncio
    async def test_enabled_engine_blocks_input(self):
        engine = MagicMock()
        engine.enabled = True
        engine.check_input = AsyncMock(return_value={
            "allowed": False, "reason": "jailbreak detected", "scrubbed_query": "bad",
        })
        result = await check_query_input("hack the planet", engine)
        assert result["allowed"] is False
        assert result["reason"] == "jailbreak detected"

    @pytest.mark.asyncio
    async def test_preserves_original_query_as_scrubbed_when_no_engine(self):
        result = await check_query_input("my query with content")
        assert result["scrubbed_query"] == "my query with content"


class TestCheckResponseOutput:
    @pytest.mark.asyncio
    async def test_no_engine_passthrough(self):
        result = await check_response_output("q", "response text")
        assert result["allowed"] is True
        assert result["response"] == "response text"
        assert result["provenance"] == {}

    @pytest.mark.asyncio
    async def test_disabled_engine_passthrough(self):
        engine = MagicMock()
        engine.enabled = False
        result = await check_response_output("q", "response text", engine)
        assert result["allowed"] is True
        assert result["response"] == "response text"
        assert result["provenance"] == {}

    @pytest.mark.asyncio
    async def test_enabled_engine_calls_check_output(self):
        engine = MagicMock()
        engine.enabled = True
        engine.check_output = AsyncMock(return_value={
            "allowed": True, "response": "verified response", "provenance": {"src": "A"},
        })
        result = await check_response_output("q", "response", engine, context={"k": "v"})
        engine.check_output.assert_called_once_with("q", "response", {"k": "v"})
        assert result["response"] == "verified response"
        assert result["provenance"] == {"src": "A"}

    @pytest.mark.asyncio
    async def test_enabled_engine_blocks_output(self):
        engine = MagicMock()
        engine.enabled = True
        engine.check_output = AsyncMock(return_value={
            "allowed": False, "response": "original", "provenance": {},
        })
        result = await check_response_output("q", "original", engine)
        assert result["allowed"] is False

    @pytest.mark.asyncio
    async def test_no_context_passes_none(self):
        engine = MagicMock()
        engine.enabled = True
        engine.check_output = AsyncMock(return_value={
            "allowed": True, "response": "ok", "provenance": {},
        })
        await check_response_output("q", "response", engine)
        engine.check_output.assert_called_once_with("q", "response", None)


class TestGuardedPipeline:
    @pytest.mark.asyncio
    async def test_no_engine_no_generate_fn_echoes_query(self):
        result = await guarded_pipeline("hello world")
        assert result["response"] == "hello world"
        assert result["input_check"]["allowed"] is True
        assert result["output_check"]["allowed"] is True

    @pytest.mark.asyncio
    async def test_no_engine_with_generate_fn(self):
        async def gen_fn(query, context):
            return f"answer: {query}"

        result = await guarded_pipeline("what is ML?", generate_fn=gen_fn)
        assert result["response"] == "answer: what is ML?"

    @pytest.mark.asyncio
    async def test_no_engine_generate_fn_receives_context(self):
        received = {}

        async def gen_fn(query, context):
            received["ctx"] = context
            return "ok"

        await guarded_pipeline("q", context={"k": "v"}, generate_fn=gen_fn)
        assert received["ctx"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_disabled_engine_echoes_query(self):
        engine = MagicMock()
        engine.enabled = False
        result = await guarded_pipeline("test", guardrails_engine=engine)
        assert result["response"] == "test"
        assert result["input_check"]["allowed"] is True

    @pytest.mark.asyncio
    async def test_enabled_engine_delegates_to_guarded_generate(self):
        engine = MagicMock()
        engine.enabled = True
        engine.guarded_generate = AsyncMock(return_value={
            "response": "guarded response",
            "input_check": {"allowed": True, "reason": None, "scrubbed_query": "test"},
            "output_check": {"allowed": True, "response": "guarded response", "provenance": {}},
        })
        result = await guarded_pipeline("test query", guardrails_engine=engine)
        engine.guarded_generate.assert_called_once()
        assert result["response"] == "guarded response"

    @pytest.mark.asyncio
    async def test_enabled_engine_passes_generate_fn(self):
        engine = MagicMock()
        engine.enabled = True
        engine.guarded_generate = AsyncMock(return_value={
            "response": "out", "input_check": {}, "output_check": {},
        })

        async def gen_fn(q, ctx):
            return "generated"

        await guarded_pipeline("q", guardrails_engine=engine, generate_fn=gen_fn)
        call_kwargs = engine.guarded_generate.call_args[1]
        assert call_kwargs["generate_fn"] is gen_fn
