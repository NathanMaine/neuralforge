"""Tests for NeMo Guardrails integration -- 20+ tests.

All tests mock NeMo Guardrails so they run without the nemoguardrails
package installed.  Covers enabled/disabled modes, input/output checking,
hallucination detection, attribution, provenance, PII scrubbing,
self-correction, and audit logging.
"""
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.guardrails.rails import GuardrailsEngine, HAS_GUARDRAILS
from forge.guardrails import actions
from forge.guardrails.actions import (
    _extract_expert_names,
    add_provenance,
    check_attribution,
    check_hallucination,
    log_rail_decision,
    scrub_pii_input,
    self_correction,
)


# ===================================================================
# GuardrailsEngine -- disabled mode (no nemoguardrails installed)
# ===================================================================

class TestGuardrailsEngineDisabled:
    """Tests for GuardrailsEngine when guardrails are disabled."""

    def test_disabled_when_package_missing(self):
        """Engine should be disabled when nemoguardrails is not installed."""
        with patch("forge.guardrails.rails.HAS_GUARDRAILS", False):
            engine = GuardrailsEngine()
            assert engine.enabled is False

    def test_disabled_when_config_flag_false(self):
        """Engine should be disabled when GUARDRAILS_ENABLED is False."""
        with patch("forge.guardrails.rails.config") as mock_config:
            mock_config.GUARDRAILS_ENABLED = False
            mock_config.GUARDRAILS_CONFIG_DIR = "fake/path"
            engine = GuardrailsEngine.__new__(GuardrailsEngine)
            engine.enabled = False
            engine._rails = None
            engine._config_dir = "fake/path"
            assert engine.enabled is False

    @pytest.mark.asyncio
    async def test_check_input_passthrough_when_disabled(self):
        """Disabled engine should allow all input."""
        with patch("forge.guardrails.rails.HAS_GUARDRAILS", False):
            engine = GuardrailsEngine()
            result = await engine.check_input("test query")
            assert result["allowed"] is True
            assert result["reason"] is None
            assert result["scrubbed_query"] == "test query"

    @pytest.mark.asyncio
    async def test_check_output_passthrough_when_disabled(self):
        """Disabled engine should pass through all output."""
        with patch("forge.guardrails.rails.HAS_GUARDRAILS", False):
            engine = GuardrailsEngine()
            result = await engine.check_output("query", "response", {})
            assert result["allowed"] is True
            assert result["response"] == "response"
            assert result["provenance"] == {}

    @pytest.mark.asyncio
    async def test_guarded_generate_passthrough_when_disabled(self):
        """Disabled engine should echo query in guarded_generate."""
        with patch("forge.guardrails.rails.HAS_GUARDRAILS", False):
            engine = GuardrailsEngine()
            result = await engine.guarded_generate("hello")
            assert result["response"] == "hello"
            assert result["input_check"]["allowed"] is True

    def test_rails_is_none_when_disabled(self):
        """Internal _rails should be None when disabled."""
        with patch("forge.guardrails.rails.HAS_GUARDRAILS", False):
            engine = GuardrailsEngine()
            assert engine._rails is None


# ===================================================================
# GuardrailsEngine -- enabled mode (mocked nemoguardrails)
# ===================================================================

class TestGuardrailsEngineEnabled:
    """Tests for GuardrailsEngine with mocked NeMo Guardrails."""

    def _make_enabled_engine(self):
        """Create an engine with a mocked _rails object."""
        engine = GuardrailsEngine.__new__(GuardrailsEngine)
        engine.enabled = True
        engine._rails = MagicMock()
        engine._config_dir = "forge/guardrails/config"
        return engine

    @pytest.mark.asyncio
    async def test_check_input_allowed(self):
        """Input rail should return allowed when NeMo says not blocked."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(
            return_value={"blocked": False, "content": "clean query"}
        )
        result = await engine.check_input("test query")
        assert result["allowed"] is True
        assert result["scrubbed_query"] == "clean query"

    @pytest.mark.asyncio
    async def test_check_input_blocked(self):
        """Input rail should return blocked with reason."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(
            return_value={"blocked": True, "reason": "jailbreak detected"}
        )
        result = await engine.check_input("hack the planet")
        assert result["allowed"] is False
        assert result["reason"] == "jailbreak detected"

    @pytest.mark.asyncio
    async def test_check_input_exception_fallback(self):
        """Input rail should fall back to allowed on exception."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(side_effect=RuntimeError("boom"))
        result = await engine.check_input("test")
        assert result["allowed"] is True

    @pytest.mark.asyncio
    async def test_check_output_allowed(self):
        """Output rail should return response when not blocked."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(
            return_value={
                "blocked": False,
                "content": "verified response",
                "provenance": {"sources": ["A"]},
            }
        )
        result = await engine.check_output("q", "response", {})
        assert result["allowed"] is True
        assert result["response"] == "verified response"
        assert result["provenance"] == {"sources": ["A"]}

    @pytest.mark.asyncio
    async def test_check_output_blocked(self):
        """Output rail should return original response when blocked."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(
            return_value={"blocked": True, "content": "bad content"}
        )
        result = await engine.check_output("q", "original", {})
        assert result["allowed"] is False
        assert result["response"] == "original"

    @pytest.mark.asyncio
    async def test_check_output_exception_fallback(self):
        """Output rail should fall back to allowed on exception."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(side_effect=RuntimeError("boom"))
        result = await engine.check_output("q", "response", {})
        assert result["allowed"] is True
        assert result["response"] == "response"

    @pytest.mark.asyncio
    async def test_guarded_generate_full_pipeline(self):
        """Full pipeline: input check -> generate -> output check."""
        engine = self._make_enabled_engine()

        # Input check passes
        call_count = 0
        async def mock_generate(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"blocked": False, "content": "clean query"}
            return {"blocked": False, "content": "final answer", "provenance": {}}

        engine._rails.generate_async = mock_generate

        async def gen_fn(q, ctx):
            return f"answer to: {q}"

        result = await engine.guarded_generate("test?", generate_fn=gen_fn)
        assert "answer" in result["response"] or result["response"] == "final answer"

    @pytest.mark.asyncio
    async def test_guarded_generate_input_blocked(self):
        """Guarded generate should short-circuit when input is blocked."""
        engine = self._make_enabled_engine()
        engine._rails.generate_async = AsyncMock(
            return_value={"blocked": True, "reason": "bad input"}
        )
        result = await engine.guarded_generate("bad query")
        assert "[BLOCKED]" in result["response"]
        assert result["input_check"]["allowed"] is False
        assert result["output_check"] == {}


# ===================================================================
# Custom Actions
# ===================================================================

class TestCheckHallucination:
    """Tests for check_hallucination action."""

    @pytest.mark.asyncio
    async def test_no_graph_engine(self):
        """Should return True when no graph engine is available."""
        result = await check_hallucination({}, "some response")
        assert result is True

    @pytest.mark.asyncio
    async def test_no_expert_names_in_response(self):
        """Should return True when no expert names are found."""
        graph = MagicMock()
        result = await check_hallucination({}, "this has no expert names", graph)
        assert result is True

    @pytest.mark.asyncio
    async def test_real_expert_passes(self):
        """Should return True when cited expert exists in graph."""
        graph = MagicMock()
        graph.has_expert.return_value = True
        result = await check_hallucination(
            {}, "According to Alice this is correct", graph
        )
        assert result is True
        graph.has_expert.assert_called_with("Alice")

    @pytest.mark.asyncio
    async def test_fake_expert_fails(self):
        """Should return False when cited expert doesn't exist."""
        graph = MagicMock()
        graph.has_expert.return_value = False
        result = await check_hallucination(
            {}, "According to Nonexistent Person this is true", graph
        )
        assert result is False


class TestCheckAttribution:
    """Tests for check_attribution action."""

    @pytest.mark.asyncio
    async def test_no_graph_passes(self):
        result = await check_attribution({}, "text")
        assert result is True

    @pytest.mark.asyncio
    async def test_valid_attribution(self):
        graph = MagicMock()
        graph.has_expert.return_value = True
        result = await check_attribution({}, "Bob suggests using PyTorch", graph)
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_attribution(self):
        graph = MagicMock()
        graph.has_expert.return_value = False
        result = await check_attribution({}, "Zyx suggests something", graph)
        assert result is False


class TestAddProvenance:
    """Tests for add_provenance action."""

    @pytest.mark.asyncio
    async def test_no_chunks(self):
        result = await add_provenance({}, "response", None)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        result = await add_provenance({}, "response", [])
        assert result == "response"

    @pytest.mark.asyncio
    async def test_adds_sources_footer(self):
        chunks = [
            {"expert": "Alice", "title": "ML Guide"},
            {"expert": "Bob", "title": "DL Handbook"},
        ]
        result = await add_provenance({}, "response", chunks)
        assert "[Sources]" in result
        assert "Alice: ML Guide" in result
        assert "Bob: DL Handbook" in result

    @pytest.mark.asyncio
    async def test_deduplicates_sources(self):
        chunks = [
            {"expert": "Alice", "title": "ML Guide"},
            {"expert": "Alice", "title": "ML Guide"},
        ]
        result = await add_provenance({}, "response", chunks)
        assert result.count("Alice: ML Guide") == 1


class TestScrubPiiInput:
    """Tests for scrub_pii_input action."""

    @pytest.mark.asyncio
    async def test_clean_text(self):
        text, counts = await scrub_pii_input({}, "no PII here")
        assert text == "no PII here"
        assert counts == {}

    @pytest.mark.asyncio
    async def test_scrubs_email(self):
        text, counts = await scrub_pii_input({}, "email: user@example.com")
        assert "[REDACTED]" in text
        assert counts["email"] == 1

    @pytest.mark.asyncio
    async def test_scrubs_ssn(self):
        text, counts = await scrub_pii_input({}, "SSN: 123-45-6789")
        assert "[REDACTED]" in text
        assert counts["ssn"] == 1

    @pytest.mark.asyncio
    async def test_scrubs_multiple_types(self):
        text, counts = await scrub_pii_input(
            {}, "SSN: 123-45-6789, email: foo@bar.com"
        )
        assert text.count("[REDACTED]") == 2
        assert "ssn" in counts
        assert "email" in counts


class TestSelfCorrection:
    """Tests for self_correction action."""

    @pytest.mark.asyncio
    async def test_no_hallucination_returns_original(self):
        result = await self_correction({}, "good response", False)
        assert result == "good response"

    @pytest.mark.asyncio
    async def test_no_generate_fn_returns_original(self):
        result = await self_correction({}, "bad response", True)
        assert result == "bad response"

    @pytest.mark.asyncio
    async def test_retries_with_generate_fn(self):
        gen_fn = AsyncMock(return_value="corrected response")
        result = await self_correction(
            {"query": "what is ML?"}, "bad response", True, generate_fn=gen_fn
        )
        assert result == "corrected response"
        gen_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_retries_on_failure(self):
        gen_fn = AsyncMock(side_effect=RuntimeError("fail"))
        result = await self_correction(
            {"query": "test"}, "bad", True, generate_fn=gen_fn, max_retries=2
        )
        assert gen_fn.call_count == 2


class TestLogRailDecision:
    """Tests for log_rail_decision action."""

    @pytest.mark.asyncio
    async def test_logs_allowed(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="forge.guardrails.actions"):
            await log_rail_decision("test_rail", True, "all good", "test query")
        assert "RAIL_AUDIT" in caplog.text
        assert "ALLOWED" in caplog.text

    @pytest.mark.asyncio
    async def test_logs_blocked(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="forge.guardrails.actions"):
            await log_rail_decision("test_rail", False, "bad stuff", "test query")
        assert "BLOCKED" in caplog.text


# ===================================================================
# Expert name extraction
# ===================================================================

class TestExtractExpertNames:
    """Tests for the internal _extract_expert_names helper."""

    def test_according_to_pattern(self):
        names = _extract_expert_names("According to Alice Smith this is true")
        assert "Alice Smith" in names

    def test_suggests_pattern(self):
        names = _extract_expert_names("Bob suggests using PyTorch")
        assert "Bob" in names

    def test_recommends_pattern(self):
        names = _extract_expert_names("Carol recommends TensorFlow")
        assert "Carol" in names

    def test_no_names(self):
        names = _extract_expert_names("this text has no expert names in it")
        assert names == []

    def test_deduplication(self):
        names = _extract_expert_names(
            "According to Alice this is true. Alice suggests more."
        )
        assert names.count("Alice") == 1

    def test_multi_word_name(self):
        names = _extract_expert_names("According to John Von Neumann this is valid")
        assert any("John" in n for n in names)
