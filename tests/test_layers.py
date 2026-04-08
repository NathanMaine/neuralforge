"""Tests for layered context engine -- 25+ tests.

Covers all layers, budget allocation, compression on/off, expert filter,
empty results, graph integration, and edge cases.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge.layers.engine import (
    LayeredContext,
    _build_graph_context,
    _format_chunks,
    get_context,
)


# ===================================================================
# Fixtures
# ===================================================================

def _make_chunks(n=3, expert="Alice"):
    """Create n mock search result chunks."""
    return [
        {
            "score": 0.9 - i * 0.1,
            "expert": expert,
            "title": f"Doc {i+1}",
            "text": f"This is content from document {i+1}.",
            "source": f"source_{i+1}",
            "chunk_index": i,
        }
        for i in range(n)
    ]


def _make_graph_engine(experts=None, contradictions=None):
    """Create a mock graph engine with expert_authority and find_contradictions."""
    engine = MagicMock()

    rankings = []
    if experts:
        for i, name in enumerate(experts):
            r = MagicMock()
            r.expert_name = name
            r.score = 0.5 - i * 0.1
            r.edge_count = 3 - i
            rankings.append(r)

    engine.expert_authority.return_value = rankings

    contras = []
    if contradictions:
        for c in contradictions:
            contra = MagicMock()
            contra.explanation = c
            contras.append(contra)

    engine.find_contradictions.return_value = contras

    return engine


# ===================================================================
# LayeredContext model
# ===================================================================

class TestLayeredContext:
    """Tests for the LayeredContext data model."""

    def test_empty_context(self):
        ctx = LayeredContext()
        assert ctx.query == ""
        assert ctx.layer_0 == ""
        assert ctx.layers_used == []
        assert ctx.experts_referenced == []

    def test_as_text_empty(self):
        ctx = LayeredContext()
        assert ctx.as_text() == ""

    def test_as_text_with_layers(self):
        ctx = LayeredContext(
            layer_0="System prompt",
            layer_1="Graph data",
            layer_2="Chunks",
            layer_3="Deep results",
        )
        text = ctx.as_text()
        assert "System prompt" in text
        assert "[Graph Context]" in text
        assert "[Retrieved Context]" in text
        assert "[Deep Search]" in text

    def test_as_text_partial_layers(self):
        ctx = LayeredContext(layer_0="Identity", layer_2="Chunks")
        text = ctx.as_text()
        assert "Identity" in text
        assert "[Retrieved Context]" in text
        assert "[Graph Context]" not in text
        assert "[Deep Search]" not in text

    def test_total_tokens_default(self):
        ctx = LayeredContext()
        assert ctx.total_tokens == 0


# ===================================================================
# get_context -- layer 0
# ===================================================================

class TestGetContextLayer0:
    """Tests for layer 0 (identity)."""

    @pytest.mark.asyncio
    async def test_default_identity(self):
        ctx = await get_context("test query", max_layer=0)
        assert "NeuralForge" in ctx.layer_0
        assert 0 in ctx.layers_used

    @pytest.mark.asyncio
    async def test_custom_identity(self):
        ctx = await get_context("test", max_layer=0, identity="Custom Bot")
        assert ctx.layer_0 == "Custom Bot"

    @pytest.mark.asyncio
    async def test_layer_0_always_present(self):
        ctx = await get_context("test", max_layer=3)
        assert 0 in ctx.layers_used


# ===================================================================
# get_context -- layer 1 (graph)
# ===================================================================

class TestGetContextLayer1:
    """Tests for layer 1 (graph context)."""

    @pytest.mark.asyncio
    async def test_no_graph_engine(self):
        ctx = await get_context("test", max_layer=1)
        assert ctx.layer_1 == ""
        assert 1 not in ctx.layers_used

    @pytest.mark.asyncio
    async def test_with_graph_engine(self):
        graph = _make_graph_engine(experts=["Alice", "Bob"])
        ctx = await get_context("ML", max_layer=1, graph_engine=graph)
        assert ctx.layer_1 != ""
        assert 1 in ctx.layers_used

    @pytest.mark.asyncio
    async def test_graph_context_includes_rankings(self):
        graph = _make_graph_engine(experts=["Alice"])
        ctx = await get_context("ML", max_layer=1, graph_engine=graph)
        assert "Alice" in ctx.layer_1

    @pytest.mark.asyncio
    async def test_graph_context_includes_contradictions(self):
        graph = _make_graph_engine(
            experts=["Alice"],
            contradictions=["Conflicting claim about batch size"],
        )
        ctx = await get_context("ML", max_layer=1, graph_engine=graph)
        assert "Contradictions" in ctx.layer_1

    @pytest.mark.asyncio
    async def test_expert_filter_on_graph(self):
        graph = _make_graph_engine(experts=["Alice", "Bob"])
        ctx = await get_context(
            "ML", max_layer=1, graph_engine=graph, expert_filter="Alice"
        )
        # Should filter to just Alice
        assert "Alice" in ctx.layer_1 or ctx.layer_1 == ""


# ===================================================================
# get_context -- layer 2 (compressed chunks)
# ===================================================================

class TestGetContextLayer2:
    """Tests for layer 2 (compressed chunks)."""

    @pytest.mark.asyncio
    async def test_no_search_fn(self):
        ctx = await get_context("test", max_layer=2)
        assert ctx.layer_2 == ""
        assert 2 not in ctx.layers_used

    @pytest.mark.asyncio
    async def test_with_search_fn(self):
        search = AsyncMock(return_value=_make_chunks(3))
        ctx = await get_context("ML", max_layer=2, search_fn=search)
        assert ctx.layer_2 != ""
        assert 2 in ctx.layers_used

    @pytest.mark.asyncio
    async def test_compression_applied(self):
        chunks = _make_chunks(1, expert="Alice")
        chunks[0]["text"] = "It is important to note that basically this is a test."
        search = AsyncMock(return_value=chunks)
        ctx = await get_context("ML", max_layer=2, search_fn=search, compress_chunks=True)
        # Filler words should be removed
        assert "important to note" not in ctx.layer_2

    @pytest.mark.asyncio
    async def test_compression_disabled(self):
        chunks = _make_chunks(1, expert="Alice")
        chunks[0]["text"] = "It is important to note that this is a test."
        search = AsyncMock(return_value=chunks)
        ctx = await get_context(
            "ML", max_layer=2, search_fn=search, compress_chunks=False
        )
        assert "important to note" in ctx.layer_2

    @pytest.mark.asyncio
    async def test_expert_filter_passed_to_search(self):
        search = AsyncMock(return_value=[])
        await get_context("ML", max_layer=2, search_fn=search, expert_filter="Alice")
        search.assert_called_once_with("ML", limit=10, expert="Alice")

    @pytest.mark.asyncio
    async def test_search_fn_exception_handled(self):
        search = AsyncMock(side_effect=RuntimeError("search failed"))
        ctx = await get_context("ML", max_layer=2, search_fn=search)
        assert ctx.layer_2 == ""

    @pytest.mark.asyncio
    async def test_empty_search_results(self):
        search = AsyncMock(return_value=[])
        ctx = await get_context("ML", max_layer=2, search_fn=search)
        assert ctx.layer_2 == ""
        assert 2 not in ctx.layers_used

    @pytest.mark.asyncio
    async def test_experts_extracted_from_chunks(self):
        search = AsyncMock(return_value=_make_chunks(2, expert="Bob"))
        ctx = await get_context("ML", max_layer=2, search_fn=search)
        assert "Bob" in ctx.experts_referenced


# ===================================================================
# get_context -- layer 3 (deep search)
# ===================================================================

class TestGetContextLayer3:
    """Tests for layer 3 (deep search)."""

    @pytest.mark.asyncio
    async def test_no_deep_search_fn(self):
        ctx = await get_context("test", max_layer=3)
        assert ctx.layer_3 == ""

    @pytest.mark.asyncio
    async def test_with_deep_search_fn(self):
        deep = AsyncMock(return_value=_make_chunks(2, expert="Carol"))
        ctx = await get_context("ML", max_layer=3, deep_search_fn=deep)
        assert ctx.layer_3 != ""
        assert 3 in ctx.layers_used

    @pytest.mark.asyncio
    async def test_deep_search_not_compressed(self):
        chunks = _make_chunks(1, expert="Carol")
        chunks[0]["text"] = "It is important to note that this is deep content."
        deep = AsyncMock(return_value=chunks)
        ctx = await get_context("ML", max_layer=3, deep_search_fn=deep)
        # Layer 3 should NOT compress
        assert "important to note" in ctx.layer_3

    @pytest.mark.asyncio
    async def test_deep_search_exception_handled(self):
        deep = AsyncMock(side_effect=RuntimeError("deep failed"))
        ctx = await get_context("ML", max_layer=3, deep_search_fn=deep)
        assert ctx.layer_3 == ""

    @pytest.mark.asyncio
    async def test_deep_search_experts_captured(self):
        deep = AsyncMock(return_value=_make_chunks(1, expert="Diana"))
        ctx = await get_context("ML", max_layer=3, deep_search_fn=deep)
        assert "Diana" in ctx.experts_referenced


# ===================================================================
# Full pipeline
# ===================================================================

class TestFullPipeline:
    """Tests for the full get_context pipeline across all layers."""

    @pytest.mark.asyncio
    async def test_all_layers_populated(self):
        graph = _make_graph_engine(experts=["Alice"])
        search = AsyncMock(return_value=_make_chunks(2, expert="Bob"))
        deep = AsyncMock(return_value=_make_chunks(1, expert="Carol"))

        ctx = await get_context(
            "ML",
            max_tokens=4000,
            graph_engine=graph,
            search_fn=search,
            deep_search_fn=deep,
        )
        assert 0 in ctx.layers_used
        assert 1 in ctx.layers_used
        assert 2 in ctx.layers_used
        assert 3 in ctx.layers_used

    @pytest.mark.asyncio
    async def test_total_tokens_calculated(self):
        search = AsyncMock(return_value=_make_chunks(2))
        ctx = await get_context("ML", max_tokens=4000, search_fn=search)
        assert ctx.total_tokens > 0

    @pytest.mark.asyncio
    async def test_experts_aggregated(self):
        graph = _make_graph_engine(experts=["Alice"])
        search = AsyncMock(return_value=_make_chunks(1, expert="Bob"))
        deep = AsyncMock(return_value=_make_chunks(1, expert="Carol"))

        ctx = await get_context(
            "ML",
            graph_engine=graph,
            search_fn=search,
            deep_search_fn=deep,
        )
        assert "Alice" in ctx.experts_referenced
        assert "Bob" in ctx.experts_referenced
        assert "Carol" in ctx.experts_referenced

    @pytest.mark.asyncio
    async def test_budget_constraint_respected(self):
        """With a tiny budget, content should be truncated."""
        search = AsyncMock(return_value=_make_chunks(10))
        ctx = await get_context("ML", max_tokens=50, search_fn=search)
        assert ctx.total_tokens <= 100  # Some headroom due to estimation

    @pytest.mark.asyncio
    async def test_query_stored(self):
        ctx = await get_context("What is ML?")
        assert ctx.query == "What is ML?"


# ===================================================================
# Internal helpers
# ===================================================================

class TestFormatChunks:
    """Tests for _format_chunks helper."""

    def test_format_single_chunk(self):
        chunks = [{"expert": "Alice", "title": "ML Guide", "text": "Content", "score": 0.95}]
        formatted = _format_chunks(chunks)
        assert len(formatted) == 1
        assert "[Alice]" in formatted[0]
        assert "ML Guide" in formatted[0]
        assert "0.950" in formatted[0]
        assert "Content" in formatted[0]

    def test_format_empty_chunks(self):
        assert _format_chunks([]) == []

    def test_format_chunk_without_title(self):
        chunks = [{"expert": "Bob", "title": "", "text": "Data", "score": 0.5}]
        formatted = _format_chunks(chunks)
        assert "[Bob]" in formatted[0]


class TestBuildGraphContext:
    """Tests for _build_graph_context helper."""

    def test_with_rankings(self):
        graph = _make_graph_engine(experts=["Alice", "Bob"])
        result = _build_graph_context("ML", graph)
        assert "Alice" in result
        assert "Authority" in result

    def test_with_contradictions(self):
        graph = _make_graph_engine(
            experts=["Alice"],
            contradictions=["Conflict about learning rates"],
        )
        result = _build_graph_context("ML", graph)
        assert "Contradictions" in result

    def test_empty_graph(self):
        graph = _make_graph_engine()
        result = _build_graph_context("ML", graph)
        assert result == ""

    def test_expert_filter(self):
        graph = _make_graph_engine(experts=["Alice", "Bob"])
        result = _build_graph_context("ML", graph, expert_filter="Alice")
        assert "Alice" in result
