"""Layered context engine -- graph-enriched, budget-aware retrieval.

Builds a multi-layer context window for the LLM:
    Layer 0 -- System identity / persona
    Layer 1 -- Graph context (PageRank rankings + contradictions)
    Layer 2 -- Compressed vector-search chunks
    Layer 3 -- Deep search (uncompressed, highest-relevance chunks)
"""
import logging
from typing import Optional

from pydantic import BaseModel, Field

from forge.layers.compressor import compress
from forge.layers.ranker import allocate_budget, estimate_tokens, truncate_to_budget

logger = logging.getLogger(__name__)

# Default system identity for layer 0
_DEFAULT_IDENTITY = (
    "You are NeuralForge, an expert knowledge assistant. "
    "Answer using only the context provided below. "
    "Cite experts by name when possible."
)


class LayeredContext(BaseModel):
    """Assembled context across all four layers."""

    query: str = ""
    layer_0: str = ""   # Identity / system prompt
    layer_1: str = ""   # Graph context (cuGraph rankings + contradictions)
    layer_2: str = ""   # Compressed chunks from vector search
    layer_3: str = ""   # Deep search uncompressed chunks
    total_tokens: int = 0
    layers_used: list[int] = Field(default_factory=list)
    experts_referenced: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        """Concatenate all layers into a single context string."""
        parts: list[str] = []
        if self.layer_0:
            parts.append(self.layer_0)
        if self.layer_1:
            parts.append(f"[Graph Context]\n{self.layer_1}")
        if self.layer_2:
            parts.append(f"[Retrieved Context]\n{self.layer_2}")
        if self.layer_3:
            parts.append(f"[Deep Search]\n{self.layer_3}")
        return "\n\n".join(parts)


async def get_context(
    query: str,
    max_tokens: int = 4000,
    max_layer: int = 3,
    expert_filter: Optional[str] = None,
    compress_chunks: bool = True,
    identity: Optional[str] = None,
    graph_engine=None,
    search_fn=None,
    deep_search_fn=None,
) -> LayeredContext:
    """Build a layered context window for the given query.

    Parameters
    ----------
    query:
        The user query.
    max_tokens:
        Maximum total token budget across all layers.
    max_layer:
        Highest layer to populate (0-3).
    expert_filter:
        Optional expert name to filter search results.
    compress_chunks:
        Whether to apply AAAK compression to layer 2 chunks.
    identity:
        Custom system identity for layer 0. Defaults to the
        built-in NeuralForge identity.
    graph_engine:
        A ``GraphEngine`` instance for layer 1 (PageRank, contradictions).
    search_fn:
        Async callable ``(query, limit, expert) -> list[dict]``
        for layer 2 vector search.
    deep_search_fn:
        Async callable ``(query, limit, expert) -> list[dict]``
        for layer 3 deep search.

    Returns
    -------
    LayeredContext
        Assembled context with budget-aware content in each layer.
    """
    ctx = LayeredContext(query=query)
    experts_seen: set[str] = set()
    layers_used: list[int] = []

    # ----------------------------------------------------------
    # Layer 0: Identity
    # ----------------------------------------------------------
    if max_layer >= 0:
        ctx.layer_0 = identity or _DEFAULT_IDENTITY
        layers_used.append(0)

    # Compute actual token counts for known layers
    identity_tokens = estimate_tokens(ctx.layer_0)

    # ----------------------------------------------------------
    # Layer 1: Graph context
    # ----------------------------------------------------------
    graph_text = ""
    if max_layer >= 1 and graph_engine is not None:
        graph_text = _build_graph_context(query, graph_engine, expert_filter)
        if graph_text:
            layers_used.append(1)
            # Extract expert names from graph context
            for ranking in _get_expert_rankings(query, graph_engine, expert_filter):
                experts_seen.add(ranking)

    graph_tokens = estimate_tokens(graph_text)

    # ----------------------------------------------------------
    # Allocate budget
    # ----------------------------------------------------------
    budget = allocate_budget(
        max_tokens=max_tokens,
        max_layer=max_layer,
        identity_tokens=identity_tokens,
        graph_tokens=graph_tokens,
    )

    # Apply budget to layer 0
    if 0 in budget:
        ctx.layer_0 = truncate_to_budget(ctx.layer_0, budget[0])

    # Apply budget to layer 1
    if 1 in budget and graph_text:
        ctx.layer_1 = truncate_to_budget(graph_text, budget[1])

    # ----------------------------------------------------------
    # Layer 2: Compressed chunks
    # ----------------------------------------------------------
    if max_layer >= 2 and search_fn is not None:
        layer2_budget = budget.get(2, 0)
        if layer2_budget > 0:
            try:
                chunks = await search_fn(query, limit=10, expert=expert_filter)
                if chunks:
                    chunk_texts = _format_chunks(chunks)
                    for c in chunks:
                        expert = c.get("expert", "")
                        if expert:
                            experts_seen.add(expert)

                    combined = "\n\n".join(chunk_texts)
                    if compress_chunks:
                        combined = compress(combined, level=1)

                    ctx.layer_2 = truncate_to_budget(combined, layer2_budget)
                    layers_used.append(2)
            except Exception as exc:
                logger.error("Layer 2 search failed: %s", exc)

    # ----------------------------------------------------------
    # Layer 3: Deep search (uncompressed)
    # ----------------------------------------------------------
    if max_layer >= 3 and deep_search_fn is not None:
        layer3_budget = budget.get(3, 0)
        if layer3_budget > 0:
            try:
                deep_chunks = await deep_search_fn(query, limit=5, expert=expert_filter)
                if deep_chunks:
                    chunk_texts = _format_chunks(deep_chunks)
                    for c in deep_chunks:
                        expert = c.get("expert", "")
                        if expert:
                            experts_seen.add(expert)

                    combined = "\n\n".join(chunk_texts)
                    ctx.layer_3 = truncate_to_budget(combined, layer3_budget)
                    layers_used.append(3)
            except Exception as exc:
                logger.error("Layer 3 deep search failed: %s", exc)

    # ----------------------------------------------------------
    # Finalize
    # ----------------------------------------------------------
    ctx.layers_used = layers_used
    ctx.experts_referenced = sorted(experts_seen)
    ctx.total_tokens = (
        estimate_tokens(ctx.layer_0)
        + estimate_tokens(ctx.layer_1)
        + estimate_tokens(ctx.layer_2)
        + estimate_tokens(ctx.layer_3)
    )

    return ctx


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _build_graph_context(
    query: str, graph_engine, expert_filter: Optional[str] = None
) -> str:
    """Build graph context text from PageRank and contradictions."""
    parts: list[str] = []

    # Expert rankings
    try:
        rankings = graph_engine.expert_authority(query)
        if expert_filter:
            rankings = [
                r for r in rankings
                if r.expert_name.lower() == expert_filter.lower()
            ]
        if rankings:
            lines = ["Expert Authority Rankings:"]
            for r in rankings[:5]:
                lines.append(
                    f"  - {r.expert_name}: score={r.score:.4f} "
                    f"(edges={r.edge_count})"
                )
            parts.append("\n".join(lines))
    except Exception as exc:
        logger.warning("Failed to get expert rankings: %s", exc)

    # Contradictions
    try:
        contradictions = graph_engine.find_contradictions(topic=query)
        if contradictions:
            lines = ["Known Contradictions:"]
            for c in contradictions[:3]:
                lines.append(f"  - {c.explanation}")
            parts.append("\n".join(lines))
    except Exception as exc:
        logger.warning("Failed to get contradictions: %s", exc)

    return "\n\n".join(parts)


def _get_expert_rankings(
    query: str, graph_engine, expert_filter: Optional[str] = None
) -> list[str]:
    """Extract expert names from authority rankings."""
    try:
        rankings = graph_engine.expert_authority(query)
        if expert_filter:
            rankings = [
                r for r in rankings
                if r.expert_name.lower() == expert_filter.lower()
            ]
        return [r.expert_name for r in rankings[:5]]
    except Exception:
        return []


def _format_chunks(chunks: list[dict]) -> list[str]:
    """Format search result chunks into readable text blocks."""
    formatted: list[str] = []
    for chunk in chunks:
        expert = chunk.get("expert", "unknown")
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        score = chunk.get("score", 0.0)

        header = f"[{expert}]"
        if title:
            header += f" {title}"
        header += f" (score: {score:.3f})"

        formatted.append(f"{header}\n{text}")

    return formatted
