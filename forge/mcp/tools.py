"""MCP tool definitions and dispatch table.

17+ tools for search, graph queries, context retrieval, and ingestion
status.  Each tool is an async function that returns a JSON-serializable
result.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _get_engine():
    from forge.api.main import get_graph_engine
    return get_graph_engine()


def _get_guardrails():
    from forge.api.main import get_guardrails_engine
    return get_guardrails_engine()


# -----------------------------------------------------------------------
# Tool implementations
# -----------------------------------------------------------------------

async def search_experts(query: str, limit: int = 10) -> str:
    """Search for expert knowledge across the knowledge base."""
    from forge.core import embeddings, qdrant_client
    vec = await embeddings.get_embedding(query)
    if vec is None:
        return json.dumps({"results": [], "error": "Embedding failed"})
    results = qdrant_client.search_vectors(vec, limit=limit)
    return json.dumps({"results": results, "count": len(results)})


async def list_experts() -> str:
    """List all known experts in the knowledge graph."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"experts": [], "error": "Engine not initialized"})
    experts = engine.get_all_experts()
    return json.dumps({
        "experts": [{"id": e.id, "name": e.name} for e in experts],
        "count": len(experts),
    })


async def stats() -> str:
    """Get graph and knowledge base statistics."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    graph_stats = engine.store.get_stats()
    return json.dumps(graph_stats.model_dump())


async def get_context(
    query: str,
    max_tokens: int = 4000,
    expert: Optional[str] = None,
) -> str:
    """Get layered context for a query (for LLM consumption)."""
    from forge.core import embeddings, qdrant_client
    from forge.layers.engine import get_context as _get_context

    engine = _get_engine()

    async def _search_fn(q, limit=10, expert=None):
        vec = await embeddings.get_embedding(q)
        if vec is None:
            return []
        return qdrant_client.search_vectors(vec, limit=limit, expert=expert)

    ctx = await _get_context(
        query=query,
        max_tokens=max_tokens,
        expert_filter=expert,
        graph_engine=engine,
        search_fn=_search_fn,
    )
    return json.dumps({
        "context": ctx.as_text(),
        "total_tokens": ctx.total_tokens,
        "layers_used": ctx.layers_used,
        "experts_referenced": ctx.experts_referenced,
    })


async def add_node(
    name: str,
    node_type: str,
    description: Optional[str] = None,
) -> str:
    """Add a node to the knowledge graph."""
    from forge.graph.models import NodeType
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    try:
        nt = NodeType(node_type)
    except ValueError:
        return json.dumps({"error": f"Invalid node_type: {node_type}"})
    node = engine.add_node(name=name, node_type=nt, description=description)
    engine.store.save()
    return json.dumps({"node": node.model_dump()})


async def add_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float = 1.0,
    confidence: float = 1.0,
) -> str:
    """Add an edge to the knowledge graph."""
    from forge.graph.models import EdgeType
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    try:
        et = EdgeType(edge_type)
    except ValueError:
        return json.dumps({"error": f"Invalid edge_type: {edge_type}"})
    edge = engine.add_edge(
        source_id=source_id,
        target_id=target_id,
        edge_type=et,
        weight=weight,
        confidence=confidence,
    )
    engine.store.save()
    return json.dumps({"edge": edge.model_dump()})


async def update_edge(
    edge_id: str,
    weight: Optional[float] = None,
    confidence: Optional[float] = None,
) -> str:
    """Update an existing edge."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    fields = {}
    if weight is not None:
        fields["weight"] = weight
    if confidence is not None:
        fields["confidence"] = confidence
    edge = engine.store.update_edge(edge_id, **fields)
    if edge is None:
        return json.dumps({"error": "Edge not found"})
    engine.store.save()
    return json.dumps({"edge": edge.model_dump()})


async def expire_edge(edge_id: str) -> str:
    """Mark an edge as expired (soft delete)."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    edge = engine.store.expire_edge(edge_id)
    if edge is None:
        return json.dumps({"error": "Edge not found"})
    engine.store.save()
    return json.dumps({"edge": edge.model_dump()})


async def query_graph(
    query: str,
    node_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Search nodes in the graph by name."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    nodes = engine.store.search_nodes(query, node_type=node_type, limit=limit)
    return json.dumps({
        "nodes": [n.model_dump() for n in nodes],
        "count": len(nodes),
    })


async def traverse_graph(
    node_id: str,
    depth: int = 2,
) -> str:
    """BFS traversal from a node."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    result = engine.traverse(node_id, depth=depth)
    return json.dumps(result.model_dump())


async def find_contradictions(topic: Optional[str] = None) -> str:
    """Find contradictions in the knowledge graph."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    contradictions = engine.find_contradictions(topic=topic)
    return json.dumps({
        "contradictions": [c.model_dump() for c in contradictions],
        "count": len(contradictions),
    })


async def find_experts_for(topic: str) -> str:
    """Find experts ranked by authority on a topic."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    rankings = engine.expert_authority(topic)
    return json.dumps({
        "topic": topic,
        "rankings": [r.model_dump() for r in rankings],
    })


async def graph_timeline(since: str, topic: Optional[str] = None) -> str:
    """Get edges created since a given date."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    edges = engine.find_changes_since(since, topic=topic)
    return json.dumps({
        "since": since,
        "edges": [e.model_dump() for e in edges],
        "count": len(edges),
    })


async def pagerank() -> str:
    """Compute PageRank scores."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    scores = engine.pagerank()
    enriched = []
    for node_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        node = engine.store.get_node(node_id)
        enriched.append({
            "node_id": node_id,
            "name": node.name if node else "unknown",
            "score": score,
        })
    return json.dumps({"pagerank": enriched})


async def communities() -> str:
    """Detect communities using Louvain algorithm."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    comms = engine.find_communities()
    groups: dict[int, list[str]] = {}
    for node_id, cid in comms.items():
        groups.setdefault(cid, []).append(node_id)
    return json.dumps({"communities": groups, "num_communities": len(groups)})


async def shortest_path(source_id: str, target_id: str) -> str:
    """Find shortest path between two nodes."""
    engine = _get_engine()
    if engine is None:
        return json.dumps({"error": "Engine not initialized"})
    path = engine.shortest_path(source_id, target_id)
    return json.dumps({"path": path, "length": max(0, len(path) - 1), "found": len(path) > 0})


async def auto_capture_status() -> str:
    """Get auto-capture system status."""
    from forge.api.routes.ingest import _auto_capture_stats
    return json.dumps(_auto_capture_stats)


# -----------------------------------------------------------------------
# Tool definitions (MCP schema)
# -----------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "search_experts",
        "description": "Search for expert knowledge across the knowledge base",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_experts",
        "description": "List all known experts in the knowledge graph",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "stats",
        "description": "Get graph and knowledge base statistics",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_context",
        "description": "Get layered RAG context for a query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query to get context for"},
                "max_tokens": {"type": "integer", "default": 4000},
                "expert": {"type": "string", "description": "Filter by expert name"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_node",
        "description": "Add a node to the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "node_type": {"type": "string", "enum": ["expert", "concept", "technique", "tool", "dataset", "model", "paper", "institution"]},
                "description": {"type": "string"},
            },
            "required": ["name", "node_type"],
        },
    },
    {
        "name": "add_edge",
        "description": "Add an edge between two nodes in the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
                "edge_type": {"type": "string"},
                "weight": {"type": "number", "default": 1.0},
                "confidence": {"type": "number", "default": 1.0},
            },
            "required": ["source_id", "target_id", "edge_type"],
        },
    },
    {
        "name": "update_edge",
        "description": "Update an existing edge's weight or confidence",
        "inputSchema": {
            "type": "object",
            "properties": {
                "edge_id": {"type": "string"},
                "weight": {"type": "number"},
                "confidence": {"type": "number"},
            },
            "required": ["edge_id"],
        },
    },
    {
        "name": "expire_edge",
        "description": "Mark an edge as expired (soft delete)",
        "inputSchema": {
            "type": "object",
            "properties": {"edge_id": {"type": "string"}},
            "required": ["edge_id"],
        },
    },
    {
        "name": "query_graph",
        "description": "Search nodes in the graph by name",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "node_type": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "traverse_graph",
        "description": "BFS traversal from a node up to N hops",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string"},
                "depth": {"type": "integer", "default": 2},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "find_contradictions",
        "description": "Find contradictions in the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
        },
    },
    {
        "name": "find_experts_for",
        "description": "Find experts ranked by authority on a topic",
        "inputSchema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
        },
    },
    {
        "name": "graph_timeline",
        "description": "Get edges created or modified since a given date",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO date string"},
                "topic": {"type": "string"},
            },
            "required": ["since"],
        },
    },
    {
        "name": "pagerank",
        "description": "Compute PageRank scores for all nodes",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "communities",
        "description": "Detect communities using Louvain algorithm",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "shortest_path",
        "description": "Find shortest path between two nodes",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "target_id": {"type": "string"},
            },
            "required": ["source_id", "target_id"],
        },
    },
    {
        "name": "auto_capture_status",
        "description": "Get auto-capture system status",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

# -----------------------------------------------------------------------
# Dispatch table
# -----------------------------------------------------------------------

TOOL_DISPATCH = {
    "search_experts": search_experts,
    "list_experts": list_experts,
    "stats": stats,
    "get_context": get_context,
    "add_node": add_node,
    "add_edge": add_edge,
    "update_edge": update_edge,
    "expire_edge": expire_edge,
    "query_graph": query_graph,
    "traverse_graph": traverse_graph,
    "find_contradictions": find_contradictions,
    "find_experts_for": find_experts_for,
    "graph_timeline": graph_timeline,
    "pagerank": pagerank,
    "communities": communities,
    "shortest_path": shortest_path,
    "auto_capture_status": auto_capture_status,
}
