"""Graph REST API -- full CRUD + cuGraph-powered queries."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from forge.graph.models import EdgeSource, EdgeType, NodeType

router = APIRouter(prefix="/api/graph", tags=["graph"])
logger = logging.getLogger(__name__)


# --- Request models ---

class CreateNodeRequest(BaseModel):
    name: str
    node_type: str
    description: Optional[str] = None
    metadata: Optional[dict] = None


class UpdateNodeRequest(BaseModel):
    description: Optional[str] = None
    metadata: Optional[dict] = None


class CreateEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    confidence: float = 1.0
    source: str = "manual"
    evidence: list[str] = Field(default_factory=list)
    metadata: Optional[dict] = None


class UpdateEdgeRequest(BaseModel):
    weight: Optional[float] = None
    confidence: Optional[float] = None
    evidence: Optional[list[str]] = None
    metadata: Optional[dict] = None


# --- Helpers ---

def _get_engine():
    from forge.api.main import get_graph_engine
    engine = get_graph_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Graph engine not initialized")
    return engine


# --- Node CRUD ---

@router.post("/nodes", status_code=201)
async def create_node(req: CreateNodeRequest):
    """Create a new graph node."""
    engine = _get_engine()
    try:
        node_type = NodeType(req.node_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid node_type: {req.node_type}")

    node = engine.add_node(
        name=req.name,
        node_type=node_type,
        description=req.description,
        metadata=req.metadata,
    )
    engine.store.save()
    return node.model_dump()


@router.get("/nodes")
async def list_nodes(
    node_type: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(100, ge=1, le=1000),
):
    """List graph nodes, optionally filtered by type or name search."""
    engine = _get_engine()

    if q:
        nodes = engine.store.search_nodes(q, node_type=node_type, limit=limit)
    elif node_type:
        nodes = engine.store.get_nodes_by_type(node_type)[:limit]
    else:
        # Return all nodes up to limit
        nodes = [
            engine.store._row_to_node(row)
            for _, row in engine.store.nodes_df.head(limit).iterrows()
        ] if not engine.store.nodes_df.empty else []

    return {"nodes": [n.model_dump() for n in nodes], "count": len(nodes)}


@router.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Get a single node by ID."""
    engine = _get_engine()
    node = engine.store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node.model_dump()


@router.patch("/nodes/{node_id}")
async def update_node(node_id: str, req: UpdateNodeRequest):
    """Update a node's description and/or metadata."""
    engine = _get_engine()
    mask = engine.store.nodes_df["id"] == node_id
    if not mask.any():
        raise HTTPException(status_code=404, detail="Node not found")

    if req.description is not None:
        engine.store.nodes_df.loc[mask, "description"] = req.description
    if req.metadata is not None:
        engine.store.nodes_df.loc[mask, "metadata"] = engine.store._serialize_json(req.metadata)

    engine.store.save()
    node = engine.store.get_node(node_id)
    return node.model_dump()


# --- Edge CRUD ---

@router.post("/edges", status_code=201)
async def create_edge(req: CreateEdgeRequest):
    """Create a new graph edge."""
    engine = _get_engine()

    try:
        edge_type = EdgeType(req.edge_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid edge_type: {req.edge_type}")

    try:
        source_enum = EdgeSource(req.source)
    except ValueError:
        source_enum = EdgeSource.manual

    # Validate nodes exist
    if engine.store.get_node(req.source_id) is None:
        raise HTTPException(status_code=404, detail=f"Source node {req.source_id} not found")
    if engine.store.get_node(req.target_id) is None:
        raise HTTPException(status_code=404, detail=f"Target node {req.target_id} not found")

    edge = engine.add_edge(
        source_id=req.source_id,
        target_id=req.target_id,
        edge_type=edge_type,
        weight=req.weight,
        confidence=req.confidence,
        source=source_enum,
        evidence=req.evidence,
        metadata=req.metadata,
    )
    engine.store.save()
    return edge.model_dump()


@router.patch("/edges/{edge_id}")
async def update_edge(edge_id: str, req: UpdateEdgeRequest):
    """Update edge fields."""
    engine = _get_engine()
    fields = {}
    if req.weight is not None:
        fields["weight"] = req.weight
    if req.confidence is not None:
        fields["confidence"] = req.confidence
    if req.evidence is not None:
        fields["evidence"] = req.evidence
    if req.metadata is not None:
        fields["metadata"] = req.metadata

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    edge = engine.store.update_edge(edge_id, **fields)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")

    engine.store.save()
    return edge.model_dump()


@router.delete("/edges/{edge_id}", status_code=204)
async def delete_edge(edge_id: str):
    """Permanently delete an edge."""
    engine = _get_engine()
    existing = engine.store.get_edge(edge_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    engine.store.delete_edge(edge_id)
    engine.store.save()


# --- Query endpoints ---

@router.get("/traverse")
async def traverse_graph(
    node_id: str = Query(...),
    depth: int = Query(2, ge=1, le=10),
    edge_types: Optional[str] = Query(None, description="Comma-separated edge types"),
    as_of: Optional[str] = Query(None, description="ISO date for temporal filter"),
):
    """BFS traversal from a node."""
    engine = _get_engine()
    types_list = edge_types.split(",") if edge_types else None
    result = engine.traverse(node_id, depth=depth, edge_types=types_list, as_of=as_of)
    return result.model_dump()


@router.get("/contradictions")
async def find_contradictions(
    topic: Optional[str] = Query(None),
):
    """Find contradicting edges in the graph."""
    engine = _get_engine()
    contradictions = engine.find_contradictions(topic=topic)
    return {
        "contradictions": [c.model_dump() for c in contradictions],
        "count": len(contradictions),
    }


@router.get("/experts-for")
async def experts_for_topic(
    topic: str = Query(..., min_length=1),
):
    """Find experts ranked by authority on a topic."""
    engine = _get_engine()
    rankings = engine.expert_authority(topic)
    return {
        "topic": topic,
        "rankings": [r.model_dump() for r in rankings],
    }


@router.get("/rankings")
async def get_rankings(
    topic: str = Query(..., min_length=1),
):
    """Get expert rankings for a topic (alias for experts-for)."""
    engine = _get_engine()
    rankings = engine.expert_authority(topic)
    return {
        "topic": topic,
        "rankings": [r.model_dump() for r in rankings],
    }


# --- cuGraph-powered endpoints ---

@router.get("/pagerank")
async def get_pagerank():
    """Compute PageRank scores for all nodes."""
    engine = _get_engine()
    scores = engine.pagerank()
    # Enrich with node names
    enriched = []
    for node_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        node = engine.store.get_node(node_id)
        enriched.append({
            "node_id": node_id,
            "name": node.name if node else "unknown",
            "node_type": node.node_type.value if node else "unknown",
            "score": score,
        })
    return {"pagerank": enriched}


@router.get("/communities")
async def get_communities():
    """Detect communities using Louvain algorithm."""
    engine = _get_engine()
    communities = engine.find_communities()
    # Group by community ID
    groups: dict[int, list[dict]] = {}
    for node_id, community_id in communities.items():
        node = engine.store.get_node(node_id)
        entry = {
            "node_id": node_id,
            "name": node.name if node else "unknown",
            "node_type": node.node_type.value if node else "unknown",
        }
        groups.setdefault(community_id, []).append(entry)

    return {
        "communities": groups,
        "num_communities": len(groups),
        "total_nodes": len(communities),
    }


@router.get("/shortest-path")
async def get_shortest_path(
    source: str = Query(..., description="Source node ID"),
    target: str = Query(..., description="Target node ID"),
):
    """Find shortest path between two nodes."""
    engine = _get_engine()
    path = engine.shortest_path(source, target)
    if not path:
        return {"path": [], "length": 0, "found": False}

    enriched = []
    for node_id in path:
        node = engine.store.get_node(node_id)
        enriched.append({
            "node_id": node_id,
            "name": node.name if node else "unknown",
            "node_type": node.node_type.value if node else "unknown",
        })

    return {"path": enriched, "length": len(path) - 1, "found": True}


@router.get("/timeline")
async def get_timeline(
    since: str = Query(..., description="ISO date string"),
    topic: Optional[str] = Query(None),
):
    """Get edges created or modified since a given date."""
    engine = _get_engine()
    edges = engine.find_changes_since(since, topic=topic)
    return {
        "since": since,
        "topic": topic,
        "edges": [e.model_dump() for e in edges],
        "count": len(edges),
    }


@router.get("/stats")
async def get_graph_stats():
    """Get comprehensive graph statistics."""
    engine = _get_engine()
    stats = engine.store.get_stats()
    return stats.model_dump()
