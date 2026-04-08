"""Dashboard API routes -- experts list, detail, and overview stats."""
import logging

from fastapi import APIRouter, HTTPException

from forge.core.utils import slugify

router = APIRouter(prefix="/api", tags=["dashboard"])
logger = logging.getLogger(__name__)


@router.get("/dashboard")
async def get_dashboard():
    """Return dashboard overview with graph stats, expert count, and service status."""
    from forge.api.main import get_graph_engine, get_guardrails_engine
    from forge.core import qdrant_client
    from forge.workers.scheduler import is_running as scheduler_running, get_jobs

    engine = get_graph_engine()
    stats = engine.store.get_stats() if engine else None

    total_chunks = 0
    qdrant_status = "red"
    try:
        total_chunks = qdrant_client.get_total_chunks()
        qdrant_status = qdrant_client.get_status()
    except Exception:
        pass

    guardrails = get_guardrails_engine()

    return {
        "graph": {
            "total_nodes": stats.total_nodes if stats else 0,
            "total_edges": stats.total_edges if stats else 0,
            "active_edges": stats.active_edges if stats else 0,
            "expired_edges": stats.expired_edges if stats else 0,
            "node_type_counts": stats.node_type_counts if stats else {},
            "edge_type_counts": stats.edge_type_counts if stats else {},
        },
        "knowledge_base": {
            "total_chunks": total_chunks,
            "qdrant_status": qdrant_status,
        },
        "services": {
            "guardrails_enabled": guardrails.enabled if guardrails else False,
            "scheduler_running": scheduler_running(),
            "scheduled_jobs": get_jobs(),
        },
    }


@router.get("/experts")
async def list_experts():
    """List all expert nodes with chunk counts."""
    from forge.api.main import get_graph_engine
    from forge.core import qdrant_client

    engine = get_graph_engine()
    if engine is None:
        return {"experts": []}

    experts = engine.get_all_experts()
    result = []
    for expert in experts:
        chunk_count = 0
        try:
            chunk_count = qdrant_client.count_chunks_for_expert(expert.name)
        except Exception:
            pass

        slug = slugify(expert.name)
        edge_count = len(engine.store.get_edges_for_node(expert.id))

        result.append({
            "id": expert.id,
            "name": expert.name,
            "slug": slug,
            "description": expert.description,
            "chunk_count": chunk_count,
            "edge_count": edge_count,
            "created_at": expert.created_at,
            "metadata": expert.metadata,
        })

    return {"experts": result}


@router.get("/expert/{slug}")
async def get_expert_detail(slug: str):
    """Get detailed information about a single expert by slug."""
    from forge.api.main import get_graph_engine
    from forge.core import qdrant_client

    engine = get_graph_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Graph engine not initialized")

    # Find expert by slug
    experts = engine.get_all_experts()
    expert = None
    for e in experts:
        if slugify(e.name) == slug:
            expert = e
            break

    if expert is None:
        raise HTTPException(status_code=404, detail=f"Expert '{slug}' not found")

    # Get edges
    edges = engine.store.get_edges_for_node(expert.id)
    connections = []
    for edge in edges:
        other_id = edge.target_id if edge.source_id == expert.id else edge.source_id
        other_node = engine.store.get_node(other_id)
        connections.append({
            "edge_id": edge.id,
            "edge_type": edge.edge_type.value,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "direction": "outgoing" if edge.source_id == expert.id else "incoming",
            "connected_node": {
                "id": other_node.id,
                "name": other_node.name,
                "node_type": other_node.node_type.value,
            } if other_node else {"id": other_id, "name": "unknown", "node_type": "unknown"},
            "valid_from": edge.valid_from,
            "valid_to": edge.valid_to,
        })

    chunk_count = 0
    try:
        chunk_count = qdrant_client.count_chunks_for_expert(expert.name)
    except Exception:
        pass

    # Get authority rankings
    rankings = []
    try:
        rankings = [
            {"topic": r.topic, "score": r.score, "edge_count": r.edge_count}
            for r in engine.expert_authority(expert.name)
        ]
    except Exception:
        pass

    return {
        "id": expert.id,
        "name": expert.name,
        "slug": slug,
        "description": expert.description,
        "chunk_count": chunk_count,
        "connections": connections,
        "rankings": rankings,
        "metadata": expert.metadata,
        "created_at": expert.created_at,
    }
