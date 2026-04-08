"""NeuralForge FastAPI application with lifespan management.

Initializes graph store, graph engine, bootstraps knowledge, and starts
the background scheduler on startup.  Cleans up on shutdown.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from forge.api.middleware.audit import AuditMiddleware
from forge.api.routes import dashboard, events, graph, ingest, proxy, search
from forge.graph.store import GraphStore
from forge.graph.engine import GraphEngine
from forge.graph.bootstrap import bootstrap_graph
from forge.guardrails.rails import GuardrailsEngine
from forge.workers.scheduler import start_scheduler, stop_scheduler
from forge.mcp.transport import router as mcp_router

logger = logging.getLogger(__name__)

# Module-level singletons set during lifespan
graph_store: GraphStore | None = None
graph_engine: GraphEngine | None = None
guardrails_engine: GuardrailsEngine | None = None


def get_graph_engine() -> GraphEngine | None:
    """Return the global GraphEngine (set during lifespan)."""
    return graph_engine


def get_graph_store() -> GraphStore | None:
    """Return the global GraphStore (set during lifespan)."""
    return graph_store


def get_guardrails_engine() -> GuardrailsEngine | None:
    """Return the global GuardrailsEngine (set during lifespan)."""
    return guardrails_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the NeuralForge application."""
    global graph_store, graph_engine, guardrails_engine

    # --- Startup ---
    logger.info("NeuralForge starting up...")

    # 1. Init graph store + engine
    graph_store = GraphStore()
    graph_engine = GraphEngine(graph_store)
    graph_engine.load()
    logger.info("Graph engine loaded: %d nodes, %d edges",
                graph_engine.node_count(), graph_engine.edge_count())

    # 2. Bootstrap graph (only runs if empty)
    try:
        bootstrap_result = bootstrap_graph(graph_engine)
        if not bootstrap_result.get("skipped"):
            graph_store.save()
            logger.info("Graph bootstrapped: %s", bootstrap_result)
    except Exception as exc:
        logger.warning("Bootstrap failed (non-fatal): %s", exc)

    # 3. Init guardrails
    guardrails_engine = GuardrailsEngine()

    # 4. Start background scheduler
    try:
        start_scheduler()
    except Exception as exc:
        logger.warning("Scheduler start failed (non-fatal): %s", exc)

    logger.info("NeuralForge ready")
    yield

    # --- Shutdown ---
    logger.info("NeuralForge shutting down...")
    stop_scheduler()
    if graph_store is not None:
        graph_store.save()
    logger.info("NeuralForge stopped")


app = FastAPI(
    title="NeuralForge",
    version="1.0.0",
    description="GPU-accelerated expert knowledge graph with layered RAG",
    lifespan=lifespan,
)

# --- Middleware ---
app.add_middleware(AuditMiddleware)

# --- Static files ---
_web_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web")
if os.path.isdir(_web_dir):
    app.mount("/static", StaticFiles(directory=_web_dir), name="static")

# --- Routers ---
app.include_router(dashboard.router)
app.include_router(search.router)
app.include_router(events.router)
app.include_router(graph.router)
app.include_router(ingest.router)
app.include_router(proxy.router)
app.include_router(mcp_router)


# --- Root routes ---

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the dashboard index.html."""
    index_path = os.path.join(_web_dir, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "NeuralForge API is running. No web UI found."})


@app.get("/health")
async def health_check():
    """Health endpoint with service status checks."""
    from forge.core import qdrant_client
    from forge.workers.scheduler import is_running as scheduler_running

    qdrant_status = "unknown"
    try:
        qdrant_status = qdrant_client.get_status()
    except Exception:
        qdrant_status = "red"

    engine = get_graph_engine()
    graph_nodes = engine.node_count() if engine else 0
    graph_edges = engine.edge_count() if engine else 0

    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "qdrant": qdrant_status,
            "graph": {
                "nodes": graph_nodes,
                "edges": graph_edges,
            },
            "guardrails": {
                "enabled": guardrails_engine.enabled if guardrails_engine else False,
            },
            "scheduler": {
                "running": scheduler_running(),
            },
        },
    }
