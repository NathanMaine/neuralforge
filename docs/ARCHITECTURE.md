# NeuralForge Architecture

Deep dive on system design, data flow, and component interactions.

---

## Design Principles

1. **Data sovereignty** -- your knowledge never leaves your hardware
2. **GPU-first** -- graph algorithms, embeddings, inference, and reranking all run on GPU
3. **Layered context** -- budget-aware retrieval assembles multi-layer context windows
4. **Temporal graphs** -- every edge has valid-from/to timestamps for knowledge evolution tracking
5. **Graceful degradation** -- cuGraph falls back to networkx, guardrails disable cleanly, Triton falls back to HTTP

---

## System Overview

```
User Query
    |
    v
[Input Rails] ──> PII Scrub, Jailbreak Check, Topic Relevance
    |
    v
[Layered Context Engine]
    |
    ├── Layer 0: System Identity / Persona
    ├── Layer 1: cuGraph Context (PageRank, Contradictions)
    ├── Layer 2: Compressed Vector Search (Qdrant + Triton + AAAK)
    └── Layer 3: Deep Search (uncompressed high-relevance)
    |
    v
[NIM LLM Generation] ──> TensorRT-LLM optimized inference
    |
    v
[Output Rails] ──> Hallucination Check, Attribution, Provenance
    |
    v
Response + Sources
```

---

## Component Architecture

### API Layer (`forge/api/`)

FastAPI application with:
- REST endpoints for ingestion, query, graph operations, and system status
- Middleware for request logging and error handling
- Async request handling for concurrent workloads

### Core Services (`forge/core/`)

| Module | Responsibility |
|:---|:---|
| `nim_client.py` | OpenAI-compatible chat, JSON classification, streaming via NIM |
| `triton_client.py` | Batch embedding with TTL cache via Triton Inference Server |
| `qdrant_client.py` | Vector search, upsert, scroll operations against Qdrant |
| `embeddings.py` | High-level embedding API with batching and caching |
| `sync.py` | Single write path ensuring Qdrant + cuGraph consistency |
| `models.py` | Pydantic v2 data models (SearchResult, IngestJob, etc.) |
| `utils.py` | Shared utilities |

### Knowledge Graph (`forge/graph/`)

| Module | Responsibility |
|:---|:---|
| `engine.py` | GPU-accelerated graph operations (PageRank, Louvain, BFS, shortest path) |
| `store.py` | Parquet-backed graph persistence with full CRUD and temporal filtering |
| `models.py` | Graph data models (Node, Edge, NodeType, EdgeType, Contradiction) |
| `discovery.py` | Auto-discovery of expert relationships via NIM classification |
| `bootstrap.py` | Initial graph construction from ingested data |

**Node types:** expert, concept, technique, tool, dataset, model, paper, institution

**Edge types:** 18 relationship types including `agrees_with`, `contradicts`, `supersedes`, `expert_in`, `depends_on`, `incompatible_with`, `cites`, and more.

**Edge sources:** manual, auto-discovered, mined (from conversation logs)

### Ingestion Pipeline (`forge/ingest/`)

| Module | Responsibility |
|:---|:---|
| `blog_scraper.py` | Multi-strategy blog discovery (RSS, sitemap, link crawling) |
| `document_loader.py` | PDF, DOCX, TXT, HTML, CSV loading via PyMuPDF |
| `chunker.py` | Semantic and fixed-size chunking with overlap |
| `upserter.py` | Batch upsert to Qdrant with deduplication |
| `pii_scrubber.py` | Regex + pattern-based PII removal before storage |
| `conversation_miner.py` | Claude/ChatGPT/Slack export parsing into chunks + edges |

### Retrieval Layers (`forge/layers/`)

| Module | Responsibility |
|:---|:---|
| `engine.py` | 4-layer context assembly with budget allocation |
| `compressor.py` | AAAK (Always Ask, Always Keep) fact-preserving compression |
| `ranker.py` | Token budget allocation across layers, BM25 keyword ranking |

**Budget allocation strategy:**

```
Layer 0 (Identity):    Fixed ~100 tokens
Layer 1 (Graph):       10-15% of remaining budget
Layer 2 (Compressed):  50-60% of remaining budget
Layer 3 (Deep):        Remaining budget
```

### Guardrails (`forge/guardrails/`)

| Module | Responsibility |
|:---|:---|
| `rails.py` | NeMo Guardrails wrapper with graceful fallback |
| `actions.py` | Custom rail actions (hallucination check, attribution, PII scrub) |
| `config/config.yml` | Rails configuration (NIM backend) |
| `config/flows.co` | Colang flow definitions for input/output/retrieval rails |

### Background Workers (`forge/workers/`)

| Module | Responsibility |
|:---|:---|
| `discovery_worker.py` | Periodic expert relationship discovery (every 6h default) |
| `scrape_worker.py` | Daily blog scraping for monitored sources |
| `scheduler.py` | APScheduler-based job scheduling |

---

## Data Flow: Ingestion

```
Source (blog/YouTube/arXiv/docs)
    |
    v
[Scraper/Loader] ──> Raw content extraction
    |
    v
[PII Scrubber] ──> Remove personal information
    |
    v
[Chunker] ──> Semantic boundary detection + overlap
    |
    v
[Triton Embedding] ──> 768-dim vectors (batched, cached)
    |
    v
[Sync Layer] ──> Atomic writes to both stores
    ├── Qdrant (vectors + metadata)
    └── cuGraph Store (expert/concept nodes + edges)
```

---

## Data Flow: Query

```
User Query
    |
    v
[Input Rails] ──> Scrub PII, detect jailbreak, check topic
    |
    v
[Embedding] ──> Query vector via Triton
    |
    v
[Qdrant Search] ──> Top-K candidate chunks
    |
    v
[Triton Reranking] ──> Cross-encoder relevance scoring
    |
    v
[cuGraph Context] ──> PageRank authority + contradiction detection
    |
    v
[AAAK Compression] ──> Fact-preserving compression for Layer 2
    |
    v
[Budget Allocation] ──> Distribute tokens across 4 layers
    |
    v
[NIM Generation] ──> TensorRT-LLM optimized LLM response
    |
    v
[Output Rails] ──> Verify citations, check attribution, add provenance
    |
    v
Response + Sources + Provenance
```

---

## Storage Architecture

### Vector Store (Qdrant)

- **Collection:** `neuralforge` (configurable)
- **Vector size:** 768 dimensions (nomic-embed-text v1.5)
- **Payload fields:** expert, title, source, chunk_index, created_at, tags
- **Persistence:** Docker volume (`qdrant-data`)

### Graph Store (Parquet)

- **Format:** Apache Parquet via pandas/pyarrow
- **Files:** `nodes.parquet`, `edges.parquet`
- **Location:** `data/graph/` (Docker volume `forge-data`)
- **In-memory:** cuGraph (GPU) or networkx (CPU fallback)

### NIM Model Cache

- **Location:** `/opt/nim/.cache` (Docker volume `nim-cache`)
- **Shared** between NeuralForge API and NIM containers
- **Size:** 15-50GB depending on model

---

## Concurrency Model

- **API:** uvicorn with 4 workers (configurable)
- **Ingestion:** Async httpx for I/O-bound scraping
- **Embedding:** Batched requests to Triton (configurable batch size)
- **Graph:** Periodic reload from Parquet store (default 300s)
- **Workers:** APScheduler with configurable intervals
