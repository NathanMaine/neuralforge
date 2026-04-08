<p align="center">
  <h1 align="center">NeuralForge</h1>
  <p align="center"><strong>Your experts. Your GPU. Your data never leaves.</strong></p>
  <p align="center">
    NeuralForge gives every GPU owner a team of domain experts on their desk — aligned to their research, tracking their field, always available, never forgetting.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-NIM-76B900?style=for-the-badge&logo=nvidia" alt="NIM">
  <img src="https://img.shields.io/badge/NVIDIA-TensorRT--LLM-76B900?style=for-the-badge&logo=nvidia" alt="TensorRT-LLM">
  <img src="https://img.shields.io/badge/NVIDIA-Triton-76B900?style=for-the-badge&logo=nvidia" alt="Triton">
  <img src="https://img.shields.io/badge/NVIDIA-NeMo_Guardrails-76B900?style=for-the-badge&logo=nvidia" alt="NeMo Guardrails">
  <img src="https://img.shields.io/badge/NVIDIA-RAPIDS_cuGraph-76B900?style=for-the-badge&logo=nvidia" alt="RAPIDS cuGraph">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia" alt="CUDA">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?style=flat-square" alt="Python 3.12">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/status-active_development-green?style=flat-square" alt="Status">
</p>

---

## What Makes This Different

Most knowledge platforms choose between cloud convenience and data control. NeuralForge refuses that trade-off.

| Capability | LangChain + OpenAI | Cloud RAG (Pinecone) | **NeuralForge** |
|:---|:---:|:---:|:---:|
| Data stays on your hardware | | | **Yes** |
| GPU-accelerated graph algorithms | | | **cuGraph (Louvain, PageRank)** |
| Expert contradiction detection | | | **Automatic** |
| Hallucination guardrails | Partial | | **NeMo Guardrails** |
| Temporal knowledge tracking | | | **Valid-from/to on every edge** |
| Multi-source ingestion (blogs, YouTube, arXiv, docs) | Plugin | Plugin | **Built-in** |
| Single-command deployment | | | **`docker compose up`** |
| PII scrubbing before storage | | | **Built-in** |

---

## Architecture

```
                         NeuralForge Stack
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │   ┌──────────────┐     ┌──────────────────────────┐  │
    │   │  NeuralForge  │────▶│  NIM (TensorRT-LLM)      │  │
    │   │  API :8090    │     │  Llama 3.1 8B    :8000   │  │
    │   │               │     └──────────────────────────┘  │
    │   │  FastAPI       │                                  │
    │   │  cuGraph       │     ┌──────────────────────────┐  │
    │   │  NeMo Rails    │────▶│  Triton Inference Server  │  │
    │   │  Layered CTX   │     │  Embed + Rerank  :8001   │  │
    │   │               │     └──────────────────────────┘  │
    │   │               │                                  │
    │   │               │     ┌──────────────────────────┐  │
    │   │               │────▶│  Qdrant                   │  │
    │   │               │     │  Vector DB       :6333   │  │
    │   └──────────────┘     └──────────────────────────┘  │
    │                                                      │
    │                     ┌────────────────┐                │
    │                     │  NVIDIA GPU(s)  │                │
    │                     │  CUDA 12.5      │                │
    │                     └────────────────┘                │
    └──────────────────────────────────────────────────────┘
```

**Four containers, one GPU, zero cloud dependencies.**

| Container | Role | NVIDIA Tech | Port |
|:---|:---|:---|:---|
| `neuralforge-api` | Knowledge platform API | RAPIDS cuGraph, NeMo Guardrails | 8090 |
| `neuralforge-nim` | LLM inference (TensorRT-LLM optimized) | NIM, TensorRT-LLM | 8000 |
| `neuralforge-triton` | Embedding + reranking | Triton Inference Server | 8001 |
| `neuralforge-qdrant` | Vector storage | -- | 6333 |

---

## Deploy in One Command

```bash
# 1. Clone
git clone https://github.com/NathanMaine/neuralforge.git
cd neuralforge

# 2. Configure
cp .env.example .env
# Edit .env and add your NGC_API_KEY

# 3. Preflight check
bash check_env.sh

# 4. Launch
docker compose up -d
```

That's it. Four GPU-accelerated containers, health-checked and ready.

---

## Ingest Your First Expert

```python
import httpx

# Add Tim Dettmers' blog as a knowledge source
resp = httpx.post("http://localhost:8090/api/v1/ingest", json={
    "source_type": "blog",
    "source_url": "https://timdettmers.com",
    "expert_name": "Tim Dettmers",
})
print(resp.json()["job_id"])
```

Or use the CLI examples:

```bash
# YouTube channel
python examples/ingest_youtube.py --channel "3Blue1Brown" --name "Grant Sanderson"

# arXiv papers
python examples/ingest_arxiv.py --author "Yann LeCun" --max-papers 20

# Local documents
python examples/ingest_documents.py --folder ./papers --expert "Research Team"
```

---

## Ask Your Knowledge Base

```bash
curl -X POST http://localhost:8090/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LoRA and why does it work?", "include_graph_context": true}'
```

Every response flows through 4 context layers:

| Layer | Source | Purpose |
|:---|:---|:---|
| 0 | System identity | Expert persona + attribution rules |
| 1 | cuGraph | PageRank rankings + contradiction warnings |
| 2 | Qdrant + Triton | AAAK-compressed vector search chunks |
| 3 | Deep search | Full uncompressed high-relevance passages |

---

## Discover Relationships

NeuralForge doesn't just store knowledge -- it discovers structure.

### Contradiction Detection

```bash
# Where do your experts disagree?
python examples/find_contradictions.py --topic "quantization"
```

When Expert A says "4-bit quantization preserves 99% of performance" and Expert B says "anything below 8-bit significantly degrades reasoning" -- NeuralForge surfaces that disagreement, with citations to both sources.

### Expert Communities

```bash
# Who clusters together?
python examples/expert_communities.py
```

Louvain community detection (GPU-accelerated via cuGraph) finds natural clusters: researchers who cite each other, authors covering the same topics, institutions with shared methodologies.

### Influence Ranking

```bash
# Who carries the most weight on a topic?
python examples/influence_ranking.py --topic "transformer architectures"
```

PageRank computed across the knowledge graph, weighted by topic-specific edge density. Not popularity -- structural authority.

---

## GPU Graph Algorithms

NeuralForge runs graph algorithms directly on the GPU via RAPIDS cuGraph, with automatic networkx fallback for CPU-only environments.

| Algorithm | What It Does | Use Case |
|:---|:---|:---|
| **PageRank** | Scores node importance by link structure | Expert authority ranking |
| **Louvain** | Detects communities in the graph | Expert clustering by domain |
| **BFS Traversal** | Walks the graph from any node | "Show me everything connected to LoRA" |
| **Shortest Path** | Finds the connection between two nodes | "How is Expert A related to Concept B?" |
| **Temporal Filtering** | Valid-from/to on every edge | "What was the consensus in 2023?" |

---

## Safety and Governance

Every query passes through NeMo Guardrails -- three rail types protecting the pipeline:

### Input Rails
- **PII Scrubbing** -- personal information is stripped before it enters the system
- **Jailbreak Detection** -- prompt injection attempts are caught and blocked
- **Topic Relevance** -- off-topic queries are redirected

### Output Rails
- **Hallucination Check** -- expert citations are verified against the knowledge graph
- **Attribution Verification** -- every named expert must exist as a graph node
- **Provenance Tracking** -- source metadata is appended to every response

### Retrieval Rails
- **Relevance Filtering** -- retrieved chunks are validated for query relevance before reaching the LLM

---

## Benchmarks

Performance measured on representative hardware. Ingestion = blog with 50 articles. Search = single query with graph context. Graph = PageRank over 10K-node graph.

| Hardware | Ingestion | Search (p95) | Graph PageRank | VRAM Used |
|:---|:---:|:---:|:---:|:---:|
| DGX Spark (GB10, 128GB) | -- | -- | -- | -- |
| H100 (80GB) | -- | -- | -- | -- |
| RTX 4090 (24GB) | -- | -- | -- | -- |
| RTX 3090 (24GB) | -- | -- | -- | -- |

*Benchmarks in progress. Hardware sponsors welcome.*

---

## Project Structure

```
neuralforge/
  forge/
    api/              # FastAPI application + middleware
    core/             # NIM client, Triton client, Qdrant client, embeddings
    graph/            # cuGraph engine, Parquet store, auto-discovery
    guardrails/       # NeMo Guardrails config + custom actions
    ingest/           # Blog scraper, document loader, chunker, PII scrubber
    layers/           # Layered context engine + AAAK compression
    workers/          # Background jobs (discovery, scraping)
    mcp/              # Model Context Protocol integration
  triton/models/      # Triton model configs (embedding, reranker)
  examples/           # 8 standalone quickstart scripts
  tests/              # Full test suite
  benchmarks/         # Performance measurement harness
  docs/               # Architecture, guides, NVIDIA stack docs
```

---

## Documentation

| Guide | Description |
|:---|:---|
| [Architecture](docs/ARCHITECTURE.md) | Deep dive on system design and data flow |
| [Quickstart](docs/QUICKSTART.md) | 5-minute deployment guide |
| [Ingestion Guide](docs/INGESTION_GUIDE.md) | How to feed your data into NeuralForge |
| [NVIDIA Stack](docs/NVIDIA_STACK.md) | Why each NVIDIA technology was chosen |
| [Graph Algorithms](docs/GRAPH_ALGORITHMS.md) | cuGraph operations explained |
| [Guardrails Guide](docs/GUARDRAILS_GUIDE.md) | Configuring NeMo Guardrails |

---

## Requirements

- NVIDIA GPU with 16GB+ VRAM (24GB+ recommended)
- NVIDIA Driver 535+, CUDA 12.5+
- Docker with NVIDIA Container Toolkit
- NGC API key ([get one here](https://org.ngc.nvidia.com/setup))
- 50GB+ free disk space (NIM model cache)

---

## Built by Nathan Maine

NVIDIA Inception Member. Building tools that put GPU-accelerated AI in the hands of individual researchers and small teams.

- GitHub: [NathanMaine](https://github.com/NathanMaine)
- HuggingFace: [Nathan-Maine](https://huggingface.co/Nathan-Maine)
- LinkedIn: [Nathan Maine](https://www.linkedin.com/in/nathan-maine/)

---

## License

Apache 2.0. See [LICENSE](LICENSE).
