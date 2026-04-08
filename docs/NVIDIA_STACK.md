# NeuralForge NVIDIA Stack

Why each NVIDIA technology was chosen and what role it plays.

---

## Stack Overview

| Technology | Role in NeuralForge | Why This One |
|:---|:---|:---|
| **NIM** | LLM inference (chat, classification, JSON generation) | Production-grade serving with TensorRT-LLM optimization. One container, one API. |
| **TensorRT-LLM** | LLM optimization (inside NIM) | 2-4x throughput over vanilla inference. Automatic quantization, KV cache optimization. |
| **Triton Inference Server** | Embedding + cross-encoder reranking | Dynamic batching, GPU execution, model hot-swap. Serves both models from one process. |
| **NeMo Guardrails** | Input/output/retrieval safety rails | Programmable guardrails with Colang. PII scrub, hallucination check, attribution verification. |
| **RAPIDS cuGraph** | GPU-accelerated graph algorithms | PageRank, Louvain, BFS on GPU. 10-100x faster than networkx on large graphs. |
| **CUDA** | Foundation runtime | Everything runs on CUDA 12.5. Unified memory model across all components. |

---

## NIM (NVIDIA Inference Microservice)

### What It Does

NIM packages a production-ready LLM behind an OpenAI-compatible API. NeuralForge uses it for:

- **Chat completion** -- answering user queries with context
- **JSON classification** -- classifying expert relationships during discovery
- **Streaming** -- real-time token streaming for interactive use

### Configuration

```yaml
# docker-compose.yml
nim-llm:
  image: nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
  environment:
    NGC_API_KEY: ${NGC_API_KEY}
```

### Why NIM Over Alternatives

| Alternative | Why NIM Wins |
|:---|:---|
| vLLM | NIM includes TensorRT-LLM optimizations out of the box |
| Ollama | NIM provides production health checks, metrics, multi-GPU support |
| Raw TensorRT-LLM | NIM handles model download, caching, and API serving |

### Swapping Models

Change the NIM model in `.env`:

```
NIM_MODEL=nvcr.io/nim/meta/llama-3.1-70b-instruct:latest
```

Available NIM models: [NGC Catalog](https://catalog.ngc.nvidia.com/models?filters=nim)

---

## TensorRT-LLM

### What It Does

TensorRT-LLM runs inside the NIM container and provides:

- **FP16/INT8/INT4 quantization** -- automatically selected based on GPU
- **KV cache optimization** -- paged attention for efficient memory use
- **Continuous batching** -- handles multiple concurrent requests efficiently
- **Speculative decoding** -- reduced latency for supported models

### You Don't Configure It

TensorRT-LLM is abstracted by NIM. The optimizations are automatic based on your GPU architecture (Ampere, Ada Lovelace, Hopper, Blackwell).

---

## Triton Inference Server

### What It Does

Triton serves two models for NeuralForge:

1. **Embedding model** (nomic-embed-text v1.5) -- converts text to 768-dim vectors
2. **Reranker** (cross-encoder) -- scores query-document relevance

### Configuration

Model configs are in `triton/models/`:

```
triton/models/
  embedding/
    config.pbtxt    # Model configuration
    1/
      model.onnx    # ONNX model (you provide this)
  reranker/
    config.pbtxt
    1/
      model.onnx
```

### Key Features Used

| Feature | How NeuralForge Uses It |
|:---|:---|
| Dynamic batching | Batches embedding requests (up to 64) for GPU efficiency |
| TensorRT acceleration | FP16 inference for embedding and reranking |
| Model versioning | Hot-swap models without container restart |
| Health/metrics | Prometheus metrics on port 8003 |

### Exporting Models to ONNX

```python
# Example: Export nomic-embed-text to ONNX
from optimum.exporters.onnx import main_export
main_export("nomic-ai/nomic-embed-text-v1.5", output="triton/models/embedding/1/")
```

---

## NeMo Guardrails

### What It Does

NeMo Guardrails provides programmable safety rails at three points in the pipeline:

```
User Query ──> [Input Rails] ──> Processing ──> [Output Rails] ──> Response
                                      |
                                [Retrieval Rails]
```

### Rails Configured

**Input Rails:**
- `check pii` -- scrubs personal information from queries
- `check jailbreak` -- detects prompt injection attempts
- `check topic relevance` -- blocks off-topic queries

**Output Rails:**
- `check hallucination` -- verifies expert citations against the knowledge graph
- `check attribution` -- ensures named experts exist as graph nodes
- `add provenance` -- appends source metadata to responses

**Retrieval Rails:**
- `check retrieval relevance` -- filters irrelevant chunks before they reach the LLM

### Configuration Files

```
forge/guardrails/config/
  config.yml    # Models and rail definitions
  flows.co      # Colang flow logic
```

### Custom Actions

NeuralForge implements 6 custom rail actions in `forge/guardrails/actions.py`:

| Action | Purpose |
|:---|:---|
| `check_hallucination` | Cross-reference citations with graph nodes |
| `check_attribution` | Verify expert names exist in the knowledge graph |
| `add_provenance` | Append source URLs and timestamps |
| `scrub_pii_input` | Remove PII from user input |
| `self_correction` | Re-generate response when hallucination is detected |
| `log_rail_decision` | Audit log for all rail decisions |

### Graceful Degradation

If `nemoguardrails` is not installed, NeuralForge runs without safety rails. All guardrail checks return `allowed=True` and pass content through unchanged. This is logged as a warning at startup.

---

## RAPIDS cuGraph

### What It Does

cuGraph runs graph algorithms directly on the GPU, providing 10-100x speedup over CPU-based alternatives for large graphs.

### Algorithms Used

| Algorithm | NeuralForge Use |
|:---|:---|
| **PageRank** | Expert authority scoring -- who is most influential on a topic |
| **Louvain** | Community detection -- find natural expert clusters |
| **BFS** | Graph traversal -- explore connections from any node |
| **Shortest Path** | Find how two nodes are connected |

### Fallback

When cuGraph is not available (no GPU, testing, development), NeuralForge falls back to networkx. The API is identical; only the execution backend changes.

```python
try:
    import cudf
    import cugraph
    HAS_CUGRAPH = True
except ImportError:
    import networkx as nx
    HAS_CUGRAPH = False
```

### Graph Data Model

- **Storage:** Apache Parquet (portable, compressed, schema-preserving)
- **In-memory:** cudf DataFrames (GPU) or pandas DataFrames (CPU)
- **Reload:** Configurable interval (default 300s) or force-reload on mutation

---

## CUDA

### Version

NeuralForge targets CUDA 12.5 via the RAPIDS base image:

```dockerfile
FROM nvcr.io/nvidia/rapidsai/base:24.12-cuda12.5-py3.12
```

### GPU Memory Management

| Component | Typical VRAM | Notes |
|:---|:---|:---|
| NIM (Llama 3.1 8B) | ~15GB | Varies by quantization |
| Triton (embedding) | ~1GB | nomic-embed-text v1.5 |
| Triton (reranker) | ~1GB | Cross-encoder |
| cuGraph | <1GB | Scales with graph size |
| **Total** | **~18GB** | Fits on 24GB GPUs |

For GPUs with less VRAM, reduce NIM model size or run Triton models on CPU.
