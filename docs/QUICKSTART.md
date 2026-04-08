# NeuralForge Quickstart

From zero to a running knowledge platform in 5 minutes.

---

## Prerequisites

- NVIDIA GPU with 16GB+ VRAM
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [NGC API key](https://org.ngc.nvidia.com/setup)

---

## Step 1: Clone and Configure

```bash
git clone https://github.com/NathanMaine/neuralforge.git
cd neuralforge

cp .env.example .env
```

Edit `.env` and set your NGC API key:

```
NGC_API_KEY=nvapi-your-key-here
```

---

## Step 2: Run Preflight Check

```bash
bash check_env.sh
```

This validates:
- NVIDIA GPU and VRAM (16GB+ required)
- Docker and Docker Compose
- NVIDIA Container Toolkit
- NGC API key
- Available disk space (50GB+ for NIM cache)
- Port availability (8090, 8000, 8001, 6333)

Fix any failures before proceeding.

---

## Step 3: Launch the Stack

```bash
docker compose up -d
```

This starts 4 containers:

| Container | Port | Purpose |
|:---|:---|:---|
| `neuralforge-api` | 8090 | NeuralForge REST API |
| `neuralforge-nim` | 8000 | NIM LLM (Llama 3.1 8B via TensorRT-LLM) |
| `neuralforge-triton` | 8001 | Triton Inference Server (embedding + reranking) |
| `neuralforge-qdrant` | 6333 | Qdrant vector database |

The NIM container will download the model on first start (~15GB). This takes a few minutes depending on your connection.

Monitor startup:

```bash
docker compose logs -f
```

---

## Step 4: Verify

```bash
# Check API health
curl http://localhost:8090/health

# Check NIM is ready
curl http://localhost:8000/v1/health/ready

# Check Qdrant
curl http://localhost:6333/healthz
```

---

## Step 5: Ingest Your First Expert

```bash
# Option A: Blog
python examples/ingest_blog.py --url "https://timdettmers.com" --expert "Tim Dettmers"

# Option B: Local documents
python examples/ingest_documents.py --folder ./my-papers --expert "My Research"

# Option C: arXiv
python examples/ingest_arxiv.py --author "Geoffrey Hinton" --max-papers 10
```

---

## Step 6: Query

```bash
python examples/query_experts.py --query "What is LoRA and why does it work?"
```

Or with curl:

```bash
curl -X POST http://localhost:8090/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantization trade-offs"}'
```

---

## Step 7: Explore the Graph

```bash
# Who disagrees?
python examples/find_contradictions.py --topic "quantization"

# Who clusters together?
python examples/expert_communities.py

# Who is the authority?
python examples/influence_ranking.py --topic "transformer architectures"
```

---

## Stopping and Restarting

```bash
# Stop (preserves data)
docker compose down

# Restart
docker compose up -d

# Full reset (deletes all data)
docker compose down -v
```

---

## Troubleshooting

### NIM container won't start
- Check VRAM: `nvidia-smi` -- NIM needs ~15GB for Llama 3.1 8B
- Check NGC key: `echo $NGC_API_KEY`
- Check logs: `docker compose logs nim-llm`

### "Connection refused" on port 8090
- NeuralForge waits for NIM health check before starting
- NIM needs 2-3 minutes on first launch (model download)
- Check: `docker compose ps` for container status

### Triton model errors
- Triton model configs are placeholders -- you need to export and place ONNX models
- See `triton/models/embedding/config.pbtxt` for expected format
- See [NVIDIA Stack docs](NVIDIA_STACK.md) for export instructions

### Low VRAM
- Reduce NIM shm_size in `docker-compose.yml`
- Use a smaller NIM model (modify `NIM_MODEL` in `.env`)
- Disable Triton reranking (embedding-only mode)

---

## Next Steps

- [Architecture](ARCHITECTURE.md) -- understand the system design
- [Ingestion Guide](INGESTION_GUIDE.md) -- all supported data sources
- [Graph Algorithms](GRAPH_ALGORITHMS.md) -- cuGraph operations in detail
- [Guardrails Guide](GUARDRAILS_GUIDE.md) -- configure safety rails
