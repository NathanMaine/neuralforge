# Contributing to NeuralForge

Thanks for your interest in NeuralForge. This project is built for people who want GPU-native knowledge intelligence, and contributions make it better for everyone.

## Quick Start for Contributors

1. Fork the repo
2. Clone your fork
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `pytest tests/ -v`
6. Push and open a PR

## Development Setup

NeuralForge uses NVIDIA technologies that require GPU access. For development without a GPU:

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/neuralforge.git
cd neuralforge

# Install dependencies
pip install -r requirements.txt

# Run tests (uses networkx fallback when cuGraph is not available)
pytest tests/ -v
```

The codebase is designed so that all tests pass without a GPU. cuGraph operations fall back to networkx automatically.

## What We Need Help With

Check the [Issues](https://github.com/NathanMaine/neuralforge/issues) tab for tasks labeled:

- `good first issue` - Small, well-defined tasks perfect for your first contribution
- `documentation` - Improve docs, add examples, fix typos
- `help wanted` - Medium complexity features that need implementation
- `enhancement` - New feature proposals
- `nvidia-stack` - NVIDIA-specific optimizations (Triton configs, TRT-LLM tuning)

## Code Standards

- Python 3.11+
- Type hints on all public functions
- Tests for every new feature (we maintain 900+ tests)
- Follow existing patterns in the codebase
- Run `pytest tests/ -v` before submitting

## PR Guidelines

- One feature per PR
- Include tests
- Update docs if you change behavior
- Keep PRs focused and reviewable

## Architecture Overview

```
forge/
  core/       - Shared clients (Triton, NIM, Qdrant, embeddings)
  graph/      - GPU-accelerated knowledge graph (cuGraph + Parquet)
  layers/     - Layered context loading + compression
  guardrails/ - NeMo Guardrails integration
  ingest/     - Ingestion pipelines (blog, document, conversation)
  api/        - FastAPI routes + middleware
  mcp/        - MCP server (17 tools)
  workers/    - Background tasks (discovery, scraping)
```

## Questions?

Open a [Discussion](https://github.com/NathanMaine/neuralforge/discussions) for questions, ideas, or feedback.
