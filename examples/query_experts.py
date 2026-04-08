#!/usr/bin/env python3
"""Query your NeuralForge knowledge base.

Sends a natural language query through the full NeuralForge pipeline:
vector search via Qdrant, graph-enriched context via cuGraph, reranking
through Triton, and answer generation via NIM — all behind NeMo Guardrails.

Usage:
    python examples/query_experts.py --query "What is LoRA and why does it work?"
    python examples/query_experts.py --query "How should I quantize a 70B model?" --expert "Tim Dettmers"
    python examples/query_experts.py --query "Explain attention" --format json

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import json
import sys

import httpx

API = "http://localhost:8090"


def query_knowledge(
    query: str,
    expert: str | None = None,
    max_tokens: int = 4000,
    include_graph: bool = True,
) -> dict:
    """Send a query to the NeuralForge knowledge base."""
    payload = {
        "query": query,
        "max_tokens": max_tokens,
        "include_graph_context": include_graph,
    }
    if expert:
        payload["expert_filter"] = expert

    resp = httpx.post(f"{API}/api/v1/query", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Query the NeuralForge knowledge base")
    parser.add_argument("--query", "-q", required=True, help="Your question")
    parser.add_argument("--expert", "-e", help="Filter to a specific expert")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Context budget (default: 4000)")
    parser.add_argument("--no-graph", action="store_true", help="Disable graph context enrichment")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api

    try:
        result = query_knowledge(
            query=args.query,
            expert=args.expert,
            max_tokens=args.max_tokens,
            include_graph=not args.no_graph,
        )

        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            # Human-readable output
            print()
            print(result.get("answer", "(no answer)"))
            print()

            # Show sources
            sources = result.get("sources", [])
            if sources:
                print("--- Sources ---")
                for s in sources:
                    expert = s.get("expert", "unknown")
                    title = s.get("title", "")
                    score = s.get("score", 0.0)
                    print(f"  [{expert}] {title} (score: {score:.3f})")

            # Show graph context if present
            experts = result.get("experts_referenced", [])
            if experts:
                print(f"\nExperts referenced: {', '.join(experts)}")

            layers = result.get("layers_used", [])
            if layers:
                layer_names = {0: "Identity", 1: "Graph", 2: "Vector", 3: "Deep"}
                used = [layer_names.get(l, str(l)) for l in layers]
                print(f"Context layers: {' -> '.join(used)}")

    except httpx.ConnectError:
        print(f"Error: Cannot connect to NeuralForge at {API}", file=sys.stderr)
        print("Make sure the stack is running: docker compose up -d", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: {e.response.status_code} — {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
