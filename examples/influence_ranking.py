#!/usr/bin/env python3
"""Rank experts by influence on a topic.

Uses GPU-accelerated PageRank (via cuGraph) combined with topic-specific
edge counting to produce authority rankings. The score reflects both
structural importance in the knowledge graph and topical relevance.

Usage:
    python examples/influence_ranking.py --topic "quantization"
    python examples/influence_ranking.py --topic "transformer architectures" --top 20
    python examples/influence_ranking.py --topic "LoRA" --format json

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import json
import sys

import httpx

API = "http://localhost:8090"


def get_rankings(topic: str, top: int = 10) -> dict:
    """Fetch expert authority rankings for a topic."""
    params = {"topic": topic, "limit": top}
    resp = httpx.get(f"{API}/api/v1/graph/authority", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_pagerank() -> dict:
    """Fetch global PageRank scores."""
    resp = httpx.get(f"{API}/api/v1/graph/pagerank", timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Rank expert influence in NeuralForge")
    parser.add_argument("--topic", "-t", required=True, help="Topic to rank experts on")
    parser.add_argument("--top", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--global-rank", action="store_true", help="Also show global PageRank")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api

    try:
        result = get_rankings(args.topic, args.top)
        rankings = result.get("rankings", [])

        if args.format == "json":
            print(json.dumps(result, indent=2))
            return

        if not rankings:
            print(f"No expert rankings found for topic: {args.topic}")
            print("Tip: Ingest experts who cover this topic, then run discovery.")
            return

        print(f"\nExpert Authority Rankings: \"{args.topic}\"\n")
        print(f"  {'Rank':<6} {'Expert':<30} {'Score':<12} {'Edges':<8}")
        print(f"  {'----':<6} {'------':<30} {'-----':<12} {'-----':<8}")

        for i, r in enumerate(rankings, 1):
            name = r.get("expert_name", "unknown")
            score = r.get("score", 0.0)
            edges = r.get("edge_count", 0)
            print(f"  {i:<6} {name:<30} {score:<12.6f} {edges:<8}")

        if args.global_rank:
            print("\n--- Global PageRank ---\n")
            pr_result = get_pagerank()
            scores = pr_result.get("scores", {})
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for node_id, score in sorted_scores[:args.top]:
                name = pr_result.get("names", {}).get(node_id, node_id)
                print(f"  {name:<40} {score:.6f}")

    except httpx.ConnectError:
        print(f"Error: Cannot connect to NeuralForge at {API}", file=sys.stderr)
        print("Make sure the stack is running: docker compose up -d", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: {e.response.status_code} — {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
