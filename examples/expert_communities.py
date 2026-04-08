#!/usr/bin/env python3
"""Discover expert clusters in your NeuralForge knowledge graph.

Runs the Louvain community detection algorithm (GPU-accelerated via
cuGraph on NVIDIA hardware) to find natural clusters of experts who
share topics, cite each other, or hold similar positions.

Usage:
    python examples/expert_communities.py
    python examples/expert_communities.py --format json
    python examples/expert_communities.py --min-size 2

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import json
import sys
from collections import defaultdict

import httpx

API = "http://localhost:8090"


def get_communities() -> dict:
    """Fetch community detection results from the graph API."""
    resp = httpx.get(f"{API}/api/v1/graph/communities", timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_experts() -> list[dict]:
    """Fetch all expert nodes."""
    resp = httpx.get(f"{API}/api/v1/experts", timeout=10)
    resp.raise_for_status()
    return resp.json().get("experts", [])


def main():
    parser = argparse.ArgumentParser(description="Discover expert communities in NeuralForge")
    parser.add_argument("--min-size", type=int, default=1, help="Minimum community size to display")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api

    try:
        community_data = get_communities()
        communities = community_data.get("communities", {})

        if args.format == "json":
            print(json.dumps(community_data, indent=2))
            return

        if not communities:
            print("No communities detected.")
            print("Tip: Ingest more experts with overlapping topics to enable community detection.")
            return

        # Group nodes by community
        grouped: dict[int, list[str]] = defaultdict(list)
        for node_id, community_id in communities.items():
            grouped[community_id].append(node_id)

        # Try to enrich with expert names
        try:
            experts = get_experts()
            name_map = {e["expert_id"]: e["name"] for e in experts}
        except Exception:
            name_map = {}

        # Sort communities by size (largest first)
        sorted_communities = sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True)

        print(f"\nDetected {len(sorted_communities)} community/communities "
              f"({sum(len(m) for m in grouped.values())} nodes total):\n")

        for community_id, members in sorted_communities:
            if len(members) < args.min_size:
                continue

            # Resolve names where possible
            display_names = [name_map.get(m, m) for m in members]
            display_names.sort()

            print(f"  Community {community_id} ({len(members)} members):")
            for name in display_names:
                print(f"    - {name}")
            print()

    except httpx.ConnectError:
        print(f"Error: Cannot connect to NeuralForge at {API}", file=sys.stderr)
        print("Make sure the stack is running: docker compose up -d", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: {e.response.status_code} — {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
