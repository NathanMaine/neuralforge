#!/usr/bin/env python3
"""Find where experts disagree in your NeuralForge knowledge base.

Queries the knowledge graph for contradiction and incompatibility edges,
surfacing places where ingested experts hold opposing views. This is one
of NeuralForge's most powerful features — instead of hiding disagreements,
it makes them visible and traceable.

Usage:
    python examples/find_contradictions.py
    python examples/find_contradictions.py --topic "quantization"
    python examples/find_contradictions.py --topic "fine-tuning" --format json

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import json
import sys

import httpx

API = "http://localhost:8090"


def find_contradictions(topic: str | None = None) -> dict:
    """Query the graph for contradictions."""
    params = {}
    if topic:
        params["topic"] = topic

    resp = httpx.get(f"{API}/api/v1/graph/contradictions", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Find expert disagreements in NeuralForge")
    parser.add_argument("--topic", "-t", help="Filter contradictions by topic")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api

    try:
        result = find_contradictions(args.topic)
        contradictions = result.get("contradictions", [])

        if args.format == "json":
            print(json.dumps(result, indent=2))
            return

        if not contradictions:
            if args.topic:
                print(f"No contradictions found for topic: {args.topic}")
            else:
                print("No contradictions found in the knowledge graph.")
            print("Tip: Ingest multiple experts on overlapping topics to discover disagreements.")
            return

        topic_label = f" on '{args.topic}'" if args.topic else ""
        print(f"\nFound {len(contradictions)} contradiction(s){topic_label}:\n")

        for i, c in enumerate(contradictions, 1):
            edge_a = c.get("edge_a", {})
            edge_b = c.get("edge_b", {})
            explanation = c.get("explanation", "")

            src_a = edge_a.get("source_name", edge_a.get("source_id", "?"))
            tgt_a = edge_a.get("target_name", edge_a.get("target_id", "?"))
            type_a = edge_a.get("edge_type", "?")

            src_b = edge_b.get("source_name", edge_b.get("source_id", "?"))
            tgt_b = edge_b.get("target_name", edge_b.get("target_id", "?"))
            type_b = edge_b.get("edge_type", "?")

            print(f"  {i}. {explanation}")
            print(f"     Edge A: {src_a} --[{type_a}]--> {tgt_a}")
            print(f"     Edge B: {src_b} --[{type_b}]--> {tgt_b}")
            conf = edge_a.get("confidence", 0)
            if conf:
                print(f"     Confidence: {conf:.2f}")
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
