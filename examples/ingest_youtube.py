#!/usr/bin/env python3
"""Ingest a YouTube channel into NeuralForge.

Submits a YouTube channel for ingestion — NeuralForge will discover videos,
extract transcripts, chunk them, embed via Triton, and build knowledge graph
edges linking the creator to concepts discussed.

Usage:
    python examples/ingest_youtube.py --channel "3Blue1Brown" --name "Grant Sanderson"
    python examples/ingest_youtube.py --channel "AndrejKarpathy" --name "Andrej Karpathy" --tags ml,transformers

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import sys

import httpx

API = "http://localhost:8090"


def ingest_channel(channel: str, name: str, tags: list[str] | None = None) -> dict:
    """Submit a YouTube channel for ingestion."""
    payload = {
        "source_type": "youtube",
        "source_url": f"https://www.youtube.com/@{channel}",
        "expert_name": name,
        "tags": tags or [],
        "options": {
            "max_videos": 50,
            "extract_transcripts": True,
            "chunk_strategy": "semantic",
        },
    }

    resp = httpx.post(f"{API}/api/v1/ingest", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def check_status(job_id: str) -> dict:
    """Check the status of an ingestion job."""
    resp = httpx.get(f"{API}/api/v1/ingest/{job_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Ingest a YouTube channel into NeuralForge")
    parser.add_argument("--channel", required=True, help="YouTube channel handle (e.g. 3Blue1Brown)")
    parser.add_argument("--name", required=True, help="Expert name for the knowledge graph")
    parser.add_argument("--tags", default="", help="Comma-separated tags (e.g. math,visualization)")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    print(f"Ingesting YouTube channel: @{args.channel}")
    print(f"Expert name: {args.name}")
    if tags:
        print(f"Tags: {', '.join(tags)}")

    try:
        result = ingest_channel(args.channel, args.name, tags)
        job_id = result.get("job_id", "unknown")
        print(f"\nJob submitted: {job_id}")
        print(f"Status: {result.get('status', 'pending')}")
        print(f"\nTrack progress:")
        print(f"  curl {API}/api/v1/ingest/{job_id}")
    except httpx.ConnectError:
        print(f"Error: Cannot connect to NeuralForge at {API}", file=sys.stderr)
        print("Make sure the stack is running: docker compose up -d", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: {e.response.status_code} — {e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
