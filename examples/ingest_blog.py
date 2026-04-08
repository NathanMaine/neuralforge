#!/usr/bin/env python3
"""Ingest a blog or website into NeuralForge.

Crawls a blog, extracts article content via trafilatura, chunks with
semantic boundaries, embeds through Triton, and adds the author as an
expert node in the knowledge graph.

Usage:
    python examples/ingest_blog.py --url "https://timdettmers.com" --expert "Tim Dettmers"
    python examples/ingest_blog.py --url "https://karpathy.github.io" --expert "Andrej Karpathy" --max-pages 100

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import sys

import httpx

API = "http://localhost:8090"


def ingest_blog(url: str, expert: str, max_pages: int = 50, tags: list[str] | None = None) -> dict:
    """Submit a blog for crawling and ingestion."""
    payload = {
        "source_type": "blog",
        "source_url": url,
        "expert_name": expert,
        "tags": tags or [],
        "options": {
            "max_pages": max_pages,
            "extract_links": True,
            "respect_robots_txt": True,
            "request_delay": 1.5,
        },
    }

    resp = httpx.post(f"{API}/api/v1/ingest", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Ingest a blog into NeuralForge")
    parser.add_argument("--url", required=True, help="Blog URL to crawl")
    parser.add_argument("--expert", required=True, help="Expert name for the knowledge graph")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to crawl (default: 50)")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    print(f"Ingesting blog: {args.url}")
    print(f"Expert: {args.expert}")
    print(f"Max pages: {args.max_pages}")

    try:
        result = ingest_blog(args.url, args.expert, args.max_pages, tags)
        job_id = result.get("job_id", "unknown")
        print(f"\nJob submitted: {job_id}")
        print(f"Status: {result.get('status', 'pending')}")
        print(f"Estimated articles: {result.get('estimated_count', 'unknown')}")
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
