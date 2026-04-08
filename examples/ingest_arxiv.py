#!/usr/bin/env python3
"""Ingest arXiv papers by author or search query.

Searches arXiv for papers, downloads PDFs, extracts text, chunks and
embeds through Triton, then links the author(s) as expert nodes in
the knowledge graph.

Usage:
    python examples/ingest_arxiv.py --author "Yann LeCun" --max-papers 20
    python examples/ingest_arxiv.py --query "quantization large language models" --expert "Quantization Research"

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import sys

import httpx

API = "http://localhost:8090"


def ingest_by_author(author: str, max_papers: int = 20, tags: list[str] | None = None) -> dict:
    """Submit an arXiv author search for ingestion."""
    payload = {
        "source_type": "arxiv",
        "source_url": f"https://arxiv.org/search/?query={author}&searchtype=author",
        "expert_name": author,
        "tags": tags or [],
        "options": {
            "search_type": "author",
            "search_query": author,
            "max_papers": max_papers,
            "download_pdfs": True,
        },
    }

    resp = httpx.post(f"{API}/api/v1/ingest", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def ingest_by_query(query: str, expert: str, max_papers: int = 20, tags: list[str] | None = None) -> dict:
    """Submit an arXiv topic search for ingestion."""
    payload = {
        "source_type": "arxiv",
        "source_url": f"https://arxiv.org/search/?query={query}&searchtype=all",
        "expert_name": expert,
        "tags": tags or [],
        "options": {
            "search_type": "query",
            "search_query": query,
            "max_papers": max_papers,
            "download_pdfs": True,
        },
    }

    resp = httpx.post(f"{API}/api/v1/ingest", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into NeuralForge")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--author", help="arXiv author name to search")
    group.add_argument("--query", help="arXiv search query (topic-based)")
    parser.add_argument("--expert", help="Expert name (defaults to author name)")
    parser.add_argument("--max-papers", type=int, default=20, help="Maximum papers to ingest (default: 20)")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    try:
        if args.author:
            expert = args.expert or args.author
            print(f"Searching arXiv for papers by: {args.author}")
            print(f"Expert name: {expert}")
            print(f"Max papers: {args.max_papers}")
            result = ingest_by_author(args.author, args.max_papers, tags)
        else:
            expert = args.expert or "Research"
            print(f"Searching arXiv for: {args.query}")
            print(f"Expert name: {expert}")
            print(f"Max papers: {args.max_papers}")
            result = ingest_by_query(args.query, expert, args.max_papers, tags)

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
