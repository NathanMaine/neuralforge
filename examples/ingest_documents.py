#!/usr/bin/env python3
"""Ingest a folder of documents into NeuralForge.

Uploads PDF, DOCX, TXT, HTML, CSV, and Markdown files from a local
directory. Each document is chunked, embedded via Triton, stored in
Qdrant, and linked to the expert node in the knowledge graph.

Usage:
    python examples/ingest_documents.py --folder "./papers" --expert "Research Team"
    python examples/ingest_documents.py --folder "/data/reports" --expert "Jane Smith" --recursive

Requires: NeuralForge stack running (docker compose up -d)
"""
import argparse
import sys
from pathlib import Path

import httpx

API = "http://localhost:8090"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".json"}


def upload_document(file_path: Path, expert: str, tags: list[str] | None = None) -> dict:
    """Upload a single document for ingestion."""
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        data = {
            "expert_name": expert,
            "tags": ",".join(tags or []),
        }
        resp = httpx.post(f"{API}/api/v1/ingest/upload", files=files, data=data, timeout=60)
        resp.raise_for_status()
        return resp.json()


def find_documents(folder: Path, recursive: bool = False) -> list[Path]:
    """Find supported documents in a folder."""
    pattern = "**/*" if recursive else "*"
    docs = []
    for ext in SUPPORTED_EXTENSIONS:
        docs.extend(folder.glob(f"{pattern}{ext}"))
    return sorted(docs)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into NeuralForge")
    parser.add_argument("--folder", required=True, help="Path to document folder")
    parser.add_argument("--expert", required=True, help="Expert name for the knowledge graph")
    parser.add_argument("--recursive", action="store_true", help="Search subfolders recursively")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--api", default=API, help="NeuralForge API URL")
    args = parser.parse_args()

    global API
    API = args.api
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {args.folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    docs = find_documents(folder, args.recursive)
    if not docs:
        print(f"No supported documents found in {args.folder}")
        print(f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(0)

    print(f"Found {len(docs)} documents in {args.folder}")
    print(f"Expert: {args.expert}")
    print()

    success = 0
    failed = 0
    for i, doc in enumerate(docs, 1):
        try:
            print(f"  [{i}/{len(docs)}] {doc.name}...", end=" ", flush=True)
            result = upload_document(doc, args.expert, tags)
            print(f"OK (job: {result.get('job_id', '?')})")
            success += 1
        except httpx.ConnectError:
            print(f"\nError: Cannot connect to NeuralForge at {API}", file=sys.stderr)
            print("Make sure the stack is running: docker compose up -d", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            print(f"FAILED ({e.response.status_code})")
            failed += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1

    print(f"\nDone: {success} uploaded, {failed} failed")


if __name__ == "__main__":
    main()
