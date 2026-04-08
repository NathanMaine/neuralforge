"""Ingest API routes -- document upload, blog scraping, auto-capture."""
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from forge.ingest.blog_scraper import add_source, load_sources, scrape_blog
from forge.ingest.chunker import chunk_text
from forge.ingest.conversation_miner import mine_conversation
from forge.ingest.document_loader import load_document, LOADERS
from forge.ingest.pii_scrubber import scrub_pii
from forge.ingest.upserter import ingest_chunks

router = APIRouter(prefix="/api/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)

# Auto-capture state
_auto_capture_enabled = False
_auto_capture_stats = {
    "enabled": False,
    "messages_captured": 0,
    "last_capture": None,
    "started_at": None,
}


# --- Request models ---

class BlogScrapeRequest(BaseModel):
    url: str
    name: str
    creator: str
    strategy: str = "auto"
    max_articles: int = 50


class AddSourceRequest(BaseModel):
    url: str
    name: str
    creator: str
    strategy: str = "auto"


class AutoCaptureRequest(BaseModel):
    messages: list[dict]
    creator: str = "auto-capture"
    title: str = ""
    source: str = "auto-capture"


# --- Document upload ---

@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    creator: str = Form("upload"),
    title: str = Form(""),
):
    """Upload and ingest a document (PDF, DOCX, TXT, HTML, CSV, MD)."""
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in LOADERS:
        supported = ", ".join(sorted(LOADERS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {supported}",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load document
        doc = load_document(tmp_path)
        doc_text = doc.get("text", "")
        doc_title = title or doc.get("title", file.filename or "untitled")

        if not doc_text.strip():
            raise HTTPException(status_code=400, detail="Document contains no extractable text")

        # PII scrub
        scrubbed_text, pii_counts = scrub_pii(doc_text)

        # Chunk
        chunks = chunk_text(scrubbed_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced no chunks")

        # Ingest
        ingested = await ingest_chunks(
            chunks=chunks,
            creator=creator,
            title=doc_title,
            source=file.filename or "upload",
            source_type="document",
        )

        from forge.api.routes.events import publish_event
        publish_event("ingest_complete", {
            "type": "document",
            "filename": file.filename,
            "chunks": len(chunks),
            "ingested": ingested,
        })

        return {
            "filename": file.filename,
            "title": doc_title,
            "creator": creator,
            "chunks": len(chunks),
            "ingested": ingested,
            "pii_scrubbed": pii_counts,
            "metadata": doc.get("metadata", {}),
        }
    finally:
        os.unlink(tmp_path)


# --- Blog scraping ---

@router.post("/blog")
async def trigger_blog_scrape(req: BlogScrapeRequest):
    """Trigger a blog scrape for the given URL."""
    source = {
        "url": req.url,
        "name": req.name,
        "creator": req.creator,
        "strategy": req.strategy,
    }
    result = await scrape_blog(source, max_articles=req.max_articles)

    from forge.api.routes.events import publish_event
    publish_event("ingest_complete", {
        "type": "blog",
        "source": req.name,
        "discovered": result.get("discovered", 0),
        "ingested": result.get("ingested", 0),
    })

    return result


# --- Source management ---

@router.get("/sources")
async def list_sources():
    """List all configured blog/content sources."""
    sources = load_sources()
    return {"sources": sources, "count": len(sources)}


@router.post("/sources")
async def create_source(req: AddSourceRequest):
    """Add a new blog/content source."""
    entry = add_source(
        url=req.url,
        name=req.name,
        creator=req.creator,
        strategy=req.strategy,
    )
    return {"source": entry, "message": "Source added successfully"}


# --- Conversation upload ---

@router.post("/conversations")
async def upload_conversation(
    file: UploadFile = File(...),
    creator: str = Form("conversation"),
    title: str = Form(""),
    fmt: Optional[str] = Form(None),
):
    """Upload and mine a conversation file (Claude, ChatGPT, Slack, JSONL, MD)."""
    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    result = await mine_conversation(
        text=text,
        creator=creator,
        title=title or file.filename or "conversation",
        source=file.filename or "upload",
        fmt=fmt,
    )

    from forge.api.routes.events import publish_event
    publish_event("ingest_complete", {
        "type": "conversation",
        "filename": file.filename,
        "messages": result.get("messages", 0),
        "ingested": result.get("ingested", 0),
    })

    return result


# --- Auto-capture ---

@router.post("/auto-capture")
async def auto_capture(req: AutoCaptureRequest):
    """Zero-config hook endpoint for capturing conversation messages.

    Accepts a list of messages and ingests them into the knowledge base.
    """
    global _auto_capture_enabled, _auto_capture_stats

    if not req.messages:
        return {"captured": 0, "message": "No messages to capture"}

    # Build text from messages
    text_parts = []
    for msg in req.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            text_parts.append(f"{role}: {content}")

    combined_text = "\n\n".join(text_parts)

    # PII scrub
    scrubbed, pii_counts = scrub_pii(combined_text)

    # Chunk and ingest
    chunks = chunk_text(scrubbed)
    ingested = 0
    if chunks:
        ingested = await ingest_chunks(
            chunks=chunks,
            creator=req.creator,
            title=req.title or "Auto-captured conversation",
            source=req.source,
            source_type="auto-capture",
        )

    _auto_capture_stats["messages_captured"] += len(req.messages)
    _auto_capture_stats["last_capture"] = datetime.now().isoformat()
    if not _auto_capture_stats["started_at"]:
        _auto_capture_stats["started_at"] = datetime.now().isoformat()
    _auto_capture_stats["enabled"] = True

    return {
        "captured": len(req.messages),
        "chunks": len(chunks),
        "ingested": ingested,
        "pii_scrubbed": pii_counts,
    }


@router.get("/auto-capture/status")
async def auto_capture_status():
    """Get the current status of the auto-capture system."""
    return _auto_capture_stats
