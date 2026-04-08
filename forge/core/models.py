"""Pydantic v2 data models for NeuralForge."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result from the knowledge base."""

    id: str = Field(..., description="Unique identifier for the result")
    title: str = Field(..., description="Title of the matched document")
    snippet: str = Field("", description="Relevant text excerpt")
    score: float = Field(0.0, description="Relevance score (0.0-1.0)")
    source: str = Field("", description="Source URL or file path")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ExpertSummary(BaseModel):
    """Summary of a domain expert / knowledge source."""

    expert_id: str = Field(..., description="Unique expert identifier")
    name: str = Field(..., description="Display name of the expert")
    domain: str = Field("", description="Primary knowledge domain")
    doc_count: int = Field(0, description="Number of indexed documents")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")


class DashboardState(BaseModel):
    """Current state for the NeuralForge dashboard."""

    total_documents: int = Field(0, description="Total documents in the knowledge base")
    total_experts: int = Field(0, description="Total registered experts")
    active_jobs: int = Field(0, description="Currently running ingest jobs")
    index_size_bytes: int = Field(0, description="Total index size in bytes")
    graph_node_count: int = Field(0, description="Number of nodes in knowledge graph")
    graph_edge_count: int = Field(0, description="Number of edges in knowledge graph")
    last_ingest: Optional[datetime] = Field(None, description="Timestamp of last ingestion")
    system_status: str = Field("idle", description="Overall system status")


class IngestJob(BaseModel):
    """Represents an ingestion job."""

    job_id: str = Field(..., description="Unique job identifier")
    source_url: str = Field("", description="URL or path being ingested")
    status: str = Field("pending", description="Job status: pending, running, completed, failed")
    doc_count: int = Field(0, description="Documents processed so far")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    file_type: str = Field("", description="Type of file being ingested")


class BlogSource(BaseModel):
    """A blog or content source for discovery and ingestion."""

    source_id: str = Field(..., description="Unique source identifier")
    name: str = Field(..., description="Display name of the source")
    url: str = Field(..., description="Base URL of the blog/source")
    feed_url: Optional[str] = Field(None, description="RSS/Atom feed URL if available")
    scrape_selector: Optional[str] = Field(None, description="CSS selector for content extraction")
    active: bool = Field(True, description="Whether this source is actively monitored")
    last_scraped: Optional[datetime] = Field(None, description="Last scrape timestamp")
    article_count: int = Field(0, description="Number of articles ingested")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
