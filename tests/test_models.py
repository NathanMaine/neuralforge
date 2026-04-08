"""Comprehensive tests for forge.core.models — 20+ tests for all Pydantic models."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from forge.core.models import (
    SearchResult,
    ExpertSummary,
    DashboardState,
    IngestJob,
    BlogSource,
)


class TestSearchResult:
    """Tests for the SearchResult model."""

    def test_minimal_construction(self):
        r = SearchResult(id="r1", title="Test Result")
        assert r.id == "r1"
        assert r.title == "Test Result"

    def test_defaults(self):
        r = SearchResult(id="r1", title="Test")
        assert r.snippet == ""
        assert r.score == 0.0
        assert r.source == ""
        assert r.metadata == {}

    def test_full_construction(self):
        r = SearchResult(
            id="r2",
            title="Full Result",
            snippet="some text",
            score=0.95,
            source="https://example.com",
            metadata={"author": "test"},
        )
        assert r.score == 0.95
        assert r.metadata["author"] == "test"

    def test_serialization(self):
        r = SearchResult(id="r1", title="Test")
        data = r.model_dump()
        assert isinstance(data, dict)
        assert data["id"] == "r1"
        assert "metadata" in data

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            SearchResult()

    def test_missing_id(self):
        with pytest.raises(ValidationError):
            SearchResult(title="No ID")


class TestExpertSummary:
    """Tests for the ExpertSummary model."""

    def test_minimal_construction(self):
        e = ExpertSummary(expert_id="e1", name="Dr. Smith")
        assert e.expert_id == "e1"
        assert e.name == "Dr. Smith"

    def test_defaults(self):
        e = ExpertSummary(expert_id="e1", name="Test")
        assert e.domain == ""
        assert e.doc_count == 0
        assert e.last_updated is None
        assert e.tags == []

    def test_with_tags(self):
        e = ExpertSummary(expert_id="e1", name="Test", tags=["ml", "nlp"])
        assert len(e.tags) == 2
        assert "ml" in e.tags

    def test_with_datetime(self):
        now = datetime.now()
        e = ExpertSummary(expert_id="e1", name="Test", last_updated=now)
        assert e.last_updated == now

    def test_serialization_round_trip(self):
        e = ExpertSummary(expert_id="e1", name="Test", domain="ML", doc_count=42)
        data = e.model_dump()
        e2 = ExpertSummary(**data)
        assert e2.expert_id == e.expert_id
        assert e2.doc_count == 42


class TestDashboardState:
    """Tests for the DashboardState model."""

    def test_all_defaults(self):
        d = DashboardState()
        assert d.total_documents == 0
        assert d.total_experts == 0
        assert d.active_jobs == 0
        assert d.index_size_bytes == 0
        assert d.graph_node_count == 0
        assert d.graph_edge_count == 0
        assert d.last_ingest is None
        assert d.system_status == "idle"

    def test_custom_values(self):
        d = DashboardState(
            total_documents=1000,
            total_experts=5,
            active_jobs=2,
            system_status="running",
        )
        assert d.total_documents == 1000
        assert d.system_status == "running"

    def test_serialization(self):
        d = DashboardState(total_documents=50)
        data = d.model_dump()
        assert data["total_documents"] == 50
        assert "system_status" in data


class TestIngestJob:
    """Tests for the IngestJob model."""

    def test_minimal_construction(self):
        j = IngestJob(job_id="j1")
        assert j.job_id == "j1"

    def test_defaults(self):
        j = IngestJob(job_id="j1")
        assert j.source_url == ""
        assert j.status == "pending"
        assert j.doc_count == 0
        assert j.error_message is None
        assert j.completed_at is None
        assert j.file_type == ""

    def test_created_at_auto_set(self):
        j = IngestJob(job_id="j1")
        assert isinstance(j.created_at, datetime)

    def test_failed_job(self):
        j = IngestJob(
            job_id="j2",
            status="failed",
            error_message="Connection timeout",
        )
        assert j.status == "failed"
        assert j.error_message == "Connection timeout"

    def test_completed_job(self):
        now = datetime.now()
        j = IngestJob(
            job_id="j3",
            status="completed",
            doc_count=42,
            completed_at=now,
        )
        assert j.doc_count == 42
        assert j.completed_at == now

    def test_missing_job_id(self):
        with pytest.raises(ValidationError):
            IngestJob()

    def test_serialization(self):
        j = IngestJob(job_id="j1", source_url="https://example.com")
        data = j.model_dump()
        assert data["job_id"] == "j1"
        assert data["source_url"] == "https://example.com"


class TestBlogSource:
    """Tests for the BlogSource model."""

    def test_minimal_construction(self):
        b = BlogSource(source_id="b1", name="Tech Blog", url="https://blog.example.com")
        assert b.source_id == "b1"
        assert b.name == "Tech Blog"
        assert b.url == "https://blog.example.com"

    def test_defaults(self):
        b = BlogSource(source_id="b1", name="Test", url="https://test.com")
        assert b.feed_url is None
        assert b.scrape_selector is None
        assert b.active is True
        assert b.last_scraped is None
        assert b.article_count == 0
        assert b.tags == []

    def test_with_feed_url(self):
        b = BlogSource(
            source_id="b1",
            name="Test",
            url="https://test.com",
            feed_url="https://test.com/rss",
        )
        assert b.feed_url == "https://test.com/rss"

    def test_inactive_source(self):
        b = BlogSource(source_id="b1", name="Test", url="https://test.com", active=False)
        assert b.active is False

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            BlogSource(source_id="b1")

    def test_missing_url(self):
        with pytest.raises(ValidationError):
            BlogSource(source_id="b1", name="Test")

    def test_serialization(self):
        b = BlogSource(
            source_id="b1",
            name="Test",
            url="https://test.com",
            tags=["tech", "ai"],
            article_count=100,
        )
        data = b.model_dump()
        assert data["article_count"] == 100
        assert len(data["tags"]) == 2

    def test_json_round_trip(self):
        b = BlogSource(source_id="b1", name="Test", url="https://test.com")
        json_str = b.model_dump_json()
        b2 = BlogSource.model_validate_json(json_str)
        assert b2.source_id == b.source_id
        assert b2.url == b.url
