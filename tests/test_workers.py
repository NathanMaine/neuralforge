"""Comprehensive tests for forge.workers -- 15+ tests.

Tests scheduler lifecycle, discovery worker, and scrape worker.
All external dependencies (Qdrant, NIM, blog scraper) are mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from forge.workers.scheduler import (
    get_scheduler,
    start_scheduler,
    stop_scheduler,
    is_running,
    get_jobs,
    _scheduler,
)
from forge.workers.discovery_worker import run_discovery
from forge.workers.scrape_worker import run_scrape


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestScheduler:
    """Tests for the scheduler module."""

    def setup_method(self):
        """Reset scheduler state before each test."""
        stop_scheduler()

    def teardown_method(self):
        """Clean up scheduler after each test."""
        stop_scheduler()

    def test_get_scheduler_singleton(self):
        """get_scheduler returns the same instance."""
        s1 = get_scheduler()
        s2 = get_scheduler()
        assert s1 is s2

    def test_is_running_initially_false(self):
        """Scheduler is not running initially."""
        # After stop_scheduler in setup, _scheduler is None
        assert not is_running()

    def test_get_jobs_empty(self):
        """No jobs when scheduler has not been started."""
        assert get_jobs() == []

    @patch("forge.workers.scheduler.AsyncIOScheduler")
    def test_start_scheduler(self, MockScheduler):
        """start_scheduler configures and starts."""
        import forge.workers.scheduler as sched_module

        mock_instance = MagicMock()
        mock_instance.running = False
        mock_instance.get_jobs.return_value = []
        MockScheduler.return_value = mock_instance

        # Reset the singleton
        sched_module._scheduler = None

        scheduler = start_scheduler()

        mock_instance.add_job.assert_called()
        mock_instance.start.assert_called_once()
        # Should have added 2 jobs (discovery + scrape)
        assert mock_instance.add_job.call_count == 2

    @patch("forge.workers.scheduler.AsyncIOScheduler")
    def test_start_scheduler_already_running(self, MockScheduler):
        """start_scheduler returns existing if already running."""
        import forge.workers.scheduler as sched_module

        mock_instance = MagicMock()
        mock_instance.running = True
        MockScheduler.return_value = mock_instance
        sched_module._scheduler = mock_instance

        scheduler = start_scheduler()

        # Should not add jobs or start again
        mock_instance.add_job.assert_not_called()
        mock_instance.start.assert_not_called()

    def test_stop_scheduler_when_not_running(self):
        """stop_scheduler is safe when nothing is running."""
        stop_scheduler()  # Should not raise

    @patch("forge.workers.scheduler.AsyncIOScheduler")
    def test_stop_scheduler_when_running(self, MockScheduler):
        """stop_scheduler shuts down a running scheduler."""
        import forge.workers.scheduler as sched_module

        mock_instance = MagicMock()
        mock_instance.running = True
        sched_module._scheduler = mock_instance

        stop_scheduler()

        mock_instance.shutdown.assert_called_once_with(wait=False)

    @patch("forge.workers.scheduler.AsyncIOScheduler")
    def test_get_jobs_with_jobs(self, MockScheduler):
        """get_jobs returns job details."""
        import forge.workers.scheduler as sched_module

        mock_job = MagicMock()
        mock_job.id = "test_job"
        mock_job.name = "Test Job"
        mock_job.next_run_time = "2024-01-01T00:00:00"
        mock_job.trigger = "interval[hours=6]"

        mock_instance = MagicMock()
        mock_instance.get_jobs.return_value = [mock_job]
        sched_module._scheduler = mock_instance

        jobs = get_jobs()

        assert len(jobs) == 1
        assert jobs[0]["id"] == "test_job"
        assert jobs[0]["name"] == "Test Job"


# ---------------------------------------------------------------------------
# Discovery Worker
# ---------------------------------------------------------------------------


class TestDiscoveryWorker:
    """Tests for discovery_worker.run_discovery."""

    @pytest.mark.asyncio
    @patch("forge.workers.discovery_worker.get_all_expert_names", return_value=["alice"])
    async def test_not_enough_experts(self, mock_experts):
        """Returns zeros when less than 2 experts."""
        result = await run_discovery()
        assert result["pairs_checked"] == 0
        assert result["relationships_found"] == 0

    @pytest.mark.asyncio
    @patch("forge.workers.discovery_worker.discover_pair", new_callable=AsyncMock)
    @patch("forge.workers.discovery_worker.get_shared_topics", return_value=["machine learning"])
    @patch("forge.workers.discovery_worker.get_all_expert_names", return_value=["alice", "bob"])
    async def test_discovery_finds_relationship(self, mock_experts, mock_topics, mock_discover):
        """Discovers relationship between expert pair."""
        mock_discover.return_value = {
            "expert_a": "alice",
            "expert_b": "bob",
            "topic": "machine learning",
            "relationship": "agrees",
            "confidence": 0.85,
            "summary": "Both agree on ML approaches.",
        }

        result = await run_discovery()
        assert result["pairs_checked"] == 1
        assert result["relationships_found"] == 1

    @pytest.mark.asyncio
    @patch("forge.workers.discovery_worker.get_shared_topics", return_value=[])
    @patch("forge.workers.discovery_worker.get_all_expert_names", return_value=["alice", "bob"])
    async def test_no_shared_topics(self, mock_experts, mock_topics):
        """No shared topics means no discoveries."""
        result = await run_discovery()
        assert result["pairs_checked"] == 1
        assert result["relationships_found"] == 0

    @pytest.mark.asyncio
    @patch("forge.workers.discovery_worker.discover_pair", new_callable=AsyncMock, return_value=None)
    @patch("forge.workers.discovery_worker.get_shared_topics", return_value=["topic"])
    @patch("forge.workers.discovery_worker.get_all_expert_names", return_value=["a", "b"])
    async def test_discovery_returns_none(self, mock_experts, mock_topics, mock_discover):
        """discover_pair returning None is handled."""
        result = await run_discovery()
        assert result["pairs_checked"] == 1
        assert result["relationships_found"] == 0

    @pytest.mark.asyncio
    @patch("forge.workers.discovery_worker.get_shared_topics", side_effect=Exception("network error"))
    @patch("forge.workers.discovery_worker.get_all_expert_names", return_value=["a", "b"])
    async def test_discovery_handles_errors(self, mock_experts, mock_topics):
        """Errors during discovery are counted."""
        result = await run_discovery()
        assert result["errors"] == 1


# ---------------------------------------------------------------------------
# Scrape Worker
# ---------------------------------------------------------------------------


class TestScrapeWorker:
    """Tests for scrape_worker.run_scrape."""

    @pytest.mark.asyncio
    @patch("forge.workers.scrape_worker.load_sources", return_value=[])
    async def test_no_sources(self, mock_load):
        """No sources returns zeros."""
        result = await run_scrape()
        assert result["sources_processed"] == 0
        assert result["total_ingested"] == 0

    @pytest.mark.asyncio
    @patch("forge.workers.scrape_worker.scrape_blog", new_callable=AsyncMock)
    @patch("forge.workers.scrape_worker.load_sources")
    async def test_scrape_one_source(self, mock_load, mock_scrape):
        """Scrapes a single source successfully."""
        mock_load.return_value = [
            {"url": "https://a.com", "name": "Test", "creator": "alice"}
        ]
        mock_scrape.return_value = {
            "source": "Test",
            "discovered": 10,
            "extracted": 5,
            "ingested": 15,
            "skipped": 5,
        }

        result = await run_scrape()
        assert result["sources_processed"] == 1
        assert result["total_discovered"] == 10
        assert result["total_ingested"] == 15

    @pytest.mark.asyncio
    @patch("forge.workers.scrape_worker.scrape_blog", new_callable=AsyncMock, side_effect=Exception("oops"))
    @patch("forge.workers.scrape_worker.load_sources")
    async def test_scrape_handles_errors(self, mock_load, mock_scrape):
        """Errors during scraping are counted."""
        mock_load.return_value = [
            {"url": "https://fail.com", "name": "Fail", "creator": "bob"}
        ]

        result = await run_scrape()
        assert result["errors"] == 1
        assert result["sources_processed"] == 0
