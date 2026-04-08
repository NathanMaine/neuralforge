"""Comprehensive tests for forge.ingest.blog_scraper -- 25+ tests.

All HTTP requests and trafilatura calls are mocked.
"""
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from forge.ingest.blog_scraper import (
    _discover_sitemap,
    _discover_rss,
    _discover_crawl,
    discover_articles,
    extract_article,
    scrape_blog,
    load_sources,
    add_source,
    _seen_hashes,
)


# ---------------------------------------------------------------------------
# Discovery -- sitemap
# ---------------------------------------------------------------------------


class TestDiscoverSitemap:
    """Tests for _discover_sitemap."""

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_sitemap_basic(self, mock_get):
        """Finds URLs from a basic sitemap.xml."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = """<?xml version="1.0"?>
        <urlset>
            <url><loc>https://blog.example.com/post-1</loc></url>
            <url><loc>https://blog.example.com/post-2</loc></url>
        </urlset>"""
        mock_get.return_value = mock_resp

        urls = _discover_sitemap("https://blog.example.com")
        assert len(urls) >= 2
        assert "https://blog.example.com/post-1" in urls

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_sitemap_not_found(self, mock_get):
        """Returns empty list when sitemap is not found."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_get.return_value = mock_resp

        urls = _discover_sitemap("https://blog.example.com")
        assert urls == []

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_sitemap_with_robots_txt(self, mock_get):
        """Discovers sitemap URL from robots.txt."""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "robots.txt" in url:
                resp.ok = True
                resp.text = "Sitemap: https://blog.example.com/custom-sitemap.xml"
            elif "custom-sitemap" in url:
                resp.ok = True
                resp.text = "<urlset><url><loc>https://blog.example.com/article-1</loc></url></urlset>"
            else:
                resp.ok = False
            return resp

        mock_get.side_effect = side_effect
        urls = _discover_sitemap("https://blog.example.com")
        assert any("article-1" in u for u in urls)

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_sitemap_request_error(self, mock_get):
        """Returns empty list on request error."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection error")
        urls = _discover_sitemap("https://blog.example.com")
        assert urls == []


# ---------------------------------------------------------------------------
# Discovery -- RSS
# ---------------------------------------------------------------------------


class TestDiscoverRss:
    """Tests for _discover_rss."""

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_rss_basic(self, mock_get):
        """Finds URLs from an RSS feed."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = """<?xml version="1.0"?>
        <rss>
            <channel>
                <item><link>https://blog.example.com/rss-post-1</link></item>
                <item><link>https://blog.example.com/rss-post-2</link></item>
            </channel>
        </rss>"""
        mock_get.return_value = mock_resp

        urls = _discover_rss("https://blog.example.com")
        assert len(urls) >= 2

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_rss_atom_feed(self, mock_get):
        """Finds URLs from Atom feed href attributes."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = '''<feed>
            <entry>
                <link href="https://blog.example.com/atom-1" />
            </entry>
        </feed>'''
        mock_get.return_value = mock_resp

        urls = _discover_rss("https://blog.example.com")
        assert any("atom-1" in u for u in urls)

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_rss_not_found(self, mock_get):
        """Returns empty list when no feeds found."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_get.return_value = mock_resp

        urls = _discover_rss("https://blog.example.com")
        assert urls == []


# ---------------------------------------------------------------------------
# Discovery -- crawl
# ---------------------------------------------------------------------------


class TestDiscoverCrawl:
    """Tests for _discover_crawl."""

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_crawl_finds_blog_links(self, mock_get):
        """Discovers blog links from homepage."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = '''<html><body>
            <a href="/blog/first-post">First</a>
            <a href="/blog/second-post">Second</a>
            <a href="/about">About</a>
        </body></html>'''
        mock_get.return_value = mock_resp

        urls = _discover_crawl("https://blog.example.com")
        assert len(urls) == 2
        assert all("/blog/" in u for u in urls)

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_crawl_year_pattern(self, mock_get):
        """Discovers URLs with year patterns."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = '''<html><body>
            <a href="/2024/01/great-post">Great Post</a>
            <a href="/contact">Contact</a>
        </body></html>'''
        mock_get.return_value = mock_resp

        urls = _discover_crawl("https://blog.example.com")
        assert len(urls) == 1
        assert "/2024/" in urls[0]

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_crawl_filters_external_links(self, mock_get):
        """External links are excluded."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = '''<html><body>
            <a href="https://other.com/blog/post">External</a>
            <a href="/blog/local">Local</a>
        </body></html>'''
        mock_get.return_value = mock_resp

        urls = _discover_crawl("https://blog.example.com")
        assert len(urls) == 1
        assert "blog.example.com" in urls[0]

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_crawl_page_down(self, mock_get):
        """Returns empty list when page is unreachable."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_get.return_value = mock_resp

        urls = _discover_crawl("https://blog.example.com")
        assert urls == []


# ---------------------------------------------------------------------------
# discover_articles (unified)
# ---------------------------------------------------------------------------


class TestDiscoverArticles:
    """Tests for discover_articles."""

    @patch("forge.ingest.blog_scraper._discover_crawl", return_value=["https://a.com/blog/1"])
    @patch("forge.ingest.blog_scraper._discover_rss", return_value=["https://a.com/blog/2"])
    @patch("forge.ingest.blog_scraper._discover_sitemap", return_value=["https://a.com/blog/3"])
    def test_auto_strategy_merges(self, mock_sm, mock_rss, mock_crawl):
        """Auto strategy merges results from all strategies."""
        urls = discover_articles("https://a.com", strategy="auto")
        assert len(urls) == 3

    @patch("forge.ingest.blog_scraper._discover_crawl", return_value=["https://a.com/blog/1"])
    @patch("forge.ingest.blog_scraper._discover_rss", return_value=["https://a.com/blog/1"])
    @patch("forge.ingest.blog_scraper._discover_sitemap", return_value=["https://a.com/blog/1"])
    def test_deduplication(self, mock_sm, mock_rss, mock_crawl):
        """Duplicate URLs are removed."""
        urls = discover_articles("https://a.com", strategy="auto")
        assert len(urls) == 1

    @patch("forge.ingest.blog_scraper._discover_sitemap", return_value=["https://a.com/1"])
    def test_sitemap_strategy(self, mock_sm):
        """Sitemap-only strategy calls sitemap discovery."""
        urls = discover_articles("https://a.com", strategy="sitemap")
        assert len(urls) == 1
        mock_sm.assert_called_once()

    @patch("forge.ingest.blog_scraper._discover_rss", return_value=["https://a.com/1"])
    def test_rss_strategy(self, mock_rss):
        """RSS-only strategy calls RSS discovery."""
        urls = discover_articles("https://a.com", strategy="rss")
        assert len(urls) == 1
        mock_rss.assert_called_once()

    @patch("forge.ingest.blog_scraper._discover_crawl", return_value=["https://a.com/1"])
    def test_crawl_strategy(self, mock_crawl):
        """Crawl-only strategy calls crawl discovery."""
        urls = discover_articles("https://a.com", strategy="crawl")
        assert len(urls) == 1
        mock_crawl.assert_called_once()

    @patch("forge.ingest.blog_scraper._discover_crawl", return_value=[])
    @patch("forge.ingest.blog_scraper._discover_rss", return_value=[])
    @patch("forge.ingest.blog_scraper._discover_sitemap", return_value=[])
    def test_unknown_strategy_falls_back(self, mock_sm, mock_rss, mock_crawl):
        """Unknown strategy falls back to auto."""
        urls = discover_articles("https://a.com", strategy="unknown_stuff")
        assert urls == []
        # All three should have been called (auto fallback)
        mock_sm.assert_called()
        mock_rss.assert_called()
        mock_crawl.assert_called()

    @patch("forge.ingest.blog_scraper._discover_sitemap")
    def test_trailing_slash_stripped(self, mock_sm):
        """Trailing slashes on URLs are normalized."""
        mock_sm.return_value = [
            "https://a.com/blog/1/",
            "https://a.com/blog/1",
        ]
        urls = discover_articles("https://a.com/", strategy="sitemap")
        assert len(urls) == 1


# ---------------------------------------------------------------------------
# extract_article
# ---------------------------------------------------------------------------


class TestExtractArticle:
    """Tests for extract_article."""

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_extract_success(self, mock_get):
        """Successful extraction returns article dict."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>Test Title</title></head><body><p>Article content here.</p></body></html>"
        mock_get.return_value = mock_resp

        with patch("trafilatura.extract", return_value="Article content here."):
            with patch("trafilatura.extract_metadata") as mock_meta:
                mock_meta_obj = MagicMock()
                mock_meta_obj.title = "Test Title"
                mock_meta.return_value = mock_meta_obj

                result = extract_article("https://blog.example.com/post-1")

        assert result is not None
        assert result["text"] == "Article content here."
        assert result["title"] == "Test Title"
        assert result["url"] == "https://blog.example.com/post-1"
        assert "extracted_at" in result

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_extract_http_error(self, mock_get):
        """Returns None on HTTP error."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = extract_article("https://blog.example.com/missing")
        assert result is None

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_extract_empty_content(self, mock_get):
        """Returns None when trafilatura extracts no text."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.text = "<html></html>"
        mock_get.return_value = mock_resp

        with patch("trafilatura.extract", return_value=None):
            result = extract_article("https://blog.example.com/empty")

        assert result is None

    @patch("forge.ingest.blog_scraper.requests.get")
    def test_extract_request_exception(self, mock_get):
        """Returns None on request exception."""
        import requests
        mock_get.side_effect = requests.RequestException("timeout")
        result = extract_article("https://blog.example.com/timeout")
        assert result is None


# ---------------------------------------------------------------------------
# scrape_blog (full pipeline)
# ---------------------------------------------------------------------------


class TestScrapeBlog:
    """Tests for scrape_blog."""

    @pytest.fixture(autouse=True)
    def clear_seen_hashes(self):
        """Clear the dedup set before each test."""
        _seen_hashes.clear()
        yield
        _seen_hashes.clear()

    @pytest.mark.asyncio
    @patch("forge.ingest.blog_scraper.ingest_chunks", new_callable=AsyncMock, return_value=3)
    @patch("forge.ingest.blog_scraper.extract_article")
    @patch("forge.ingest.blog_scraper.discover_articles")
    @patch("forge.ingest.blog_scraper.time.sleep")
    async def test_full_pipeline(self, mock_sleep, mock_discover, mock_extract, mock_ingest):
        """Full pipeline discovers, extracts, and ingests."""
        mock_discover.return_value = ["https://a.com/blog/1"]
        mock_extract.return_value = {
            "url": "https://a.com/blog/1",
            "title": "Test Post",
            "text": "Some article content about machine learning. " * 20,
            "extracted_at": "2024-01-01T00:00:00",
        }

        source = {"url": "https://a.com", "name": "Test Blog", "creator": "alice"}
        result = await scrape_blog(source)

        assert result["source"] == "Test Blog"
        assert result["discovered"] == 1
        assert result["extracted"] == 1
        assert result["ingested"] == 3

    @pytest.mark.asyncio
    @patch("forge.ingest.blog_scraper.discover_articles", return_value=[])
    async def test_no_articles_discovered(self, mock_discover):
        """Returns zeros when no articles are discovered."""
        source = {"url": "https://empty.com", "name": "Empty", "creator": "bob"}
        result = await scrape_blog(source)

        assert result["discovered"] == 0
        assert result["extracted"] == 0
        assert result["ingested"] == 0

    @pytest.mark.asyncio
    @patch("forge.ingest.blog_scraper.extract_article", return_value=None)
    @patch("forge.ingest.blog_scraper.discover_articles", return_value=["https://a.com/1"])
    @patch("forge.ingest.blog_scraper.time.sleep")
    async def test_extraction_failure_skipped(self, mock_sleep, mock_discover, mock_extract):
        """Failed extractions are counted as skipped."""
        source = {"url": "https://a.com", "name": "Test", "creator": "alice"}
        result = await scrape_blog(source)

        assert result["skipped"] == 1
        assert result["extracted"] == 0

    @pytest.mark.asyncio
    @patch("forge.ingest.blog_scraper.ingest_chunks", new_callable=AsyncMock, return_value=2)
    @patch("forge.ingest.blog_scraper.extract_article")
    @patch("forge.ingest.blog_scraper.discover_articles")
    @patch("forge.ingest.blog_scraper.time.sleep")
    async def test_pii_scrubbing_applied(self, mock_sleep, mock_discover, mock_extract, mock_ingest):
        """PII is scrubbed before ingestion."""
        mock_discover.return_value = ["https://a.com/1"]
        mock_extract.return_value = {
            "url": "https://a.com/1",
            "title": "PII Post",
            "text": "Contact alice@example.com or call 555-123-4567. " * 20,
            "extracted_at": "2024-01-01",
        }

        source = {"url": "https://a.com", "name": "PII Blog", "creator": "alice"}
        await scrape_blog(source)

        # Check that ingested text has PII redacted
        call_args = mock_ingest.call_args
        chunks = call_args[1]["chunks"] if "chunks" in call_args[1] else call_args[0][0]
        for chunk in chunks:
            assert "alice@example.com" not in chunk
            assert "555-123-4567" not in chunk

    @pytest.mark.asyncio
    @patch("forge.ingest.blog_scraper.ingest_chunks", new_callable=AsyncMock, return_value=2)
    @patch("forge.ingest.blog_scraper.extract_article")
    @patch("forge.ingest.blog_scraper.discover_articles")
    @patch("forge.ingest.blog_scraper.time.sleep")
    async def test_dedup_by_content_hash(self, mock_sleep, mock_discover, mock_extract, mock_ingest):
        """Duplicate content is skipped via content hash."""
        same_text = "Identical article content. " * 20
        mock_discover.return_value = ["https://a.com/1", "https://a.com/2"]
        mock_extract.return_value = {
            "url": "https://a.com/1",
            "title": "Post",
            "text": same_text,
            "extracted_at": "2024-01-01",
        }

        source = {"url": "https://a.com", "name": "Dup Blog", "creator": "alice"}
        result = await scrape_blog(source)

        assert result["extracted"] == 1
        assert result["skipped"] == 1


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


class TestSourceManagement:
    """Tests for load_sources and add_source."""

    @patch("forge.ingest.blog_scraper._SOURCES_PATH")
    def test_load_empty_sources(self, mock_path):
        """Empty sources.json returns empty list."""
        mock_path.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="[]")):
            sources = load_sources()
        assert sources == []

    @patch("forge.ingest.blog_scraper._SOURCES_PATH")
    def test_load_missing_file(self, mock_path):
        """Missing file returns empty list."""
        mock_path.exists.return_value = False
        sources = load_sources()
        assert sources == []

    @patch("forge.ingest.blog_scraper._SOURCES_PATH")
    def test_load_invalid_json(self, mock_path):
        """Invalid JSON returns empty list."""
        mock_path.exists.return_value = True
        with patch("builtins.open", mock_open(read_data="not json")):
            sources = load_sources()
        assert sources == []

    @patch("forge.ingest.blog_scraper._SOURCES_PATH")
    def test_load_non_list_json(self, mock_path):
        """Non-list JSON returns empty list."""
        mock_path.exists.return_value = True
        with patch("builtins.open", mock_open(read_data='{"key": "value"}')):
            sources = load_sources()
        assert sources == []

    def test_add_source(self, tmp_path):
        """add_source appends to sources.json."""
        sources_file = tmp_path / "sources.json"
        sources_file.write_text("[]")

        with patch("forge.ingest.blog_scraper._SOURCES_PATH", sources_file):
            entry = add_source(
                url="https://blog.example.com",
                name="Example Blog",
                creator="alice",
            )

        assert entry["url"] == "https://blog.example.com"
        assert entry["name"] == "Example Blog"
        assert entry["creator"] == "alice"
        assert "added_at" in entry

        # Verify it was persisted
        with patch("forge.ingest.blog_scraper._SOURCES_PATH", sources_file):
            loaded = load_sources()
        assert len(loaded) == 1
        assert loaded[0]["url"] == "https://blog.example.com"
