"""Blog scraper with multi-strategy discovery and trafilatura extraction.

Discovers articles via sitemap, RSS, and link-crawl strategies,
extracts clean text with trafilatura, scrubs PII, and ingests
via the batch upserter (Triton).
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

from forge.config import SCRAPE_REQUEST_DELAY, SCRAPE_REQUEST_TIMEOUT
from forge.core.utils import content_hash, now_iso
from forge.ingest.chunker import chunk_text
from forge.ingest.pii_scrubber import scrub_pii
from forge.ingest.upserter import ingest_chunks

logger = logging.getLogger(__name__)

_SOURCES_PATH = Path(__file__).parent / "sources.json"

# ---- User-Agent for polite scraping ----
_HEADERS = {
    "User-Agent": "NeuralForge/1.0 (Knowledge Graph Builder)",
}


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


def load_sources() -> list[dict]:
    """Load configured blog sources from sources.json.

    Returns
    -------
    list[dict]
        Each dict has at least ``url``, ``name``, and ``creator`` keys.
    """
    if not _SOURCES_PATH.exists():
        return []
    try:
        with open(_SOURCES_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("sources.json is not a list, returning empty")
            return []
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load sources.json: %s", exc)
        return []


def add_source(
    url: str,
    name: str,
    creator: str,
    strategy: str = "auto",
) -> dict:
    """Add a blog source to sources.json.

    Parameters
    ----------
    url:
        Base URL of the blog.
    name:
        Human-readable name.
    creator:
        Expert/creator name for ingested chunks.
    strategy:
        Discovery strategy: ``'sitemap'``, ``'rss'``, ``'crawl'``, or ``'auto'``.

    Returns
    -------
    dict
        The newly added source entry.
    """
    sources = load_sources()
    entry = {
        "url": url.rstrip("/"),
        "name": name,
        "creator": creator,
        "strategy": strategy,
        "added_at": now_iso(),
    }
    sources.append(entry)
    with open(_SOURCES_PATH, "w") as f:
        json.dump(sources, f, indent=2)
    logger.info("Added source: %s (%s)", name, url)
    return entry


# ---------------------------------------------------------------------------
# Discovery strategies
# ---------------------------------------------------------------------------


def _discover_sitemap(base_url: str) -> list[str]:
    """Discover article URLs from sitemap.xml.

    Tries ``/sitemap.xml``, ``/sitemap_index.xml``, and
    ``/robots.txt`` to find sitemap URLs.

    Returns
    -------
    list[str]
        Discovered article URLs.
    """
    urls: list[str] = []
    sitemap_candidates = [
        urljoin(base_url, "/sitemap.xml"),
        urljoin(base_url, "/sitemap_index.xml"),
    ]

    # Check robots.txt for sitemap references
    try:
        resp = requests.get(
            urljoin(base_url, "/robots.txt"),
            headers=_HEADERS,
            timeout=SCRAPE_REQUEST_TIMEOUT,
        )
        if resp.ok:
            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    if sitemap_url not in sitemap_candidates:
                        sitemap_candidates.append(sitemap_url)
    except requests.RequestException:
        pass

    for sitemap_url in sitemap_candidates:
        try:
            resp = requests.get(
                sitemap_url,
                headers=_HEADERS,
                timeout=SCRAPE_REQUEST_TIMEOUT,
            )
            if not resp.ok:
                continue
            # Extract <loc> entries
            locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", resp.text)
            for loc in locs:
                # Sub-sitemaps: recurse one level
                if "sitemap" in loc.lower() and loc.endswith(".xml"):
                    try:
                        sub_resp = requests.get(
                            loc,
                            headers=_HEADERS,
                            timeout=SCRAPE_REQUEST_TIMEOUT,
                        )
                        if sub_resp.ok:
                            sub_locs = re.findall(
                                r"<loc>\s*(.*?)\s*</loc>", sub_resp.text
                            )
                            urls.extend(sub_locs)
                    except requests.RequestException:
                        pass
                else:
                    urls.append(loc)
        except requests.RequestException:
            continue

    return urls


def _discover_rss(base_url: str) -> list[str]:
    """Discover article URLs from RSS/Atom feeds.

    Tries common feed paths: ``/feed``, ``/rss``, ``/atom.xml``,
    ``/feed.xml``, ``/rss.xml``, ``/index.xml``.

    Returns
    -------
    list[str]
        Discovered article URLs.
    """
    urls: list[str] = []
    feed_paths = ["/feed", "/rss", "/atom.xml", "/feed.xml", "/rss.xml", "/index.xml"]

    for path in feed_paths:
        feed_url = urljoin(base_url, path)
        try:
            resp = requests.get(
                feed_url,
                headers=_HEADERS,
                timeout=SCRAPE_REQUEST_TIMEOUT,
            )
            if not resp.ok:
                continue
            # Extract <link> entries (RSS and Atom)
            links = re.findall(r"<link>\s*(.*?)\s*</link>", resp.text)
            urls.extend(links)
            # Also try href attribute (Atom)
            href_links = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', resp.text)
            urls.extend(href_links)
        except requests.RequestException:
            continue

    return urls


def _discover_crawl(base_url: str) -> list[str]:
    """Discover article URLs by crawling the homepage for links.

    Extracts all ``<a href="...">`` links from the base URL page
    and filters for likely article paths (containing ``/blog/``,
    ``/post/``, ``/article/``, year patterns like ``/2024/``, etc.).

    Returns
    -------
    list[str]
        Discovered article URLs.
    """
    urls: list[str] = []
    try:
        resp = requests.get(
            base_url,
            headers=_HEADERS,
            timeout=SCRAPE_REQUEST_TIMEOUT,
        )
        if not resp.ok:
            return urls

        links = re.findall(r'href=["\']([^"\']+)["\']', resp.text)
        parsed_base = urlparse(base_url)

        article_patterns = re.compile(
            r"/(blog|post|article|news|insights|writing|essays?)/|"
            r"/20\d{2}/",
            re.IGNORECASE,
        )

        for link in links:
            # Make absolute
            absolute = urljoin(base_url, link)
            parsed = urlparse(absolute)

            # Same domain only
            if parsed.netloc != parsed_base.netloc:
                continue

            # Filter for article-like paths
            if article_patterns.search(parsed.path):
                urls.append(absolute)

    except requests.RequestException:
        pass

    return urls


def discover_articles(
    base_url: str,
    strategy: str = "auto",
) -> list[str]:
    """Discover article URLs from a blog using the specified strategy.

    Parameters
    ----------
    base_url:
        Base URL of the blog.
    strategy:
        ``'sitemap'``, ``'rss'``, ``'crawl'``, or ``'auto'`` (try all).

    Returns
    -------
    list[str]
        Deduplicated list of discovered article URLs.
    """
    base_url = base_url.rstrip("/")
    urls: list[str] = []

    if strategy == "sitemap":
        urls = _discover_sitemap(base_url)
    elif strategy == "rss":
        urls = _discover_rss(base_url)
    elif strategy == "crawl":
        urls = _discover_crawl(base_url)
    elif strategy == "auto":
        # Try all strategies, merge results
        urls.extend(_discover_sitemap(base_url))
        urls.extend(_discover_rss(base_url))
        urls.extend(_discover_crawl(base_url))
    else:
        logger.warning("Unknown strategy %r, falling back to auto", strategy)
        return discover_articles(base_url, strategy="auto")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for url in urls:
        normalized = url.rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    logger.info(
        "Discovered %d articles from %s (strategy=%s)",
        len(deduped),
        base_url,
        strategy,
    )
    return deduped


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_article(url: str) -> Optional[dict]:
    """Extract clean text from an article URL using trafilatura.

    Parameters
    ----------
    url:
        The article URL to fetch and extract.

    Returns
    -------
    dict or None
        ``{url, title, text, extracted_at}`` on success, None on failure.
    """
    try:
        import trafilatura

        resp = requests.get(
            url,
            headers=_HEADERS,
            timeout=SCRAPE_REQUEST_TIMEOUT,
        )
        if not resp.ok:
            logger.warning("HTTP %d fetching %s", resp.status_code, url)
            return None

        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not text or not text.strip():
            logger.debug("No text extracted from %s", url)
            return None

        # Extract title via trafilatura metadata
        metadata = trafilatura.extract_metadata(resp.text)
        title = ""
        if metadata and metadata.title:
            title = metadata.title
        if not title:
            # Fallback: extract from <title> tag
            title_match = re.search(r"<title>(.*?)</title>", resp.text, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()

        return {
            "url": url,
            "title": title or url.split("/")[-1],
            "text": text,
            "extracted_at": now_iso(),
        }

    except ImportError:
        logger.error("trafilatura not installed")
        return None
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None
    except Exception as exc:
        logger.exception("Unexpected error extracting %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

# Track hashes of already-ingested articles to skip duplicates
_seen_hashes: set[str] = set()


async def scrape_blog(
    source: dict,
    max_articles: int = 50,
) -> dict:
    """Full pipeline: discover, extract, PII-scrub, chunk, and ingest.

    Parameters
    ----------
    source:
        Source config dict with ``url``, ``name``, ``creator``, and
        optionally ``strategy`` keys.
    max_articles:
        Maximum number of articles to process per run.

    Returns
    -------
    dict
        Summary with ``source``, ``discovered``, ``extracted``,
        ``ingested``, and ``skipped`` counts.
    """
    base_url = source["url"]
    creator = source["creator"]
    strategy = source.get("strategy", "auto")
    name = source.get("name", base_url)

    # Discover
    article_urls = discover_articles(base_url, strategy=strategy)

    discovered = len(article_urls)
    extracted = 0
    ingested = 0
    skipped = 0

    for url in article_urls[:max_articles]:
        # Polite delay
        time.sleep(SCRAPE_REQUEST_DELAY)

        # Extract
        article = extract_article(url)
        if article is None:
            skipped += 1
            continue

        # Dedup by content hash
        h = content_hash(article["text"])
        if h in _seen_hashes:
            skipped += 1
            logger.debug("Skipping duplicate article: %s", url)
            continue
        _seen_hashes.add(h)

        extracted += 1

        # PII scrub
        scrubbed_text, pii_counts = scrub_pii(article["text"])
        if pii_counts:
            logger.info("Scrubbed PII from %s: %s", url, pii_counts)

        # Chunk
        chunks = chunk_text(scrubbed_text)
        if not chunks:
            continue

        # Ingest via batch upserter
        count = await ingest_chunks(
            chunks=chunks,
            creator=creator,
            title=article.get("title", ""),
            source=url,
            source_type="blog",
        )
        ingested += count

    summary = {
        "source": name,
        "discovered": discovered,
        "extracted": extracted,
        "ingested": ingested,
        "skipped": skipped,
    }
    logger.info("Scrape complete for %s: %s", name, summary)
    return summary
