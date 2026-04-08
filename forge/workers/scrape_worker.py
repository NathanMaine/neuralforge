"""Scrape worker -- iterates configured blog sources and scrapes articles.

Wraps :func:`forge.ingest.blog_scraper.scrape_blog` to run as a
scheduled background task.
"""
import logging

from forge.ingest.blog_scraper import load_sources, scrape_blog

logger = logging.getLogger(__name__)


async def run_scrape() -> dict:
    """Run one scrape pass over all configured blog sources.

    Returns
    -------
    dict
        Summary with ``sources_processed``, ``total_discovered``,
        ``total_ingested``, and ``errors`` counts.
    """
    sources = load_sources()
    if not sources:
        logger.info("No blog sources configured, skipping scrape")
        return {
            "sources_processed": 0,
            "total_discovered": 0,
            "total_ingested": 0,
            "errors": 0,
        }

    sources_processed = 0
    total_discovered = 0
    total_ingested = 0
    errors = 0

    for source in sources:
        try:
            result = await scrape_blog(source)
            sources_processed += 1
            total_discovered += result.get("discovered", 0)
            total_ingested += result.get("ingested", 0)
            logger.info(
                "Scraped %s: discovered=%d, ingested=%d",
                source.get("name", source["url"]),
                result.get("discovered", 0),
                result.get("ingested", 0),
            )
        except Exception as exc:
            logger.exception(
                "Error scraping %s: %s",
                source.get("name", source.get("url")),
                exc,
            )
            errors += 1

    summary = {
        "sources_processed": sources_processed,
        "total_discovered": total_discovered,
        "total_ingested": total_ingested,
        "errors": errors,
    }
    logger.info("Scrape pass complete: %s", summary)
    return summary
