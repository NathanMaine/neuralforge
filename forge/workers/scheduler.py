"""Background task scheduler using APScheduler.

Runs discovery (every 6 hours) and blog scraping (daily at 2 AM)
as background jobs.
"""
import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from forge.config import DISCOVERY_INTERVAL_HOURS

logger = logging.getLogger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> AsyncIOScheduler:
    """Get or create the singleton scheduler instance.

    Returns
    -------
    AsyncIOScheduler
        The application scheduler.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
    return _scheduler


def start_scheduler() -> AsyncIOScheduler:
    """Configure and start the background scheduler.

    Registers:
    - Discovery worker: runs every ``DISCOVERY_INTERVAL_HOURS`` hours.
    - Scrape worker: runs daily at 2:00 AM.

    Returns
    -------
    AsyncIOScheduler
        The running scheduler instance.
    """
    scheduler = get_scheduler()

    if scheduler.running:
        logger.warning("Scheduler is already running")
        return scheduler

    # Import workers here to avoid circular imports
    from forge.workers.discovery_worker import run_discovery
    from forge.workers.scrape_worker import run_scrape

    # Discovery: every N hours
    scheduler.add_job(
        run_discovery,
        "interval",
        hours=DISCOVERY_INTERVAL_HOURS,
        id="discovery_worker",
        name="Expert Discovery",
        replace_existing=True,
    )

    # Blog scraping: daily at 2 AM
    scheduler.add_job(
        run_scrape,
        "cron",
        hour=2,
        minute=0,
        id="scrape_worker",
        name="Blog Scraper",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started: discovery every %dh, scraping daily at 2 AM",
        DISCOVERY_INTERVAL_HOURS,
    )
    return scheduler


def stop_scheduler() -> None:
    """Shut down the scheduler gracefully."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
    _scheduler = None


def is_running() -> bool:
    """Return True if the scheduler is currently running."""
    return _scheduler is not None and _scheduler.running


def get_jobs() -> list[dict]:
    """Return a list of scheduled jobs with their details.

    Returns
    -------
    list[dict]
        Each dict has ``id``, ``name``, ``next_run``, and ``trigger`` keys.
    """
    if _scheduler is None:
        return []

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else None,
            "trigger": str(job.trigger),
        })
    return jobs
