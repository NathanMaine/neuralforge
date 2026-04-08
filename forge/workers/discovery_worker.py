"""Discovery worker -- iterates expert pairs and discovers relationships.

Wraps :func:`forge.graph.discovery.discover_pair` to run as a
scheduled background task.
"""
import itertools
import logging

from forge.config import DISCOVERY_PAIRS_PER_RUN
from forge.core.qdrant_client import get_all_expert_names
from forge.graph.discovery import discover_pair, create_edge_from_discovery, get_shared_topics

logger = logging.getLogger(__name__)


async def run_discovery() -> dict:
    """Run one discovery pass over expert pairs.

    Iterates over pairs of known experts, finds shared topics,
    and discovers relationships via NIM classification.

    Returns
    -------
    dict
        Summary with ``pairs_checked``, ``relationships_found``,
        and ``errors`` counts.
    """
    experts = get_all_expert_names()
    if len(experts) < 2:
        logger.info("Need at least 2 experts for discovery, found %d", len(experts))
        return {"pairs_checked": 0, "relationships_found": 0, "errors": 0}

    pairs = list(itertools.combinations(experts, 2))
    pairs_to_check = pairs[:DISCOVERY_PAIRS_PER_RUN]

    pairs_checked = 0
    relationships_found = 0
    errors = 0

    for expert_a, expert_b in pairs_to_check:
        try:
            # Find shared topics
            topics = get_shared_topics(expert_a, expert_b)
            if not topics:
                pairs_checked += 1
                continue

            # Check the first shared topic
            topic = topics[0]
            result = await discover_pair(expert_a, expert_b, topic)

            if result is not None:
                relationships_found += 1
                logger.info(
                    "Discovered: %s -[%s]-> %s on %s (conf=%.2f)",
                    expert_a,
                    result["relationship"],
                    expert_b,
                    topic,
                    result["confidence"],
                )

            pairs_checked += 1

        except Exception as exc:
            logger.exception(
                "Error discovering pair %s/%s: %s",
                expert_a,
                expert_b,
                exc,
            )
            errors += 1

    summary = {
        "pairs_checked": pairs_checked,
        "relationships_found": relationships_found,
        "errors": errors,
    }
    logger.info("Discovery pass complete: %s", summary)
    return summary
