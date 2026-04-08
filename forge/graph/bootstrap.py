"""Graph bootstrap -- populate graph on first startup.

Scans Qdrant for existing creators and builds expert nodes plus
a set of common concept nodes so the graph is ready for discovery.
"""
import logging

from forge.core import qdrant_client
from forge.graph.engine import GraphEngine

logger = logging.getLogger(__name__)

# Common knowledge-domain concepts seeded on first boot
DEFAULT_CONCEPTS = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "data engineering",
    "cybersecurity",
    "cloud computing",
    "distributed systems",
    "software architecture",
    "devops",
    "database systems",
    "ethics in AI",
    "robotics",
    "quantum computing",
    "blockchain",
    "internet of things",
    "edge computing",
    "data privacy",
    "human computer interaction",
]


def bootstrap_graph(engine: GraphEngine) -> dict:
    """Populate the graph on first startup.

    Only runs if the graph is empty.  Scans Qdrant for existing creator
    names, creates expert nodes for each, and seeds 20 common concept
    nodes.

    Parameters
    ----------
    engine:
        The :class:`GraphEngine` to populate.

    Returns
    -------
    dict
        Summary with ``experts_created``, ``concepts_created``,
        ``skipped`` (bool -- True if graph was non-empty), and
        ``expert_names`` list.
    """
    result = {
        "experts_created": 0,
        "concepts_created": 0,
        "skipped": False,
        "expert_names": [],
    }

    if not engine.is_empty():
        logger.info("Graph already populated (%d nodes), skipping bootstrap", engine.node_count())
        result["skipped"] = True
        return result

    # Scan Qdrant for creators
    creators = qdrant_client.get_all_expert_names()
    for name in creators:
        engine.add_expert(name)
        result["experts_created"] += 1
        result["expert_names"].append(name)

    # Seed concept nodes
    for concept in DEFAULT_CONCEPTS:
        engine.add_concept(concept)
        result["concepts_created"] += 1

    logger.info(
        "Bootstrap complete: %d experts, %d concepts",
        result["experts_created"],
        result["concepts_created"],
    )
    return result
