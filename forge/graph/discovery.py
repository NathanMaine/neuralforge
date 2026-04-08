"""Auto-discover expert relationships via NIM classification.

Compares chunks from pairs of experts to find agreements, disagreements,
and shared topics, then creates graph edges representing those
relationships.
"""
import logging
from typing import Optional

from forge.config import DISCOVERY_CONFIDENCE_FLOOR
from forge.core import nim_client, qdrant_client, embeddings
from forge.graph.engine import GraphEngine

logger = logging.getLogger(__name__)

CLASSIFICATION_PROMPT = (
    "Given excerpts from two experts on {topic}, classify their "
    "relationship as one of: agrees, disagrees, extends, or unrelated.\n\n"
    "Expert A ({expert_a}):\n{text_a}\n\n"
    "Expert B ({expert_b}):\n{text_b}\n\n"
    "Respond with JSON: {{\"relationship\": \"...\", \"confidence\": 0.0-1.0, "
    "\"summary\": \"brief explanation\"}}"
)


async def discover_pair(
    expert_a: str,
    expert_b: str,
    topic: str,
) -> Optional[dict]:
    """Discover the relationship between two experts on a given topic.

    Fetches chunks from Qdrant for each expert on the topic, sends
    them to NIM for classification, and applies the confidence floor.

    Parameters
    ----------
    expert_a:
        Name of the first expert.
    expert_b:
        Name of the second expert.
    topic:
        The topic to compare on.

    Returns
    -------
    dict or None
        Relationship dict with ``expert_a``, ``expert_b``, ``topic``,
        ``relationship``, ``confidence``, and ``summary`` keys.
        None if classification fails or confidence is below floor.
    """
    # Get topic embedding for search
    topic_vec = await embeddings.get_embedding(topic)
    if topic_vec is None:
        logger.warning("Failed to embed topic %r", topic)
        return None

    # Search for relevant chunks from each expert
    chunks_a = qdrant_client.search_vectors(topic_vec, limit=3, expert=expert_a)
    chunks_b = qdrant_client.search_vectors(topic_vec, limit=3, expert=expert_b)

    if not chunks_a or not chunks_b:
        logger.debug(
            "No chunks found for pair %s/%s on topic %r",
            expert_a, expert_b, topic,
        )
        return None

    # Build text excerpts
    text_a = "\n".join(c["text"] for c in chunks_a if c.get("text"))
    text_b = "\n".join(c["text"] for c in chunks_b if c.get("text"))

    if not text_a or not text_b:
        return None

    # Classify via NIM
    prompt = CLASSIFICATION_PROMPT.format(
        topic=topic,
        expert_a=expert_a,
        expert_b=expert_b,
        text_a=text_a,
        text_b=text_b,
    )

    result = await nim_client.classify_json(prompt)
    if result is None:
        logger.warning("NIM classification failed for %s vs %s", expert_a, expert_b)
        return None

    confidence = result.get("confidence", 0.0)
    if confidence < DISCOVERY_CONFIDENCE_FLOOR:
        logger.debug(
            "Confidence %.2f below floor %.2f for %s vs %s on %s",
            confidence, DISCOVERY_CONFIDENCE_FLOOR,
            expert_a, expert_b, topic,
        )
        return None

    return {
        "expert_a": expert_a,
        "expert_b": expert_b,
        "topic": topic,
        "relationship": result.get("relationship", "unrelated"),
        "confidence": confidence,
        "summary": result.get("summary", ""),
    }


def create_edge_from_discovery(engine: GraphEngine, result: dict) -> None:
    """Create a graph edge from a discovery result.

    Parameters
    ----------
    engine:
        The graph engine to add the edge to.
    result:
        Discovery result dict from :func:`discover_pair`.
    """
    engine.add_relationship(
        result["expert_a"],
        result["expert_b"],
        rel_type=result["relationship"],
        topic=result["topic"],
        confidence=result["confidence"],
        summary=result.get("summary", ""),
    )
    logger.info(
        "Created edge: %s -[%s]-> %s (topic=%s, conf=%.2f)",
        result["expert_a"],
        result["relationship"],
        result["expert_b"],
        result["topic"],
        result["confidence"],
    )


def get_shared_topics(expert_a: str, expert_b: str) -> list[str]:
    """Find overlapping topics from Qdrant chunk metadata.

    Scrolls chunks for each expert and extracts ``source_type`` and
    ``title`` fields to identify shared topics.

    Parameters
    ----------
    expert_a:
        First expert name.
    expert_b:
        Second expert name.

    Returns
    -------
    list[str]
        Sorted list of shared topic strings (from titles).
    """
    def _get_titles(expert: str) -> set[str]:
        """Scroll Qdrant and collect unique titles for an expert."""
        titles: set[str] = set()
        try:
            payload = {
                "limit": 100,
                "with_payload": ["title"],
                "with_vector": False,
                "filter": {
                    "must": [
                        {"key": "creator", "match": {"value": expert}}
                    ]
                },
            }
            resp = qdrant_client._post(
                f"/collections/{qdrant_client.QDRANT_COLLECTION}/points/scroll",
                json=payload,
            )
            resp.raise_for_status()
            points = resp.json().get("result", {}).get("points", [])
            for pt in points:
                title = (pt.get("payload") or {}).get("title", "")
                if title:
                    titles.add(title.lower())
        except Exception:
            logger.exception("Error fetching titles for %s", expert)
        return titles

    titles_a = _get_titles(expert_a)
    titles_b = _get_titles(expert_b)
    return sorted(titles_a & titles_b)
