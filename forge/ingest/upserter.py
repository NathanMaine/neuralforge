"""Batch upserter -- embeds chunks via Triton and upserts to Qdrant.

Uses :func:`forge.core.embeddings.get_embeddings_batch` for efficient
batched embedding instead of one-at-a-time calls.
"""
import logging
import uuid

from forge.core import embeddings, qdrant_client
from forge.core.utils import now_iso

logger = logging.getLogger(__name__)


async def ingest_chunks(
    chunks: list[str],
    creator: str,
    title: str,
    source: str,
    source_type: str = "document",
) -> int:
    """Batch embed chunks via Triton and upsert to Qdrant.

    Parameters
    ----------
    chunks:
        Text chunks to ingest.
    creator:
        Expert/creator name for these chunks.
    title:
        Document or content title.
    source:
        Source URL or file path.
    source_type:
        Type of source (default ``"document"``).

    Returns
    -------
    int
        Number of chunks successfully upserted.
    """
    if not chunks:
        return 0

    # Batch embed all chunks
    vecs = await embeddings.get_embeddings_batch(chunks)

    # Build points, skipping any with failed embeddings
    points: list[dict] = []
    timestamp = now_iso()

    for i, (text, vec) in enumerate(zip(chunks, vecs)):
        if vec is None:
            logger.warning("Skipping chunk %d -- embedding is None", i)
            continue
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": vec,
            "payload": {
                "text": text,
                "creator": creator,
                "title": title,
                "source": source,
                "source_type": source_type,
                "chunk_index": i,
                "ingested_at": timestamp,
            },
        })

    if not points:
        logger.warning("No valid embeddings produced for %d chunks", len(chunks))
        return 0

    upserted = qdrant_client.upsert_points(points)
    logger.info(
        "Upserted %d/%d chunks for %r (%s)",
        upserted, len(chunks), creator, title,
    )
    return upserted
