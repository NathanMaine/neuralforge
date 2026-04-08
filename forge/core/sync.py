"""Data consistency -- single write path for Qdrant + cuGraph sync.

Every write to the knowledge base goes through DataSync so that both
the vector store (Qdrant) and the knowledge graph stay in lockstep.
"""
import logging
import uuid
from typing import Optional

from forge.core import embeddings, qdrant_client
from forge.core.utils import now_iso
from forge.graph.engine import GraphEngine

logger = logging.getLogger(__name__)


class DataSync:
    """Ensures Qdrant and the knowledge graph are always consistent.

    All ingestion flows through :meth:`ingest_and_sync` so that both
    stores are updated in the same pipeline.
    """

    def __init__(self, graph_engine: GraphEngine):
        self.graph = graph_engine

    # ------------------------------------------------------------------
    # Single write path
    # ------------------------------------------------------------------

    async def ingest_and_sync(
        self,
        chunks: list[str],
        creator: str,
        title: str,
        source: str,
        source_type: str = "document",
    ) -> dict:
        """Embed chunks, upsert to Qdrant, and update the knowledge graph.

        Parameters
        ----------
        chunks:
            List of text chunks to ingest.
        creator:
            The expert/creator name owning these chunks.
        title:
            Document or content title.
        source:
            Source URL or file path.
        source_type:
            Type of source (``document``, ``blog``, ``video``, etc.).

        Returns
        -------
        dict
            Summary with ``chunks_upserted``, ``expert_created``, and
            ``errors`` keys.
        """
        result = {
            "chunks_upserted": 0,
            "expert_created": False,
            "errors": [],
        }

        if not chunks:
            return result

        # Step 1: Batch embed
        try:
            vecs = await embeddings.get_embeddings_batch(chunks)
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            result["errors"].append(f"embedding_failed: {exc}")
            return result

        # Step 2: Build Qdrant points (skip chunks with failed embeddings)
        points: list[dict] = []
        for i, (text, vec) in enumerate(zip(chunks, vecs)):
            if vec is None:
                result["errors"].append(f"chunk_{i}_embedding_none")
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
                    "ingested_at": now_iso(),
                },
            })

        # Step 3: Upsert to Qdrant
        if points:
            try:
                upserted = qdrant_client.upsert_points(points)
                result["chunks_upserted"] = upserted
            except Exception as exc:
                logger.error("Qdrant upsert failed: %s", exc)
                result["errors"].append(f"qdrant_upsert_failed: {exc}")
                return result

        # Step 4: Ensure expert node exists in graph
        if not self.graph.has_expert(creator):
            self.graph.add_expert(creator, source_type=source_type)
            result["expert_created"] = True

        return result

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------

    def check_consistency(self) -> dict:
        """Compare Qdrant creators vs graph expert nodes.

        Returns
        -------
        dict
            ``consistent`` (bool), ``qdrant_only`` (list of creator names
            in Qdrant but not in graph), ``graph_only`` (list of expert
            names in graph but not in Qdrant).
        """
        qdrant_names = set(qdrant_client.get_all_expert_names())
        graph_experts = {
            n.name for n in self.graph.get_all_experts()
        }

        qdrant_only = sorted(qdrant_names - graph_experts)
        graph_only = sorted(graph_experts - qdrant_names)

        return {
            "consistent": len(qdrant_only) == 0 and len(graph_only) == 0,
            "qdrant_only": qdrant_only,
            "graph_only": graph_only,
        }

    # ------------------------------------------------------------------
    # Repair
    # ------------------------------------------------------------------

    def repair(self) -> dict:
        """Create missing graph nodes for Qdrant-only creators.

        Returns
        -------
        dict
            ``repaired`` (int) and ``names`` (list of created expert names).
        """
        status = self.check_consistency()
        created: list[str] = []

        for name in status["qdrant_only"]:
            self.graph.add_expert(name)
            created.append(name)

        return {
            "repaired": len(created),
            "names": created,
        }
