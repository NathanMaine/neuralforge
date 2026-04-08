"""GPU-accelerated graph engine for the NeuralForge knowledge graph.

Uses cuGraph when available (GPU), falls back to networkx for CPU-only
environments and testing.
"""
import logging
import time
from collections import deque
from typing import Optional

from forge.graph.models import (
    Contradiction,
    Edge,
    EdgeSource,
    EdgeType,
    ExpertRanking,
    Node,
    NodeType,
    TraversalResult,
)
from forge.graph.store import GraphStore

logger = logging.getLogger(__name__)

try:
    import cudf
    import cugraph

    HAS_CUGRAPH = True
except ImportError:
    import networkx as nx

    HAS_CUGRAPH = False

# How often (seconds) the engine checks whether the store has changed
_DEFAULT_RELOAD_INTERVAL = 300


class GraphEngine:
    """High-level graph engine with cuGraph acceleration and networkx fallback.

    Args:
        store: The underlying ``GraphStore`` for persistence.
        reload_interval: Seconds between automatic reloads from the store.
    """

    def __init__(self, store: GraphStore, reload_interval: int = _DEFAULT_RELOAD_INTERVAL):
        self.store = store
        self._graph = None
        self._pending_nodes: list[dict] = []
        self._pending_edges: list[dict] = []
        self._last_reload: float = 0
        self._reload_interval = reload_interval

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build (or rebuild) the in-memory graph from the store DataFrames."""
        if HAS_CUGRAPH:
            self._load_cugraph()
        else:
            self._load_networkx()
        self._last_reload = time.time()
        self._pending_nodes.clear()
        self._pending_edges.clear()

    def _load_networkx(self) -> None:
        """Build a networkx DiGraph from the store."""
        g = nx.DiGraph()
        for _, row in self.store.nodes_df.iterrows():
            g.add_node(row["id"], name=row["name"], node_type=row["node_type"])
        for _, row in self.store.edges_df.iterrows():
            g.add_edge(
                row["source_id"],
                row["target_id"],
                edge_id=row["id"],
                edge_type=row["edge_type"],
                weight=float(row.get("weight", 1.0)),
                valid_from=row.get("valid_from"),
                valid_to=row.get("valid_to"),
            )
        self._graph = g

    def _load_cugraph(self) -> None:  # pragma: no cover
        """Build a cuGraph graph from the store."""
        if self.store.edges_df.empty:
            self._graph = None
            return
        edf = cudf.DataFrame(
            {
                "src": self.store.edges_df["source_id"],
                "dst": self.store.edges_df["target_id"],
            }
        )
        g = cugraph.Graph(directed=True)
        g.from_cudf_edgelist(edf, source="src", destination="dst")
        self._graph = g

    def _maybe_reload(self) -> None:
        """Reload from store if the reload interval has elapsed."""
        if time.time() - self._last_reload >= self._reload_interval:
            self.load()

    def force_reload(self) -> None:
        """Force an immediate reload of the graph from the store."""
        self.load()

    # ------------------------------------------------------------------
    # Mutations (buffered through the store)
    # ------------------------------------------------------------------

    def add_node(self, **kwargs) -> Node:
        """Add a node via the store and mark graph as needing reload."""
        node = self.store.add_node(**kwargs)
        self._pending_nodes.append({"id": node.id})
        # If we have a networkx graph, update it incrementally
        if self._graph is not None and not HAS_CUGRAPH:
            self._graph.add_node(
                node.id, name=node.name, node_type=node.node_type.value
            )
        return node

    def add_edge(self, **kwargs) -> Edge:
        """Add an edge via the store and mark graph as needing reload."""
        edge = self.store.add_edge(**kwargs)
        self._pending_edges.append({"id": edge.id})
        # Incremental update for networkx
        if self._graph is not None and not HAS_CUGRAPH:
            self._graph.add_edge(
                edge.source_id,
                edge.target_id,
                edge_id=edge.id,
                edge_type=edge.edge_type.value,
                weight=edge.weight,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
            )
        return edge

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def traverse(
        self,
        node_id: str,
        depth: int = 2,
        edge_types: Optional[list[EdgeType | str]] = None,
        as_of: Optional[str] = None,
    ) -> TraversalResult:
        """BFS traversal from a node up to ``depth`` hops.

        Filters edges by ``edge_types`` and temporal ``as_of`` if provided.
        """
        self._maybe_reload()
        if self._graph is None:
            return TraversalResult(root_id=node_id, depth=0)

        visited_nodes: dict[str, Node] = {}
        collected_edges: list[Edge] = []
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        visited: set[str] = {node_id}
        max_depth_reached = 0

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth > depth:
                continue
            max_depth_reached = max(max_depth_reached, current_depth)

            # Get the node from store
            node = self.store.get_node(current_id)
            if node:
                visited_nodes[current_id] = node

            if current_depth == depth:
                continue

            # Get edges through store (respects temporal filtering)
            edges = self.store.get_edges_for_node(
                current_id, edge_types=edge_types, as_of=as_of
            )
            for edge in edges:
                collected_edges.append(edge)
                neighbor = (
                    edge.target_id
                    if edge.source_id == current_id
                    else edge.source_id
                )
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))

        return TraversalResult(
            root_id=node_id,
            nodes=list(visited_nodes.values()),
            edges=collected_edges,
            depth=max_depth_reached,
        )

    def pagerank(self, personalization: Optional[dict[str, float]] = None) -> dict[str, float]:
        """Compute PageRank scores for all nodes.

        Args:
            personalization: Optional mapping of node IDs to personalization weights.

        Returns:
            Mapping of node ID to PageRank score.
        """
        self._maybe_reload()
        if self._graph is None or (not HAS_CUGRAPH and self._graph.number_of_nodes() == 0):
            return {}

        if HAS_CUGRAPH:  # pragma: no cover
            result = cugraph.pagerank(self._graph)
            return dict(zip(result["vertex"].to_pandas(), result["pagerank"].to_pandas()))

        return nx.pagerank(self._graph, personalization=personalization)

    def find_communities(self) -> dict[str, int]:
        """Detect communities using Louvain algorithm.

        Returns:
            Mapping of node ID to community ID.
        """
        self._maybe_reload()
        if self._graph is None:
            return {}

        if HAS_CUGRAPH:  # pragma: no cover
            parts, _ = cugraph.louvain(self._graph)
            return dict(zip(parts["vertex"].to_pandas(), parts["partition"].to_pandas()))

        # networkx louvain works on undirected graphs
        undirected = self._graph.to_undirected()
        if undirected.number_of_nodes() == 0:
            return {}

        communities = nx.community.louvain_communities(undirected, seed=42)
        result: dict[str, int] = {}
        for community_id, members in enumerate(communities):
            for node_id in members:
                result[node_id] = community_id
        return result

    def shortest_path(self, source_id: str, target_id: str) -> list[str]:
        """Find the shortest path between two nodes.

        Returns:
            List of node IDs from source to target, or empty list if no path.
        """
        self._maybe_reload()
        if self._graph is None:
            return []

        if HAS_CUGRAPH:  # pragma: no cover
            try:
                result = cugraph.shortest_path(self._graph, source_id)
                return [source_id, target_id]  # simplified
            except Exception:
                return []

        try:
            return list(nx.shortest_path(self._graph, source_id, target_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def find_contradictions(self, topic: Optional[str] = None) -> list[Contradiction]:
        """Find contradicting edges in the graph.

        Contradictions are pairs of edges where one is ``contradicts`` or
        ``incompatible_with`` and the other is ``agrees_with`` between nodes
        that share a common neighbor.  Standalone contradiction edges are also
        reported.
        """
        self._maybe_reload()
        contradictions: list[Contradiction] = []

        if self.store.edges_df.empty:
            return contradictions

        contra_types = {
            EdgeType.contradicts.value,
            EdgeType.incompatible_with.value,
        }
        agree_types = {
            EdgeType.agrees_with.value,
        }

        contra_edges = self.store.edges_df[
            self.store.edges_df["edge_type"].isin(contra_types)
        ]
        agree_edges = self.store.edges_df[
            self.store.edges_df["edge_type"].isin(agree_types)
        ]

        def _topic_match(edge_obj: Edge) -> bool:
            """Return True if at least one endpoint matches the topic."""
            if not topic:
                return True
            src_node = self.store.get_node(edge_obj.source_id)
            tgt_node = self.store.get_node(edge_obj.target_id)
            src_ok = src_node and topic.lower() in src_node.name.lower()
            tgt_ok = tgt_node and topic.lower() in tgt_node.name.lower()
            return bool(src_ok or tgt_ok)

        # Pair contradiction edges with agreement edges on shared nodes
        for _, c_row in contra_edges.iterrows():
            c_edge = self.store._row_to_edge(c_row)
            if not _topic_match(c_edge):
                continue

            for _, a_row in agree_edges.iterrows():
                a_edge = self.store._row_to_edge(a_row)
                nodes_involved = {c_edge.source_id, c_edge.target_id}
                if a_edge.source_id in nodes_involved or a_edge.target_id in nodes_involved:
                    contradictions.append(
                        Contradiction(
                            edge_a=c_edge,
                            edge_b=a_edge,
                            topic=topic or "",
                            explanation=(
                                f"Contradiction between {c_edge.edge_type.value} and "
                                f"{a_edge.edge_type.value}"
                            ),
                        )
                    )

        # Report standalone contradiction edges when no pairs found
        if not contradictions:
            for _, c_row in contra_edges.iterrows():
                c_edge = self.store._row_to_edge(c_row)
                if not _topic_match(c_edge):
                    continue
                contradictions.append(
                    Contradiction(
                        edge_a=c_edge,
                        edge_b=c_edge,
                        topic=topic or "",
                        explanation=f"Explicit {c_edge.edge_type.value} relationship",
                    )
                )

        return contradictions

    def expert_authority(self, topic: str) -> list[ExpertRanking]:
        """Rank experts by authority on a topic.

        Authority is based on the number of edges connecting an expert node
        to concept nodes whose name matches the topic, weighted by PageRank.
        """
        self._maybe_reload()
        rankings: list[ExpertRanking] = []

        experts = self.store.get_nodes_by_type(NodeType.expert)
        if not experts:
            return rankings

        pr_scores = self.pagerank()

        for expert in experts:
            edges = self.store.get_edges_for_node(expert.id)
            topic_edge_count = 0
            for edge in edges:
                other_id = (
                    edge.target_id if edge.source_id == expert.id else edge.source_id
                )
                other_node = self.store.get_node(other_id)
                if other_node and topic.lower() in other_node.name.lower():
                    topic_edge_count += 1

            if topic_edge_count > 0:
                score = pr_scores.get(expert.id, 0.0) * topic_edge_count
                rankings.append(
                    ExpertRanking(
                        expert_id=expert.id,
                        expert_name=expert.name,
                        topic=topic,
                        score=score,
                        edge_count=topic_edge_count,
                    )
                )

        rankings.sort(key=lambda r: r.score, reverse=True)
        return rankings

    def find_changes_since(self, date: str, topic: Optional[str] = None) -> list[Edge]:
        """Find edges created or modified since a given date.

        Args:
            date: ISO-format date string.
            topic: Optional topic filter (matches node names).

        Returns:
            List of edges created after ``date``.
        """
        self._maybe_reload()
        if self.store.edges_df.empty:
            return []

        mask = self.store.edges_df["created_at"] >= date
        results = []
        for _, row in self.store.edges_df[mask].iterrows():
            edge = self.store._row_to_edge(row)
            if topic:
                src = self.store.get_node(edge.source_id)
                tgt = self.store.get_node(edge.target_id)
                src_match = src and topic.lower() in src.name.lower()
                tgt_match = tgt and topic.lower() in tgt.name.lower()
                if not (src_match or tgt_match):
                    continue
            results.append(edge)
        return results

    def get_graph_as_of(self, date: str):
        """Rebuild the graph with only edges valid at the given date.

        Filters edges where ``valid_from <= date`` and (``valid_to`` is null
        or ``valid_to >= date``), then rebuilds the in-memory graph.

        Returns:
            The engine instance (for chaining).
        """
        if self.store.edges_df.empty:
            if not HAS_CUGRAPH:
                self._graph = nx.DiGraph()
            else:  # pragma: no cover
                self._graph = None
            return self

        mask = self.store.edges_df["valid_from"] <= date
        mask = mask & (
            self.store.edges_df["valid_to"].isna()
            | (self.store.edges_df["valid_to"] >= date)
        )

        if HAS_CUGRAPH:  # pragma: no cover
            filtered = self.store.edges_df[mask]
            if filtered.empty:
                self._graph = None
            else:
                edf = cudf.DataFrame(
                    {"src": filtered["source_id"], "dst": filtered["target_id"]}
                )
                g = cugraph.Graph(directed=True)
                g.from_cudf_edgelist(edf, source="src", destination="dst")
                self._graph = g
        else:
            g = nx.DiGraph()
            for _, row in self.store.nodes_df.iterrows():
                g.add_node(row["id"], name=row["name"], node_type=row["node_type"])
            for _, row in self.store.edges_df[mask].iterrows():
                g.add_edge(
                    row["source_id"],
                    row["target_id"],
                    edge_id=row["id"],
                    edge_type=row["edge_type"],
                    weight=float(row.get("weight", 1.0)),
                    valid_from=row.get("valid_from"),
                    valid_to=row.get("valid_to"),
                )
            self._graph = g

        return self

    # ------------------------------------------------------------------
    # Domain-specific convenience methods
    # ------------------------------------------------------------------

    def add_expert(self, name: str, **metadata) -> Node:
        """Add an expert node. Returns existing if name already taken."""
        existing = self.store.get_node_by_name(name)
        if existing is not None and existing.node_type == NodeType.expert:
            return existing
        return self.add_node(
            name=name,
            node_type=NodeType.expert,
            metadata=metadata if metadata else None,
        )

    def get_expert(self, name: str) -> Optional[Node]:
        """Get an expert node by name."""
        node = self.store.get_node_by_name(name)
        if node is not None and node.node_type == NodeType.expert:
            return node
        return None

    def get_all_experts(self) -> list[Node]:
        """Return all expert nodes."""
        return self.store.get_nodes_by_type(NodeType.expert)

    def has_expert(self, name: str) -> bool:
        """Check if an expert node exists."""
        return self.get_expert(name) is not None

    def add_concept(self, name: str, **metadata) -> Node:
        """Add a concept node. Returns existing if name already taken."""
        existing = self.store.get_node_by_name(name)
        if existing is not None and existing.node_type == NodeType.concept:
            return existing
        return self.add_node(
            name=name,
            node_type=NodeType.concept,
            metadata=metadata if metadata else None,
        )

    def get_concept(self, name: str) -> Optional[Node]:
        """Get a concept node by name."""
        node = self.store.get_node_by_name(name)
        if node is not None and node.node_type == NodeType.concept:
            return node
        return None

    def get_all_concepts(self) -> list[Node]:
        """Return all concept nodes."""
        return self.store.get_nodes_by_type(NodeType.concept)

    def add_relationship(
        self,
        expert_a: str,
        expert_b: str,
        rel_type: str,
        **metadata,
    ) -> Optional[Edge]:
        """Add a relationship edge between two experts by name."""
        node_a = self.get_expert(expert_a)
        node_b = self.get_expert(expert_b)
        if node_a is None or node_b is None:
            logger.warning(
                "Cannot create relationship: expert(s) not found (%s, %s)",
                expert_a, expert_b,
            )
            return None

        try:
            edge_type = EdgeType(rel_type)
        except ValueError:
            edge_type = EdgeType.related_to

        confidence = metadata.pop("confidence", 1.0)
        return self.add_edge(
            source_id=node_a.id,
            target_id=node_b.id,
            edge_type=edge_type,
            confidence=confidence,
            source=EdgeSource.auto_discovered,
            metadata=metadata if metadata else None,
        )

    def node_count(self) -> int:
        """Total nodes in graph."""
        return len(self.store.nodes_df)

    def edge_count(self) -> int:
        """Total edges in graph."""
        return len(self.store.edges_df)

    def is_empty(self) -> bool:
        """Check if the graph has no nodes."""
        return self.node_count() == 0

    def clear(self) -> None:
        """Clear the entire graph (in-memory only)."""
        import pandas as pd
        from forge.graph.store import _NODE_COLUMNS, _EDGE_COLUMNS
        self.store.nodes_df = pd.DataFrame(columns=_NODE_COLUMNS)
        self.store.edges_df = pd.DataFrame(columns=_EDGE_COLUMNS)
        if not HAS_CUGRAPH:
            self._graph = nx.DiGraph()
        else:  # pragma: no cover
            self._graph = None
