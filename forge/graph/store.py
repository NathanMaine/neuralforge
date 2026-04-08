"""Parquet-backed persistence for the NeuralForge knowledge graph.

Uses pandas (NOT cuDF) so the store works on any platform without a GPU.
Nodes and edges are stored as DataFrames and persisted to Parquet files.
"""
import json
import os
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pandas as pd

from forge.graph.models import (
    Edge,
    EdgeSource,
    EdgeType,
    GraphStats,
    Node,
    NodeType,
)

# Column schemas for the DataFrames
_NODE_COLUMNS = ["id", "name", "node_type", "description", "metadata", "created_at"]
_EDGE_COLUMNS = [
    "id",
    "source_id",
    "target_id",
    "edge_type",
    "weight",
    "confidence",
    "source",
    "evidence",
    "metadata",
    "valid_from",
    "valid_to",
    "created_at",
]


class GraphStore:
    """Parquet-backed store for graph nodes and edges.

    Args:
        data_dir: Directory for ``nodes.parquet`` and ``edges.parquet``.
                  Defaults to ``data/graph``.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or os.environ.get("GRAPH_DATA_DIR", "data/graph")
        os.makedirs(self.data_dir, exist_ok=True)

        self._nodes_path = os.path.join(self.data_dir, "nodes.parquet")
        self._edges_path = os.path.join(self.data_dir, "edges.parquet")

        self.nodes_df = self._load_or_create(self._nodes_path, _NODE_COLUMNS)
        self.edges_df = self._load_or_create(self._edges_path, _EDGE_COLUMNS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_or_create(path: str, columns: list[str]) -> pd.DataFrame:
        """Load a Parquet file if it exists, otherwise return an empty DataFrame."""
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _serialize_json(value) -> str:
        """Serialize a value to a JSON string for storage."""
        if isinstance(value, str):
            return value
        return json.dumps(value)

    @staticmethod
    def _deserialize_json(value: str):
        """Deserialize a JSON string back to a Python object."""
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    @staticmethod
    def _nan_to_none(value):
        """Convert pandas NaN to Python None."""
        if pd.isna(value):
            return None
        return value

    def _row_to_node(self, row: pd.Series) -> Node:
        """Convert a DataFrame row to a Node model."""
        return Node(
            id=row["id"],
            name=row["name"],
            node_type=NodeType(row["node_type"]),
            description=self._nan_to_none(row.get("description")),
            metadata=self._deserialize_json(row.get("metadata", "{}")),
            created_at=row["created_at"],
        )

    def _row_to_edge(self, row: pd.Series) -> Edge:
        """Convert a DataFrame row to an Edge model."""
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            edge_type=EdgeType(row["edge_type"]),
            weight=float(row.get("weight", 1.0)),
            confidence=float(row.get("confidence", 1.0)),
            source=EdgeSource(row.get("source", "manual")),
            evidence=self._deserialize_json(row.get("evidence", "[]")),
            metadata=self._deserialize_json(row.get("metadata", "{}")),
            valid_from=row["valid_from"],
            valid_to=self._nan_to_none(row.get("valid_to")),
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        node_type: NodeType | str,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Node:
        """Add a new node to the graph. Returns the created Node."""
        if isinstance(node_type, str):
            node_type = NodeType(node_type)

        node_id = str(uuid4())
        now = datetime.now().isoformat()
        row = {
            "id": node_id,
            "name": name,
            "node_type": node_type.value,
            "description": description,
            "metadata": self._serialize_json(metadata or {}),
            "created_at": now,
        }
        self.nodes_df = pd.concat(
            [self.nodes_df, pd.DataFrame([row])], ignore_index=True
        )
        return Node(
            id=node_id,
            name=name,
            node_type=node_type,
            description=description,
            metadata=metadata or {},
            created_at=now,
        )

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID. Returns None if not found."""
        mask = self.nodes_df["id"] == node_id
        matches = self.nodes_df[mask]
        if matches.empty:
            return None
        return self._row_to_node(matches.iloc[0])

    def get_node_by_name(self, name: str) -> Optional[Node]:
        """Get a node by its exact name. Returns existing node to prevent duplicates."""
        mask = self.nodes_df["name"] == name
        matches = self.nodes_df[mask]
        if matches.empty:
            return None
        return self._row_to_node(matches.iloc[0])

    def get_nodes_by_type(self, node_type: NodeType | str) -> list[Node]:
        """Get all nodes of a specific type."""
        if isinstance(node_type, NodeType):
            node_type = node_type.value
        mask = self.nodes_df["node_type"] == node_type
        return [self._row_to_node(row) for _, row in self.nodes_df[mask].iterrows()]

    def search_nodes(
        self,
        query: str,
        node_type: Optional[NodeType | str] = None,
        limit: int = 20,
    ) -> list[Node]:
        """Search nodes by name substring (case-insensitive).

        Optionally filter by node type. Returns at most ``limit`` results.
        """
        if self.nodes_df.empty:
            return []

        mask = self.nodes_df["name"].str.contains(query, case=False, na=False)

        if node_type is not None:
            type_val = node_type.value if isinstance(node_type, NodeType) else node_type
            mask = mask & (self.nodes_df["node_type"] == type_val)

        results = self.nodes_df[mask].head(limit)
        return [self._row_to_node(row) for _, row in results.iterrows()]

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        weight: float = 1.0,
        confidence: float = 1.0,
        source: EdgeSource | str = EdgeSource.manual,
        evidence: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Edge:
        """Add a new edge to the graph. Returns the created Edge."""
        if isinstance(edge_type, str):
            edge_type = EdgeType(edge_type)
        if isinstance(source, str):
            source = EdgeSource(source)

        edge_id = str(uuid4())
        now = datetime.now().isoformat()
        row = {
            "id": edge_id,
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type.value,
            "weight": weight,
            "confidence": confidence,
            "source": source.value,
            "evidence": self._serialize_json(evidence or []),
            "metadata": self._serialize_json(metadata or {}),
            "valid_from": now,
            "valid_to": None,
            "created_at": now,
        }
        self.edges_df = pd.concat(
            [self.edges_df, pd.DataFrame([row])], ignore_index=True
        )
        return Edge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
            source=source,
            evidence=evidence or [],
            metadata=metadata or {},
            valid_from=now,
            valid_to=None,
            created_at=now,
        )

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by its ID. Returns None if not found."""
        mask = self.edges_df["id"] == edge_id
        matches = self.edges_df[mask]
        if matches.empty:
            return None
        return self._row_to_edge(matches.iloc[0])

    def update_edge(self, edge_id: str, **fields) -> Optional[Edge]:
        """Update fields on an existing edge. Returns the updated Edge or None."""
        mask = self.edges_df["id"] == edge_id
        if not mask.any():
            return None

        for key, value in fields.items():
            if key in ("evidence", "metadata"):
                value = self._serialize_json(value)
            if key == "edge_type" and isinstance(value, EdgeType):
                value = value.value
            if key == "source" and isinstance(value, EdgeSource):
                value = value.value
            self.edges_df.loc[mask, key] = value

        return self._row_to_edge(self.edges_df[mask].iloc[0])

    def expire_edge(self, edge_id: str, valid_to: Optional[str] = None) -> Optional[Edge]:
        """Mark an edge as expired by setting its valid_to timestamp.

        Args:
            edge_id: The edge to expire.
            valid_to: ISO-format timestamp. Defaults to now.

        Returns:
            The updated Edge, or None if not found.
        """
        valid_to = valid_to or datetime.now().isoformat()
        return self.update_edge(edge_id, valid_to=valid_to)

    def delete_edge(self, edge_id: str) -> None:
        """Permanently remove an edge from the store."""
        mask = self.edges_df["id"] != edge_id
        self.edges_df = self.edges_df[mask].reset_index(drop=True)

    def get_edges_for_node(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType | str]] = None,
        as_of: Optional[str] = None,
    ) -> list[Edge]:
        """Get all edges connected to a node (as source or target).

        Args:
            node_id: The node to query.
            edge_types: Optional list of edge types to filter on.
            as_of: ISO-format date for temporal filtering. Only returns edges
                   where ``valid_from <= as_of`` and (``valid_to`` is null or
                   ``valid_to >= as_of``).
        """
        if self.edges_df.empty:
            return []

        mask = (self.edges_df["source_id"] == node_id) | (
            self.edges_df["target_id"] == node_id
        )

        if edge_types:
            type_values = [
                et.value if isinstance(et, EdgeType) else et for et in edge_types
            ]
            mask = mask & self.edges_df["edge_type"].isin(type_values)

        if as_of is not None:
            mask = mask & (self.edges_df["valid_from"] <= as_of)
            mask = mask & (
                self.edges_df["valid_to"].isna() | (self.edges_df["valid_to"] >= as_of)
            )

        return [self._row_to_edge(row) for _, row in self.edges_df[mask].iterrows()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write DataFrames to Parquet files."""
        self.nodes_df.to_parquet(self._nodes_path, index=False)
        self.edges_df.to_parquet(self._edges_path, index=False)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> GraphStats:
        """Return statistics about the current graph."""
        node_type_counts: dict[str, int] = {}
        if not self.nodes_df.empty:
            node_type_counts = self.nodes_df["node_type"].value_counts().to_dict()

        edge_type_counts: dict[str, int] = {}
        active_edges = 0
        expired_edges = 0
        if not self.edges_df.empty:
            edge_type_counts = self.edges_df["edge_type"].value_counts().to_dict()
            active_edges = int(self.edges_df["valid_to"].isna().sum())
            expired_edges = int(self.edges_df["valid_to"].notna().sum())

        return GraphStats(
            total_nodes=len(self.nodes_df),
            total_edges=len(self.edges_df),
            node_type_counts=node_type_counts,
            edge_type_counts=edge_type_counts,
            active_edges=active_edges,
            expired_edges=expired_edges,
        )
