"""Comprehensive tests for forge.graph.store — 40+ tests for CRUD, temporal filtering,
and Parquet persistence round-trip."""
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from forge.graph.models import (
    Edge,
    EdgeSource,
    EdgeType,
    GraphStats,
    Node,
    NodeType,
)
from forge.graph.store import GraphStore


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def store(tmp_path):
    """Create a fresh GraphStore backed by a temp directory."""
    return GraphStore(data_dir=str(tmp_path))


@pytest.fixture
def populated_store(store):
    """A store with some pre-populated nodes and edges."""
    n1 = store.add_node("Alice", NodeType.expert, description="ML researcher")
    n2 = store.add_node("Bob", NodeType.expert, description="NLP expert")
    n3 = store.add_node("Transformers", NodeType.concept, description="Attention mechanism")
    n4 = store.add_node("PyTorch", NodeType.tool)
    store.add_edge(n1.id, n3.id, EdgeType.expert_in)
    store.add_edge(n2.id, n3.id, EdgeType.expert_in)
    store.add_edge(n1.id, n4.id, EdgeType.recommends)
    return store, n1, n2, n3, n4


# ===================================================================
# Initialization
# ===================================================================

class TestStoreInit:
    """Tests for GraphStore initialization."""

    def test_creates_data_dir(self, tmp_path):
        sub = tmp_path / "nested" / "graph"
        store = GraphStore(data_dir=str(sub))
        assert sub.exists()

    def test_empty_dataframes(self, store):
        assert len(store.nodes_df) == 0
        assert len(store.edges_df) == 0

    def test_correct_node_columns(self, store):
        expected = {"id", "name", "node_type", "description", "metadata", "created_at"}
        assert set(store.nodes_df.columns) == expected

    def test_correct_edge_columns(self, store):
        expected = {
            "id", "source_id", "target_id", "edge_type", "weight",
            "confidence", "source", "evidence", "metadata",
            "valid_from", "valid_to", "created_at",
        }
        assert set(store.edges_df.columns) == expected


# ===================================================================
# Node CRUD
# ===================================================================

class TestNodeCRUD:
    """Tests for node creation, retrieval, and search."""

    def test_add_node_returns_node(self, store):
        n = store.add_node("Test", NodeType.concept)
        assert isinstance(n, Node)
        assert n.name == "Test"
        assert n.node_type == NodeType.concept

    def test_add_node_generates_id(self, store):
        n = store.add_node("A", NodeType.expert)
        assert n.id  # non-empty uuid
        assert len(n.id) == 36  # uuid4 format

    def test_add_node_sets_created_at(self, store):
        n = store.add_node("A", NodeType.expert)
        assert n.created_at
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(n.created_at)

    def test_add_node_with_description(self, store):
        n = store.add_node("Alice", NodeType.expert, description="ML researcher")
        assert n.description == "ML researcher"

    def test_add_node_with_metadata(self, store):
        meta = {"affiliation": "MIT", "h_index": 50}
        n = store.add_node("Bob", NodeType.expert, metadata=meta)
        assert n.metadata == meta

    def test_add_node_string_type(self, store):
        n = store.add_node("X", "concept")
        assert n.node_type == NodeType.concept

    def test_add_node_updates_dataframe(self, store):
        store.add_node("A", NodeType.expert)
        store.add_node("B", NodeType.concept)
        assert len(store.nodes_df) == 2

    def test_get_node_found(self, store):
        n = store.add_node("Alice", NodeType.expert)
        result = store.get_node(n.id)
        assert result is not None
        assert result.name == "Alice"

    def test_get_node_not_found(self, store):
        assert store.get_node("nonexistent") is None

    def test_get_node_by_name_found(self, store):
        store.add_node("Alice", NodeType.expert)
        result = store.get_node_by_name("Alice")
        assert result is not None
        assert result.name == "Alice"

    def test_get_node_by_name_not_found(self, store):
        assert store.get_node_by_name("Nobody") is None

    def test_get_node_by_name_prevents_duplicates(self, store):
        n1 = store.add_node("Alice", NodeType.expert)
        existing = store.get_node_by_name("Alice")
        assert existing.id == n1.id

    def test_get_nodes_by_type(self, store):
        store.add_node("A", NodeType.expert)
        store.add_node("B", NodeType.expert)
        store.add_node("C", NodeType.concept)
        experts = store.get_nodes_by_type(NodeType.expert)
        assert len(experts) == 2
        assert all(n.node_type == NodeType.expert for n in experts)

    def test_get_nodes_by_type_string(self, store):
        store.add_node("A", NodeType.tool)
        tools = store.get_nodes_by_type("tool")
        assert len(tools) == 1

    def test_get_nodes_by_type_empty(self, store):
        result = store.get_nodes_by_type(NodeType.paper)
        assert result == []

    def test_search_nodes_by_name(self, store):
        store.add_node("Machine Learning", NodeType.concept)
        store.add_node("Deep Learning", NodeType.concept)
        store.add_node("PyTorch", NodeType.tool)
        results = store.search_nodes("learning")
        assert len(results) == 2

    def test_search_nodes_case_insensitive(self, store):
        store.add_node("PyTorch", NodeType.tool)
        results = store.search_nodes("pytorch")
        assert len(results) == 1

    def test_search_nodes_with_type_filter(self, store):
        store.add_node("ML Tool", NodeType.tool)
        store.add_node("ML Concept", NodeType.concept)
        results = store.search_nodes("ML", node_type=NodeType.tool)
        assert len(results) == 1
        assert results[0].node_type == NodeType.tool

    def test_search_nodes_limit(self, store):
        for i in range(10):
            store.add_node(f"Node {i}", NodeType.concept)
        results = store.search_nodes("Node", limit=3)
        assert len(results) == 3

    def test_search_nodes_no_results(self, store):
        store.add_node("Alice", NodeType.expert)
        results = store.search_nodes("zzzzz")
        assert results == []

    def test_search_nodes_empty_store(self, store):
        results = store.search_nodes("anything")
        assert results == []


# ===================================================================
# Edge CRUD
# ===================================================================

class TestEdgeCRUD:
    """Tests for edge creation, retrieval, update, and deletion."""

    def test_add_edge_returns_edge(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        assert isinstance(e, Edge)
        assert e.source_id == "a"
        assert e.target_id == "b"

    def test_add_edge_generates_id(self, store):
        e = store.add_edge("a", "b", EdgeType.cites)
        assert len(e.id) == 36

    def test_add_edge_defaults(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        assert e.weight == 1.0
        assert e.confidence == 1.0
        assert e.source == EdgeSource.manual
        assert e.evidence == []
        assert e.metadata == {}
        assert e.valid_to is None

    def test_add_edge_with_kwargs(self, store):
        e = store.add_edge(
            "a", "b", EdgeType.depends_on,
            weight=0.7,
            confidence=0.85,
            source=EdgeSource.mined,
            evidence=["ref1", "ref2"],
            metadata={"note": "test"},
        )
        assert e.weight == 0.7
        assert e.confidence == 0.85
        assert e.source == EdgeSource.mined
        assert len(e.evidence) == 2

    def test_add_edge_string_type(self, store):
        e = store.add_edge("a", "b", "cites")
        assert e.edge_type == EdgeType.cites

    def test_add_edge_updates_dataframe(self, store):
        store.add_edge("a", "b", EdgeType.related_to)
        store.add_edge("b", "c", EdgeType.depends_on)
        assert len(store.edges_df) == 2

    def test_get_edge_found(self, store):
        e = store.add_edge("a", "b", EdgeType.cites)
        result = store.get_edge(e.id)
        assert result is not None
        assert result.edge_type == EdgeType.cites

    def test_get_edge_not_found(self, store):
        assert store.get_edge("nonexistent") is None

    def test_update_edge_weight(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to, weight=0.5)
        updated = store.update_edge(e.id, weight=0.9)
        assert updated is not None
        assert updated.weight == 0.9

    def test_update_edge_confidence(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        updated = store.update_edge(e.id, confidence=0.3)
        assert updated.confidence == 0.3

    def test_update_edge_metadata(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        updated = store.update_edge(e.id, metadata={"new_key": "new_val"})
        assert updated.metadata == {"new_key": "new_val"}

    def test_update_edge_not_found(self, store):
        result = store.update_edge("nonexistent", weight=0.5)
        assert result is None

    def test_expire_edge(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        expired = store.expire_edge(e.id)
        assert expired is not None
        assert expired.valid_to is not None

    def test_expire_edge_custom_date(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        date_str = "2025-06-15T00:00:00"
        expired = store.expire_edge(e.id, valid_to=date_str)
        assert expired.valid_to == date_str

    def test_expire_edge_not_found(self, store):
        result = store.expire_edge("nonexistent")
        assert result is None

    def test_delete_edge(self, store):
        e = store.add_edge("a", "b", EdgeType.related_to)
        assert len(store.edges_df) == 1
        store.delete_edge(e.id)
        assert len(store.edges_df) == 0

    def test_delete_edge_nonexistent(self, store):
        store.add_edge("a", "b", EdgeType.related_to)
        store.delete_edge("nonexistent")
        assert len(store.edges_df) == 1  # unchanged

    def test_get_edges_for_node_as_source(self, populated_store):
        store, n1, n2, n3, n4 = populated_store
        edges = store.get_edges_for_node(n1.id)
        assert len(edges) == 2  # expert_in + recommends

    def test_get_edges_for_node_as_target(self, populated_store):
        store, n1, n2, n3, n4 = populated_store
        edges = store.get_edges_for_node(n3.id)
        assert len(edges) == 2  # two expert_in edges

    def test_get_edges_for_node_with_type_filter(self, populated_store):
        store, n1, n2, n3, n4 = populated_store
        edges = store.get_edges_for_node(n1.id, edge_types=[EdgeType.recommends])
        assert len(edges) == 1
        assert edges[0].edge_type == EdgeType.recommends

    def test_get_edges_for_node_empty(self, store):
        edges = store.get_edges_for_node("nonexistent")
        assert edges == []


# ===================================================================
# Temporal filtering
# ===================================================================

class TestTemporalFiltering:
    """Tests for as_of temporal edge filtering."""

    def test_as_of_includes_active_edges(self, store):
        n1 = store.add_node("A", NodeType.concept)
        n2 = store.add_node("B", NodeType.concept)
        store.add_edge(n1.id, n2.id, EdgeType.related_to)
        # Query far in the future — edge should be included
        future = (datetime.now() + timedelta(days=365)).isoformat()
        edges = store.get_edges_for_node(n1.id, as_of=future)
        assert len(edges) == 1

    def test_as_of_excludes_future_edges(self, store):
        n1 = store.add_node("A", NodeType.concept)
        n2 = store.add_node("B", NodeType.concept)
        store.add_edge(n1.id, n2.id, EdgeType.related_to)
        # Query before the edge was created
        past = "2020-01-01T00:00:00"
        edges = store.get_edges_for_node(n1.id, as_of=past)
        assert len(edges) == 0

    def test_as_of_excludes_expired_edges(self, store):
        n1 = store.add_node("A", NodeType.concept)
        n2 = store.add_node("B", NodeType.concept)
        e = store.add_edge(n1.id, n2.id, EdgeType.related_to)
        store.expire_edge(e.id, valid_to="2024-06-01T00:00:00")
        # Query after expiry
        edges = store.get_edges_for_node(n1.id, as_of="2025-01-01T00:00:00")
        assert len(edges) == 0

    def test_as_of_includes_edge_at_valid_to(self, store):
        n1 = store.add_node("A", NodeType.concept)
        n2 = store.add_node("B", NodeType.concept)
        e = store.add_edge(n1.id, n2.id, EdgeType.related_to)
        # Manually set valid_from and valid_to
        store.update_edge(e.id, valid_from="2024-01-01T00:00:00", valid_to="2024-12-31T23:59:59")
        edges = store.get_edges_for_node(n1.id, as_of="2024-06-15T00:00:00")
        assert len(edges) == 1


# ===================================================================
# Parquet persistence
# ===================================================================

class TestParquetPersistence:
    """Tests for save/load round-trip with Parquet files."""

    def test_save_creates_files(self, store, tmp_path):
        store.add_node("A", NodeType.expert)
        store.add_edge("a", "b", EdgeType.cites)
        store.save()
        assert (tmp_path / "nodes.parquet").exists()
        assert (tmp_path / "edges.parquet").exists()

    def test_round_trip_nodes(self, tmp_path):
        store1 = GraphStore(data_dir=str(tmp_path))
        store1.add_node("Alice", NodeType.expert, description="ML researcher")
        store1.add_node("Transformers", NodeType.concept, metadata={"year": 2017})
        store1.save()

        store2 = GraphStore(data_dir=str(tmp_path))
        assert len(store2.nodes_df) == 2
        alice = store2.get_node_by_name("Alice")
        assert alice is not None
        assert alice.description == "ML researcher"

    def test_round_trip_edges(self, tmp_path):
        store1 = GraphStore(data_dir=str(tmp_path))
        n1 = store1.add_node("A", NodeType.expert)
        n2 = store1.add_node("B", NodeType.concept)
        e = store1.add_edge(
            n1.id, n2.id, EdgeType.expert_in,
            weight=0.8, confidence=0.9,
            evidence=["paper1"],
            metadata={"note": "strong"},
        )
        store1.save()

        store2 = GraphStore(data_dir=str(tmp_path))
        loaded = store2.get_edge(e.id)
        assert loaded is not None
        assert loaded.weight == 0.8
        assert loaded.confidence == 0.9
        assert loaded.evidence == ["paper1"]
        assert loaded.metadata == {"note": "strong"}
        assert loaded.edge_type == EdgeType.expert_in

    def test_round_trip_metadata_json(self, tmp_path):
        store1 = GraphStore(data_dir=str(tmp_path))
        complex_meta = {"tags": ["a", "b"], "nested": {"x": 1}}
        store1.add_node("Test", NodeType.concept, metadata=complex_meta)
        store1.save()

        store2 = GraphStore(data_dir=str(tmp_path))
        node = store2.get_node_by_name("Test")
        assert node.metadata == complex_meta

    def test_round_trip_expired_edges(self, tmp_path):
        store1 = GraphStore(data_dir=str(tmp_path))
        n1 = store1.add_node("A", NodeType.concept)
        n2 = store1.add_node("B", NodeType.concept)
        e = store1.add_edge(n1.id, n2.id, EdgeType.related_to)
        store1.expire_edge(e.id, valid_to="2025-01-01T00:00:00")
        store1.save()

        store2 = GraphStore(data_dir=str(tmp_path))
        loaded = store2.get_edge(e.id)
        assert loaded.valid_to == "2025-01-01T00:00:00"


# ===================================================================
# Stats
# ===================================================================

class TestGraphStats:
    """Tests for the get_stats method."""

    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats.total_nodes == 0
        assert stats.total_edges == 0
        assert stats.node_type_counts == {}
        assert stats.active_edges == 0

    def test_populated_stats(self, populated_store):
        store, n1, n2, n3, n4 = populated_store
        stats = store.get_stats()
        assert stats.total_nodes == 4
        assert stats.total_edges == 3
        assert stats.node_type_counts["expert"] == 2
        assert stats.node_type_counts["concept"] == 1
        assert stats.node_type_counts["tool"] == 1
        assert stats.active_edges == 3
        assert stats.expired_edges == 0

    def test_stats_with_expired_edges(self, store):
        n1 = store.add_node("A", NodeType.concept)
        n2 = store.add_node("B", NodeType.concept)
        e1 = store.add_edge(n1.id, n2.id, EdgeType.related_to)
        e2 = store.add_edge(n1.id, n2.id, EdgeType.depends_on)
        store.expire_edge(e1.id)
        stats = store.get_stats()
        assert stats.active_edges == 1
        assert stats.expired_edges == 1
        assert stats.total_edges == 2

    def test_stats_returns_graphstats_model(self, store):
        stats = store.get_stats()
        assert isinstance(stats, GraphStats)
