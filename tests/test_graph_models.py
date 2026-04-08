"""Comprehensive tests for forge.graph.models — 20+ tests for enums and Pydantic models."""
from datetime import datetime

import pytest
from pydantic import ValidationError

from forge.graph.models import (
    Contradiction,
    Edge,
    EdgeSource,
    EdgeType,
    ExpertRanking,
    GraphStats,
    Node,
    NodeType,
    TraversalResult,
)


# ===================================================================
# NodeType enum
# ===================================================================

class TestNodeType:
    """Tests for the NodeType enum."""

    def test_has_eight_values(self):
        assert len(NodeType) == 8

    def test_all_values(self):
        expected = {
            "expert", "concept", "technique", "tool",
            "dataset", "model", "paper", "institution",
        }
        assert {nt.value for nt in NodeType} == expected

    def test_string_mixin(self):
        assert NodeType.expert == "expert"
        assert str(NodeType.concept) == "NodeType.concept"

    def test_from_string(self):
        assert NodeType("expert") == NodeType.expert
        assert NodeType("paper") == NodeType.paper

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            NodeType("invalid_type")


# ===================================================================
# EdgeType enum
# ===================================================================

class TestEdgeType:
    """Tests for the EdgeType enum."""

    def test_has_eighteen_values(self):
        assert len(EdgeType) == 18

    def test_all_values(self):
        expected = {
            "expert_in", "authored_by", "depends_on", "alternative_to",
            "agrees_with", "contradicts", "supersedes", "recommends",
            "covers", "preferred_over", "incompatible_with", "derived_from",
            "used_by", "related_to", "is_a", "has_part", "cites", "funds",
        }
        assert {et.value for et in EdgeType} == expected

    def test_from_string(self):
        assert EdgeType("contradicts") == EdgeType.contradicts
        assert EdgeType("funds") == EdgeType.funds

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            EdgeType("not_a_type")


# ===================================================================
# EdgeSource enum
# ===================================================================

class TestEdgeSource:
    """Tests for the EdgeSource enum."""

    def test_has_three_values(self):
        assert len(EdgeSource) == 3

    def test_all_values(self):
        expected = {"manual", "auto-discovered", "mined"}
        assert {es.value for es in EdgeSource} == expected

    def test_auto_discovered_has_hyphen(self):
        assert EdgeSource.auto_discovered.value == "auto-discovered"


# ===================================================================
# Node model
# ===================================================================

class TestNode:
    """Tests for the Node Pydantic model."""

    def test_minimal_construction(self):
        n = Node(name="Test Node", node_type=NodeType.concept)
        assert n.name == "Test Node"
        assert n.node_type == NodeType.concept
        assert n.id  # auto-generated uuid

    def test_auto_id_generation(self):
        n1 = Node(name="A", node_type=NodeType.expert)
        n2 = Node(name="B", node_type=NodeType.expert)
        assert n1.id != n2.id

    def test_auto_created_at(self):
        n = Node(name="Test", node_type=NodeType.tool)
        assert n.created_at  # ISO string

    def test_defaults(self):
        n = Node(name="Test", node_type=NodeType.dataset)
        assert n.description is None
        assert n.metadata == {}

    def test_full_construction(self):
        n = Node(
            id="custom-id",
            name="Full Node",
            node_type=NodeType.paper,
            description="A test paper",
            metadata={"year": 2024},
            created_at="2024-01-01T00:00:00",
        )
        assert n.id == "custom-id"
        assert n.description == "A test paper"
        assert n.metadata["year"] == 2024

    def test_serialization_round_trip(self):
        n = Node(name="Test", node_type=NodeType.model, metadata={"k": "v"})
        data = n.model_dump()
        n2 = Node(**data)
        assert n2.name == n.name
        assert n2.node_type == n.node_type
        assert n2.metadata == n.metadata

    def test_json_round_trip(self):
        n = Node(name="Test", node_type=NodeType.institution)
        json_str = n.model_dump_json()
        n2 = Node.model_validate_json(json_str)
        assert n2.id == n.id

    def test_missing_required_name(self):
        with pytest.raises(ValidationError):
            Node(node_type=NodeType.concept)

    def test_missing_required_type(self):
        with pytest.raises(ValidationError):
            Node(name="No Type")


# ===================================================================
# Edge model
# ===================================================================

class TestEdge:
    """Tests for the Edge Pydantic model."""

    def test_minimal_construction(self):
        e = Edge(source_id="a", target_id="b", edge_type=EdgeType.related_to)
        assert e.source_id == "a"
        assert e.target_id == "b"
        assert e.id  # auto-generated

    def test_defaults(self):
        e = Edge(source_id="a", target_id="b", edge_type=EdgeType.cites)
        assert e.weight == 1.0
        assert e.confidence == 1.0
        assert e.source == EdgeSource.manual
        assert e.evidence == []
        assert e.metadata == {}
        assert e.valid_to is None

    def test_full_construction(self):
        e = Edge(
            id="edge-1",
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.depends_on,
            weight=0.8,
            confidence=0.9,
            source=EdgeSource.mined,
            evidence=["paper123"],
            metadata={"note": "strong"},
            valid_from="2024-01-01T00:00:00",
            valid_to="2025-01-01T00:00:00",
        )
        assert e.weight == 0.8
        assert e.source == EdgeSource.mined
        assert len(e.evidence) == 1

    def test_serialization(self):
        e = Edge(source_id="a", target_id="b", edge_type=EdgeType.funds)
        data = e.model_dump()
        assert data["source_id"] == "a"
        assert "valid_from" in data

    def test_missing_source_id(self):
        with pytest.raises(ValidationError):
            Edge(target_id="b", edge_type=EdgeType.cites)


# ===================================================================
# TraversalResult model
# ===================================================================

class TestTraversalResult:
    """Tests for the TraversalResult model."""

    def test_minimal(self):
        t = TraversalResult(root_id="r1")
        assert t.root_id == "r1"
        assert t.nodes == []
        assert t.edges == []
        assert t.depth == 0

    def test_with_data(self):
        n = Node(name="Test", node_type=NodeType.concept)
        e = Edge(source_id="a", target_id="b", edge_type=EdgeType.related_to)
        t = TraversalResult(root_id="a", nodes=[n], edges=[e], depth=2)
        assert len(t.nodes) == 1
        assert len(t.edges) == 1
        assert t.depth == 2


# ===================================================================
# Contradiction model
# ===================================================================

class TestContradiction:
    """Tests for the Contradiction model."""

    def test_construction(self):
        e1 = Edge(source_id="a", target_id="b", edge_type=EdgeType.agrees_with)
        e2 = Edge(source_id="a", target_id="b", edge_type=EdgeType.contradicts)
        c = Contradiction(
            edge_a=e1,
            edge_b=e2,
            topic="ML",
            explanation="Conflicting claims",
        )
        assert c.topic == "ML"
        assert c.explanation == "Conflicting claims"

    def test_defaults(self):
        e = Edge(source_id="a", target_id="b", edge_type=EdgeType.contradicts)
        c = Contradiction(edge_a=e, edge_b=e)
        assert c.topic == ""
        assert c.explanation == ""


# ===================================================================
# ExpertRanking model
# ===================================================================

class TestExpertRanking:
    """Tests for the ExpertRanking model."""

    def test_construction(self):
        r = ExpertRanking(
            expert_id="e1",
            expert_name="Dr. Smith",
            topic="NLP",
            score=0.95,
            edge_count=5,
        )
        assert r.expert_name == "Dr. Smith"
        assert r.score == 0.95

    def test_defaults(self):
        r = ExpertRanking(expert_id="e1")
        assert r.expert_name == ""
        assert r.topic == ""
        assert r.score == 0.0
        assert r.edge_count == 0


# ===================================================================
# GraphStats model
# ===================================================================

class TestGraphStats:
    """Tests for the GraphStats model."""

    def test_all_defaults(self):
        s = GraphStats()
        assert s.total_nodes == 0
        assert s.total_edges == 0
        assert s.node_type_counts == {}
        assert s.edge_type_counts == {}
        assert s.active_edges == 0
        assert s.expired_edges == 0

    def test_with_counts(self):
        s = GraphStats(
            total_nodes=100,
            total_edges=250,
            node_type_counts={"expert": 20, "concept": 80},
            edge_type_counts={"related_to": 150, "cites": 100},
            active_edges=200,
            expired_edges=50,
        )
        assert s.total_nodes == 100
        assert s.node_type_counts["expert"] == 20
        assert s.active_edges + s.expired_edges == s.total_edges

    def test_serialization(self):
        s = GraphStats(total_nodes=5)
        data = s.model_dump()
        assert data["total_nodes"] == 5
        s2 = GraphStats(**data)
        assert s2.total_nodes == 5
