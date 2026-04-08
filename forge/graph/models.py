"""Pydantic v2 graph models for the NeuralForge knowledge graph."""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""

    expert = "expert"
    concept = "concept"
    technique = "technique"
    tool = "tool"
    dataset = "dataset"
    model = "model"
    paper = "paper"
    institution = "institution"


class EdgeType(str, Enum):
    """Types of edges (relationships) between nodes."""

    expert_in = "expert_in"
    authored_by = "authored_by"
    depends_on = "depends_on"
    alternative_to = "alternative_to"
    agrees_with = "agrees_with"
    contradicts = "contradicts"
    supersedes = "supersedes"
    recommends = "recommends"
    covers = "covers"
    preferred_over = "preferred_over"
    incompatible_with = "incompatible_with"
    derived_from = "derived_from"
    used_by = "used_by"
    related_to = "related_to"
    is_a = "is_a"
    has_part = "has_part"
    cites = "cites"
    funds = "funds"


class EdgeSource(str, Enum):
    """How the edge was created."""

    manual = "manual"
    auto_discovered = "auto-discovered"
    mined = "mined"


class Node(BaseModel):
    """A node in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique node identifier")
    name: str = Field(..., description="Display name of the node")
    node_type: NodeType = Field(..., description="Type classification of the node")
    description: Optional[str] = Field(None, description="Human-readable description")
    metadata: dict = Field(default_factory=dict, description="Arbitrary key-value metadata")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-format creation timestamp",
    )


class Edge(BaseModel):
    """A directed edge (relationship) in the knowledge graph."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique edge identifier")
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    edge_type: EdgeType = Field(..., description="Type of relationship")
    weight: float = Field(1.0, description="Edge weight / strength (0.0-1.0)")
    confidence: float = Field(1.0, description="Confidence in this relationship (0.0-1.0)")
    source: EdgeSource = Field(EdgeSource.manual, description="How this edge was created")
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence references")
    metadata: dict = Field(default_factory=dict, description="Arbitrary key-value metadata")
    valid_from: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-format start of validity period",
    )
    valid_to: Optional[str] = Field(None, description="ISO-format end of validity period (None = still valid)")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO-format creation timestamp",
    )


class TraversalResult(BaseModel):
    """Result of a graph traversal operation."""

    root_id: str = Field(..., description="Starting node of the traversal")
    nodes: list[Node] = Field(default_factory=list, description="Nodes visited during traversal")
    edges: list[Edge] = Field(default_factory=list, description="Edges traversed")
    depth: int = Field(0, description="Maximum depth reached")


class Contradiction(BaseModel):
    """A detected contradiction between two edges or claims."""

    edge_a: Edge = Field(..., description="First conflicting edge")
    edge_b: Edge = Field(..., description="Second conflicting edge")
    topic: str = Field("", description="Topic or domain of the contradiction")
    explanation: str = Field("", description="Human-readable explanation of the conflict")


class ExpertRanking(BaseModel):
    """Authority ranking for an expert on a specific topic."""

    expert_id: str = Field(..., description="ID of the expert node")
    expert_name: str = Field("", description="Display name of the expert")
    topic: str = Field("", description="Topic for which authority is measured")
    score: float = Field(0.0, description="Authority score")
    edge_count: int = Field(0, description="Number of relevant edges")


class GraphStats(BaseModel):
    """Statistics about the knowledge graph."""

    total_nodes: int = Field(0, description="Total number of nodes")
    total_edges: int = Field(0, description="Total number of edges")
    node_type_counts: dict[str, int] = Field(default_factory=dict, description="Count per node type")
    edge_type_counts: dict[str, int] = Field(default_factory=dict, description="Count per edge type")
    active_edges: int = Field(0, description="Edges with no valid_to (still active)")
    expired_edges: int = Field(0, description="Edges with valid_to set")
