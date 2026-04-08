"""Comprehensive tests for forge.graph.engine — 30+ tests using networkx fallback:
traverse, pagerank, communities, shortest path, contradictions, mutations, buffering."""
import time

import pytest

from forge.graph.engine import GraphEngine, HAS_CUGRAPH
from forge.graph.models import (
    Contradiction,
    Edge,
    EdgeType,
    ExpertRanking,
    Node,
    NodeType,
    TraversalResult,
)
from forge.graph.store import GraphStore


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def store(tmp_path):
    """Fresh GraphStore backed by a temp directory."""
    return GraphStore(data_dir=str(tmp_path))


@pytest.fixture
def engine(store):
    """GraphEngine with a fresh store, already loaded."""
    eng = GraphEngine(store, reload_interval=0)
    eng.load()
    return eng


@pytest.fixture
def populated_engine(store):
    """Engine with a small graph pre-built for query tests.

    Graph structure:
        Alice (expert) --expert_in--> ML (concept)
        Bob   (expert) --expert_in--> ML (concept)
        Alice (expert) --recommends-> PyTorch (tool)
        Bob   (expert) --depends_on-> TensorFlow (tool)
        PyTorch (tool) --alternative_to-> TensorFlow (tool)
    """
    alice = store.add_node("Alice", NodeType.expert)
    bob = store.add_node("Bob", NodeType.expert)
    ml = store.add_node("ML", NodeType.concept)
    pytorch = store.add_node("PyTorch", NodeType.tool)
    tf = store.add_node("TensorFlow", NodeType.tool)

    store.add_edge(alice.id, ml.id, EdgeType.expert_in)
    store.add_edge(bob.id, ml.id, EdgeType.expert_in)
    store.add_edge(alice.id, pytorch.id, EdgeType.recommends)
    store.add_edge(bob.id, tf.id, EdgeType.depends_on)
    store.add_edge(pytorch.id, tf.id, EdgeType.alternative_to)

    eng = GraphEngine(store, reload_interval=0)
    eng.load()
    return eng, alice, bob, ml, pytorch, tf


# ===================================================================
# Initialization
# ===================================================================

class TestEngineInit:
    """Tests for GraphEngine initialization."""

    def test_creates_with_store(self, store):
        eng = GraphEngine(store)
        assert eng.store is store

    def test_graph_is_none_before_load(self, store):
        eng = GraphEngine(store)
        assert eng._graph is None

    def test_load_creates_graph(self, engine):
        assert engine._graph is not None

    def test_uses_networkx_fallback(self):
        # We're running on macOS without cuGraph
        assert not HAS_CUGRAPH

    def test_pending_lists_empty_after_load(self, engine):
        assert engine._pending_nodes == []
        assert engine._pending_edges == []


# ===================================================================
# Mutations
# ===================================================================

class TestMutations:
    """Tests for add_node and add_edge through the engine."""

    def test_add_node_returns_node(self, engine):
        n = engine.add_node(name="Test", node_type=NodeType.concept)
        assert isinstance(n, Node)
        assert n.name == "Test"

    def test_add_node_updates_store(self, engine):
        engine.add_node(name="A", node_type=NodeType.expert)
        assert len(engine.store.nodes_df) == 1

    def test_add_node_tracks_pending(self, engine):
        n = engine.add_node(name="A", node_type=NodeType.expert)
        assert len(engine._pending_nodes) == 1
        assert engine._pending_nodes[0]["id"] == n.id

    def test_add_node_updates_graph_incrementally(self, engine):
        n = engine.add_node(name="A", node_type=NodeType.expert)
        assert engine._graph.has_node(n.id)

    def test_add_edge_returns_edge(self, engine):
        n1 = engine.add_node(name="A", node_type=NodeType.concept)
        n2 = engine.add_node(name="B", node_type=NodeType.concept)
        e = engine.add_edge(source_id=n1.id, target_id=n2.id, edge_type=EdgeType.related_to)
        assert isinstance(e, Edge)

    def test_add_edge_updates_store(self, engine):
        n1 = engine.add_node(name="A", node_type=NodeType.concept)
        n2 = engine.add_node(name="B", node_type=NodeType.concept)
        engine.add_edge(source_id=n1.id, target_id=n2.id, edge_type=EdgeType.related_to)
        assert len(engine.store.edges_df) == 1

    def test_add_edge_tracks_pending(self, engine):
        n1 = engine.add_node(name="A", node_type=NodeType.concept)
        n2 = engine.add_node(name="B", node_type=NodeType.concept)
        e = engine.add_edge(source_id=n1.id, target_id=n2.id, edge_type=EdgeType.related_to)
        assert len(engine._pending_edges) == 1

    def test_add_edge_updates_graph_incrementally(self, engine):
        n1 = engine.add_node(name="A", node_type=NodeType.concept)
        n2 = engine.add_node(name="B", node_type=NodeType.concept)
        engine.add_edge(source_id=n1.id, target_id=n2.id, edge_type=EdgeType.related_to)
        assert engine._graph.has_edge(n1.id, n2.id)

    def test_force_reload_clears_pending(self, engine):
        engine.add_node(name="A", node_type=NodeType.concept)
        assert len(engine._pending_nodes) == 1
        engine.force_reload()
        assert len(engine._pending_nodes) == 0


# ===================================================================
# Traversal
# ===================================================================

class TestTraversal:
    """Tests for BFS traversal."""

    def test_traverse_returns_traversal_result(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        result = eng.traverse(alice.id, depth=1)
        assert isinstance(result, TraversalResult)
        assert result.root_id == alice.id

    def test_traverse_depth_1(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        result = eng.traverse(alice.id, depth=1)
        node_ids = {n.id for n in result.nodes}
        assert alice.id in node_ids
        assert ml.id in node_ids
        assert pytorch.id in node_ids

    def test_traverse_depth_2(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        result = eng.traverse(alice.id, depth=2)
        node_ids = {n.id for n in result.nodes}
        # Should reach Bob (via ML) and TensorFlow (via PyTorch)
        assert bob.id in node_ids
        assert tf.id in node_ids

    def test_traverse_depth_0(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        result = eng.traverse(alice.id, depth=0)
        node_ids = {n.id for n in result.nodes}
        assert alice.id in node_ids
        assert len(result.edges) == 0

    def test_traverse_with_edge_type_filter(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        result = eng.traverse(alice.id, depth=1, edge_types=[EdgeType.recommends])
        node_ids = {n.id for n in result.nodes}
        assert pytorch.id in node_ids
        # ML should NOT be included (expert_in is filtered out)
        assert ml.id not in node_ids

    def test_traverse_empty_graph(self, engine):
        result = engine.traverse("nonexistent", depth=2)
        assert result.nodes == []

    def test_traverse_no_graph(self, store):
        eng = GraphEngine(store)
        result = eng.traverse("x", depth=1)
        assert result.depth == 0


# ===================================================================
# PageRank
# ===================================================================

class TestPageRank:
    """Tests for PageRank computation."""

    def test_pagerank_returns_dict(self, populated_engine):
        eng, *_ = populated_engine
        pr = eng.pagerank()
        assert isinstance(pr, dict)

    def test_pagerank_all_nodes_present(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        pr = eng.pagerank()
        assert alice.id in pr
        assert bob.id in pr
        assert ml.id in pr

    def test_pagerank_scores_sum_to_one(self, populated_engine):
        eng, *_ = populated_engine
        pr = eng.pagerank()
        total = sum(pr.values())
        assert abs(total - 1.0) < 0.01

    def test_pagerank_empty_graph(self, engine):
        pr = engine.pagerank()
        assert pr == {}

    def test_pagerank_ml_has_high_score(self, populated_engine):
        """ML concept has 2 incoming edges, should rank higher."""
        eng, alice, bob, ml, pytorch, tf = populated_engine
        pr = eng.pagerank()
        assert pr[ml.id] > pr[alice.id]


# ===================================================================
# Communities
# ===================================================================

class TestCommunities:
    """Tests for community detection."""

    def test_find_communities_returns_dict(self, populated_engine):
        eng, *_ = populated_engine
        comm = eng.find_communities()
        assert isinstance(comm, dict)

    def test_find_communities_assigns_all_nodes(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        comm = eng.find_communities()
        assert alice.id in comm
        assert bob.id in comm
        assert ml.id in comm

    def test_find_communities_empty_graph(self, engine):
        comm = engine.find_communities()
        assert comm == {}

    def test_find_communities_values_are_ints(self, populated_engine):
        eng, *_ = populated_engine
        comm = eng.find_communities()
        assert all(isinstance(v, int) for v in comm.values())


# ===================================================================
# Shortest path
# ===================================================================

class TestShortestPath:
    """Tests for shortest path computation."""

    def test_shortest_path_direct(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        path = eng.shortest_path(alice.id, ml.id)
        assert len(path) == 2
        assert path[0] == alice.id
        assert path[-1] == ml.id

    def test_shortest_path_indirect(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        path = eng.shortest_path(alice.id, tf.id)
        assert len(path) >= 2
        assert path[0] == alice.id
        assert path[-1] == tf.id

    def test_shortest_path_no_path(self, engine):
        n1 = engine.add_node(name="Isolated1", node_type=NodeType.concept)
        n2 = engine.add_node(name="Isolated2", node_type=NodeType.concept)
        path = engine.shortest_path(n1.id, n2.id)
        assert path == []

    def test_shortest_path_nonexistent_node(self, engine):
        path = engine.shortest_path("no-a", "no-b")
        assert path == []

    def test_shortest_path_same_node(self, populated_engine):
        eng, alice, *_ = populated_engine
        path = eng.shortest_path(alice.id, alice.id)
        assert path == [alice.id]


# ===================================================================
# Contradictions
# ===================================================================

class TestContradictions:
    """Tests for contradiction detection."""

    def test_no_contradictions_in_clean_graph(self, populated_engine):
        eng, *_ = populated_engine
        contradictions = eng.find_contradictions()
        assert contradictions == []

    def test_finds_explicit_contradiction(self, engine):
        a = engine.add_node(name="Claim A", node_type=NodeType.concept)
        b = engine.add_node(name="Claim B", node_type=NodeType.concept)
        engine.add_edge(source_id=a.id, target_id=b.id, edge_type=EdgeType.contradicts)
        contradictions = engine.find_contradictions()
        assert len(contradictions) >= 1
        assert isinstance(contradictions[0], Contradiction)

    def test_finds_contradiction_vs_agreement(self, engine):
        a = engine.add_node(name="Node A", node_type=NodeType.concept)
        b = engine.add_node(name="Node B", node_type=NodeType.concept)
        c = engine.add_node(name="Node C", node_type=NodeType.concept)
        engine.add_edge(source_id=a.id, target_id=b.id, edge_type=EdgeType.contradicts)
        engine.add_edge(source_id=a.id, target_id=c.id, edge_type=EdgeType.agrees_with)
        contradictions = engine.find_contradictions()
        assert len(contradictions) >= 1

    def test_contradiction_with_topic_filter(self, engine):
        a = engine.add_node(name="ML Claim", node_type=NodeType.concept)
        b = engine.add_node(name="NLP Claim", node_type=NodeType.concept)
        engine.add_edge(source_id=a.id, target_id=b.id, edge_type=EdgeType.contradicts)
        # Filter by "ML" topic
        contradictions = engine.find_contradictions(topic="ML")
        assert len(contradictions) >= 1

    def test_contradiction_topic_filter_excludes(self, engine):
        a = engine.add_node(name="Alpha", node_type=NodeType.concept)
        b = engine.add_node(name="Beta", node_type=NodeType.concept)
        engine.add_edge(source_id=a.id, target_id=b.id, edge_type=EdgeType.contradicts)
        # Filter by unrelated topic
        contradictions = engine.find_contradictions(topic="quantum")
        assert len(contradictions) == 0

    def test_empty_graph_no_contradictions(self, engine):
        assert engine.find_contradictions() == []


# ===================================================================
# Expert authority
# ===================================================================

class TestExpertAuthority:
    """Tests for expert authority ranking."""

    def test_returns_rankings(self, populated_engine):
        eng, *_ = populated_engine
        rankings = eng.expert_authority("ML")
        assert isinstance(rankings, list)

    def test_experts_ranked_by_topic(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        rankings = eng.expert_authority("ML")
        assert len(rankings) >= 1
        assert all(isinstance(r, ExpertRanking) for r in rankings)

    def test_rankings_sorted_by_score(self, populated_engine):
        eng, *_ = populated_engine
        rankings = eng.expert_authority("ML")
        if len(rankings) >= 2:
            assert rankings[0].score >= rankings[1].score

    def test_no_experts_for_unknown_topic(self, populated_engine):
        eng, *_ = populated_engine
        rankings = eng.expert_authority("quantum_computing_xyz")
        assert rankings == []

    def test_empty_graph_returns_empty(self, engine):
        rankings = engine.expert_authority("anything")
        assert rankings == []


# ===================================================================
# find_changes_since
# ===================================================================

class TestFindChangesSince:
    """Tests for temporal change tracking."""

    def test_finds_recent_edges(self, populated_engine):
        eng, *_ = populated_engine
        # All edges were created just now
        changes = eng.find_changes_since("2020-01-01T00:00:00")
        assert len(changes) == 5

    def test_no_changes_in_future(self, populated_engine):
        eng, *_ = populated_engine
        changes = eng.find_changes_since("2099-01-01T00:00:00")
        assert len(changes) == 0

    def test_topic_filter(self, populated_engine):
        eng, alice, bob, ml, pytorch, tf = populated_engine
        changes = eng.find_changes_since("2020-01-01T00:00:00", topic="ML")
        assert len(changes) >= 1
        # All returned edges should involve ML-related nodes
        for edge in changes:
            src = eng.store.get_node(edge.source_id)
            tgt = eng.store.get_node(edge.target_id)
            assert (
                (src and "ml" in src.name.lower())
                or (tgt and "ml" in tgt.name.lower())
            )


# ===================================================================
# get_graph_as_of
# ===================================================================

class TestGetGraphAsOf:
    """Tests for temporal graph reconstruction."""

    def test_returns_engine(self, populated_engine):
        eng, *_ = populated_engine
        result = eng.get_graph_as_of("2099-01-01T00:00:00")
        assert result is eng

    def test_future_date_keeps_all_edges(self, populated_engine):
        eng, *_ = populated_engine
        eng.get_graph_as_of("2099-01-01T00:00:00")
        assert eng._graph.number_of_edges() == 5

    def test_past_date_removes_all_edges(self, populated_engine):
        eng, *_ = populated_engine
        eng.get_graph_as_of("2000-01-01T00:00:00")
        assert eng._graph.number_of_edges() == 0

    def test_empty_edges(self, engine):
        result = engine.get_graph_as_of("2025-01-01T00:00:00")
        assert result is engine


# ===================================================================
# Reload behavior
# ===================================================================

class TestReloadBehavior:
    """Tests for auto-reload and force-reload mechanics."""

    def test_maybe_reload_triggers_on_interval(self, store):
        eng = GraphEngine(store, reload_interval=0)
        # _last_reload is 0, so any call should trigger reload
        eng._maybe_reload()
        assert eng._last_reload > 0

    def test_force_reload_updates_timestamp(self, engine):
        old_ts = engine._last_reload
        time.sleep(0.01)
        engine.force_reload()
        assert engine._last_reload > old_ts
