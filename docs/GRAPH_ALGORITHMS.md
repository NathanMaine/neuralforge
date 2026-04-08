# NeuralForge Graph Algorithms

How NeuralForge uses GPU-accelerated graph algorithms via RAPIDS cuGraph to discover structure in your knowledge base.

---

## Overview

NeuralForge maintains a directed knowledge graph where:
- **Nodes** represent experts, concepts, techniques, tools, datasets, models, papers, and institutions
- **Edges** represent relationships (18 types) with temporal validity, confidence scores, and provenance

Graph algorithms run on the GPU via cuGraph when available, with automatic networkx fallback for CPU-only environments.

---

## PageRank

### What It Does

PageRank scores every node by its structural importance in the graph. A node is important if it is connected to other important nodes.

### NeuralForge Use: Expert Authority

PageRank is combined with topic-specific edge counting to produce authority rankings:

```
authority_score = pagerank(expert) * count(topic_edges(expert))
```

This means an expert with high PageRank AND many edges related to a specific topic ranks highest for that topic.

### API

```bash
# Topic-specific authority ranking
curl "http://localhost:8090/api/v1/graph/authority?topic=quantization&limit=10"

# Global PageRank scores
curl "http://localhost:8090/api/v1/graph/pagerank"
```

### Example Output

```
Expert Authority Rankings: "quantization"

  Rank   Expert                         Score        Edges
  ----   ------                         -----        -----
  1      Tim Dettmers                   0.042318     12
  2      Elias Frantar                  0.031205     8
  3      Song Han                       0.028901     7
```

### Implementation

```python
# cuGraph path (GPU)
result = cugraph.pagerank(graph)

# networkx path (CPU fallback)
scores = nx.pagerank(graph, personalization=weights)
```

---

## Louvain Community Detection

### What It Does

The Louvain algorithm partitions the graph into communities by maximizing modularity -- groups of nodes that are more densely connected to each other than to the rest of the graph.

### NeuralForge Use: Expert Clustering

Discovers natural expert clusters:
- Researchers who cite each other
- Authors covering the same domain
- Institutions with shared methodologies

### API

```bash
curl "http://localhost:8090/api/v1/graph/communities"
```

### Example Output

```
Detected 4 communities (23 nodes total):

  Community 0 (8 members):
    - Geoffrey Hinton
    - Yann LeCun
    - Yoshua Bengio
    - Ilya Sutskever

  Community 1 (6 members):
    - Tim Dettmers
    - Elias Frantar
    - Song Han

  Community 2 (5 members):
    - Andrej Karpathy
    - Grant Sanderson
    - Jeremy Howard
```

### Implementation

```python
# cuGraph path (GPU)
partitions, modularity = cugraph.louvain(graph)

# networkx path (CPU fallback)
communities = nx.community.louvain_communities(graph.to_undirected(), seed=42)
```

---

## BFS Traversal

### What It Does

Breadth-first search walks the graph from a starting node, visiting all reachable nodes up to a specified depth.

### NeuralForge Use: Context Enrichment

When building context for a query, BFS traversal from matching expert/concept nodes gathers related knowledge:

```
"Show me everything connected to LoRA within 2 hops"
```

### Filtering

Traversals can be filtered by:
- **Edge types** -- only follow specific relationship types
- **Temporal window** -- only traverse edges valid at a given date

### API

```bash
curl "http://localhost:8090/api/v1/graph/traverse?node_id=abc123&depth=2&edge_types=agrees_with,contradicts"
```

### Implementation

```python
result = engine.traverse(
    node_id="abc123",
    depth=2,
    edge_types=[EdgeType.agrees_with, EdgeType.contradicts],
    as_of="2025-01-01",
)
# result.nodes -- visited nodes
# result.edges -- traversed edges
# result.depth -- max depth reached
```

---

## Shortest Path

### What It Does

Finds the shortest directed path between two nodes in the graph.

### NeuralForge Use: Relationship Discovery

"How is Expert A connected to Concept B?" -- traces the chain of relationships.

### API

```bash
curl "http://localhost:8090/api/v1/graph/path?source=node_a&target=node_b"
```

### Implementation

```python
path = engine.shortest_path(source_id, target_id)
# Returns list of node IDs from source to target, or empty if no path
```

---

## Contradiction Detection

### What It Does

Scans the graph for conflicting edges -- places where one expert contradicts another, or where agreement and disagreement coexist between the same nodes.

### How It Works

1. Find all `contradicts` and `incompatible_with` edges
2. Find all `agrees_with` edges
3. Pair them when they share a common node
4. If no pairs found, report standalone contradiction edges

### API

```bash
# All contradictions
curl "http://localhost:8090/api/v1/graph/contradictions"

# Topic-filtered
curl "http://localhost:8090/api/v1/graph/contradictions?topic=quantization"
```

### Example Output

```
Found 2 contradictions on 'quantization':

  1. Contradiction between contradicts and agrees_with
     Edge A: Tim Dettmers --[contradicts]--> Song Han
     Edge B: Elias Frantar --[agrees_with]--> Tim Dettmers
     Confidence: 0.87

  2. Explicit incompatible_with relationship
     Edge A: Method A --[incompatible_with]--> Method B
     Confidence: 0.92
```

---

## Temporal Filtering

### What It Does

Every edge in the NeuralForge graph has `valid_from` and `valid_to` timestamps. This enables time-travel queries:

```python
# Rebuild the graph as it existed on a specific date
engine.get_graph_as_of("2024-06-01")

# Find edges created since a date
changes = engine.find_changes_since("2025-01-01", topic="LoRA")
```

### Use Cases

- **Knowledge evolution** -- "What was the consensus on quantization in 2023?"
- **Change tracking** -- "What new relationships were discovered this month?"
- **Supersession** -- when a new paper supersedes an old one, the old edge gets a `valid_to`

---

## Auto-Discovery

### What It Does

Every 6 hours (configurable), the discovery worker:

1. Selects pairs of experts with shared topics
2. Retrieves relevant chunks from Qdrant for each expert
3. Sends paired excerpts to NIM for relationship classification
4. Creates graph edges for relationships above the confidence floor (default 0.6)

### Classification Types

| Relationship | Graph Edge |
|:---|:---|
| `agrees` | `agrees_with` |
| `disagrees` | `contradicts` |
| `extends` | `derived_from` |
| `unrelated` | No edge created |

### Configuration

```bash
# In .env
DISCOVERY_INTERVAL_HOURS=6
DISCOVERY_PAIRS_PER_RUN=20
DISCOVERY_CONFIDENCE_FLOOR=0.6
```

---

## Performance: cuGraph vs networkx

| Operation | 1K nodes | 10K nodes | 100K nodes |
|:---|:---:|:---:|:---:|
| PageRank (cuGraph) | <1ms | ~5ms | ~50ms |
| PageRank (networkx) | ~10ms | ~500ms | ~30s |
| Louvain (cuGraph) | <1ms | ~10ms | ~100ms |
| Louvain (networkx) | ~50ms | ~2s | ~120s |

*Approximate. Actual performance depends on graph density and GPU model.*

The crossover point where cuGraph becomes faster than networkx is typically around 1,000-5,000 nodes. Below that, networkx overhead is negligible. NeuralForge uses cuGraph by default when available, regardless of graph size, for consistency.
