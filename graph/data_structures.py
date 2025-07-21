"""Custom graph data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """Representation of a graph node."""

    id: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Representation of a graph edge."""

    source: str
    target: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    """Simple adjacency-based graph representation."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = Node(id=node_id, attrs=attrs)

    def add_edge(self, source: str, target: str, **attrs: Any) -> None:
        self.edges.append(Edge(source=source, target=target, attrs=attrs))
