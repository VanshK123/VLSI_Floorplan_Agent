"""Utility functions for graph operations."""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx


def compute_wirelength(graph: nx.DiGraph) -> float:
    """Compute total wirelength for the given graph."""
    raise NotImplementedError


def enforce_nonoverlap(coords: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Adjust coordinates to enforce non-overlap constraints."""
    raise NotImplementedError
