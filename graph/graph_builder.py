"""Builds a networkx graph from parsed design data."""
from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from .data_structures import Graph


class GraphBuilder:
    """Constructs graphs for GNN processing."""

    def build(
        self,
        def_data: Dict[str, Any],
        lef_data: Dict[str, Any],
        netlist_data: Dict[str, Any],
    ) -> nx.DiGraph:
        """Create a directed graph with attributes."""
        raise NotImplementedError
