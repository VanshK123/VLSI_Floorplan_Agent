"""Evolutionary algorithm orchestration using NSGA-II."""
from __future__ import annotations

from typing import Dict, List

from ..sta_integration.sta_runner import STARunner
from .fitness import area, delay


class GeneticAlgorithm:
    """NSGA-II optimizer for floorplanning."""

    def __init__(self, config: Dict[str, int], sta_runner: STARunner) -> None:
        self.config = config
        self.sta_runner = sta_runner

    def run(self, graph, initial_population: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Execute optimization and return Pareto front placements."""
        raise NotImplementedError
