"""Fitness evaluation functions for evolutionary search."""
from __future__ import annotations

from typing import Dict, Tuple


def area(placement: Dict[str, Tuple[float, float]]) -> float:
    """Compute the area of the placement."""
    raise NotImplementedError


def delay(placement: Dict[str, Tuple[float, float]], netlist) -> float:
    """Estimate timing delay based on placement and netlist distances."""
    raise NotImplementedError
