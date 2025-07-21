"""Utilities for running model inference."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from .model import FloorplanGNN


def run_inference(model_path: Path, graph) -> Dict[str, Tuple[float, float]]:
    """Load a model checkpoint and predict coordinates."""
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    raise NotImplementedError
