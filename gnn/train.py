"""Training loop for the GNN model."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from .model import FloorplanGNN


def train(config: Dict[str, Any], data_loader) -> None:
    """Train the model and save checkpoints."""
    raise NotImplementedError
