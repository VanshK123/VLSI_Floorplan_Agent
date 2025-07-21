"""Graph neural network model for floorplanning."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class FloorplanGNN(nn.Module):
    """Multi-layer GAT with coordinate regression head."""

    def __init__(self, num_layers: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        raise NotImplementedError

    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a forward pass and output coordinates."""
        raise NotImplementedError
