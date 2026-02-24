"""Feedforward MLP architecture for tabular binary classification."""

from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """Fully-connected MLP with optional batch normalisation and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw logits for the positive class, shape ``(batch,)``."""
        logits = self.network(features)
        return logits.squeeze(-1)
