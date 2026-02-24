"""Unit tests for model architecture basics."""

from __future__ import annotations

import torch

from breast_cancer_ai.model import TabularMLP


def test_tabular_mlp_forward_shape() -> None:
    """The model forward pass should output one logit per input row."""
    model = TabularMLP(input_dim=30, hidden_dims=(32, 16), dropout=0.1)
    features = torch.randn(4, 30)
    output = model(features)
    assert output.shape == (4,)
