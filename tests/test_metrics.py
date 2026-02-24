"""Unit tests for clinical metric utilities."""

from __future__ import annotations

from typing import cast

import numpy as np

from breast_cancer_ai.metrics import (
    compute_binary_metrics,
    select_threshold_for_sensitivity,
)


def test_compute_binary_metrics_basic_case() -> None:
    """Binary metrics should be exact on a simple perfectly classified sample."""
    y_true = np.array([1, 1, 0, 0], dtype=int)
    y_prob = np.array([0.95, 0.75, 0.40, 0.05], dtype=float)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_select_threshold_for_sensitivity() -> None:
    """Threshold selector should satisfy the requested sensitivity when feasible."""
    y_true = np.array([1, 1, 1, 0, 0, 0], dtype=int)
    y_prob = np.array([0.9, 0.8, 0.55, 0.65, 0.35, 0.2], dtype=float)

    threshold, details = select_threshold_for_sensitivity(
        y_true, y_prob, target_sensitivity=1.0
    )
    details_typed = cast(dict[str, dict[str, float]], details)

    assert 0.0 <= threshold <= 1.0
    assert details_typed["chosen_metrics"]["sensitivity"] >= 1.0
