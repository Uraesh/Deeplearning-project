"""Unit tests for dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from breast_cancer_ai.data import load_tabular_dataset


def test_load_tabular_dataset_sanitizes_columns_and_maps_target(tmp_path: Path) -> None:
    """Dataset loader should sanitize names, drop IDs, and map labels."""
    sample_df = pd.DataFrame(
        {
            "id": [1, 2],
            "diagnosis": ["M", "B"],
            "concave points_mean": [0.12, 0.07],
            "radius_mean": [17.0, 11.1],
        }
    )
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)

    features, labels, metadata = load_tabular_dataset(
        path=csv_path,
        target_col="diagnosis",
        id_col="id",
    )

    assert "id" not in features.columns
    assert "concave_points_mean" in features.columns
    assert labels.tolist() == [1, 0]
    assert metadata["target_col"] == "diagnosis"
