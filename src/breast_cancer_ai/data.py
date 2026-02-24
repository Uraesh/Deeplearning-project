"""Dataset loading and split helpers for breast cancer tabular data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd

LABEL_MAP = {"M": 1, "B": 0}
_PANDAS_ANY = cast(Any, pd)


class DroppedColumns(TypedDict):
    """Columns removed during cleanup with their removal reason."""

    unnamed: list[str]
    all_nan: list[str]


class DatasetMetadata(TypedDict):
    """Metadata returned by the dataset loader."""

    feature_names: list[str]
    target_col: str
    id_col: str | None
    column_rename_map: dict[str, str]
    dropped_columns: DroppedColumns


def _read_csv_typed(path: Path) -> pd.DataFrame:
    """Read CSV with an explicit DataFrame return type for static analyzers."""
    return cast(pd.DataFrame, _PANDAS_ANY.read_csv(str(path)))


def _to_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric with coercion, keeping a Series return type."""
    return cast(pd.Series, _PANDAS_ANY.to_numeric(series, errors="coerce"))


def sanitize_column_name(name: str) -> str:
    """Normalize a column name to a consistent snake-like format."""
    return name.strip().replace(" ", "_")


def _stratified_split_indices(
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices preserving class proportions."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")

    label_array = labels.to_numpy()
    classes, class_counts = np.unique(label_array, return_counts=True)

    if len(classes) < 2:
        raise ValueError("At least two classes are required for stratified splitting.")

    rng = np.random.default_rng(seed=random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_value, class_count in zip(classes, class_counts, strict=True):
        class_indices = np.where(label_array == class_value)[0]
        rng.shuffle(class_indices)

        class_test_count = int(round(float(class_count) * test_size))
        class_test_count = max(1, class_test_count)
        class_test_count = min(class_test_count, int(class_count) - 1)

        test_indices.extend(class_indices[:class_test_count].tolist())
        train_indices.extend(class_indices[class_test_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def load_tabular_dataset(
    path: str | Path,
    target_col: str = "diagnosis",
    id_col: str | None = "id",
) -> tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
    """Load the CSV dataset, encode labels, and return cleaned features and metadata."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataframe = _read_csv_typed(dataset_path)
    rename_map = {column: sanitize_column_name(column) for column in dataframe.columns}
    dataframe = dataframe.rename(columns=rename_map)

    target_clean = rename_map.get(target_col, sanitize_column_name(target_col))
    id_clean = rename_map.get(id_col, sanitize_column_name(id_col)) if id_col else None

    if target_clean not in dataframe.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    target_raw = dataframe[target_clean]
    if target_raw.dtype == "O":
        target_values = target_raw.astype(str).str.strip().map(LABEL_MAP)
        if target_values.isna().any():
            bad_values = sorted(set(target_raw.astype(str)) - set(LABEL_MAP))
            raise ValueError(f"Unsupported target labels: {bad_values}")
    else:
        target_values = target_raw.astype(int)
        if not set(target_values.unique()).issubset({0, 1}):
            raise ValueError("Numeric target labels must be 0/1.")

    drop_columns = [target_clean]
    if id_clean and id_clean in dataframe.columns:
        drop_columns.append(id_clean)

    feature_frame = dataframe.drop(columns=drop_columns)

    unnamed_cols = [col for col in feature_frame.columns if col.lower().startswith("unnamed")]
    if unnamed_cols:
        feature_frame = feature_frame.drop(columns=unnamed_cols)

    feature_frame = feature_frame.apply(_to_numeric_series)

    all_nan_cols = [col for col in feature_frame.columns if feature_frame[col].isna().all()]
    if all_nan_cols:
        feature_frame = feature_frame.drop(columns=all_nan_cols)

    if feature_frame.empty:
        raise ValueError("No usable feature columns remain after cleaning.")

    metadata: DatasetMetadata = {
        "feature_names": list(feature_frame.columns),
        "target_col": target_clean,
        "id_col": id_clean,
        "column_rename_map": rename_map,
        "dropped_columns": {
            "unnamed": unnamed_cols,
            "all_nan": all_nan_cols,
        },
    }
    return feature_frame, target_values.astype(int), metadata


def split_train_test(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    train_idx, test_idx = _stratified_split_indices(
        labels=labels,
        test_size=test_size,
        random_state=random_state,
    )

    train_features = features.iloc[train_idx].reset_index(drop=True)
    test_features = features.iloc[test_idx].reset_index(drop=True)
    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    test_labels = labels.iloc[test_idx].reset_index(drop=True)

    return train_features, test_features, train_labels, test_labels


def split_train_val_test(
    features: pd.DataFrame,
    labels: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create stratified train/validation/test splits for tabular classification."""
    train_val_features, test_features, train_val_labels, test_labels = split_train_test(
        features=features,
        labels=labels,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, val_idx = _stratified_split_indices(
        labels=train_val_labels,
        test_size=val_size,
        random_state=random_state,
    )

    train_features = train_val_features.iloc[train_idx].reset_index(drop=True)
    val_features = train_val_features.iloc[val_idx].reset_index(drop=True)
    train_labels = train_val_labels.iloc[train_idx].reset_index(drop=True)
    val_labels = train_val_labels.iloc[val_idx].reset_index(drop=True)

    return {
        "X_train": train_features,
        "X_val": val_features,
        "X_test": test_features,
        "y_train": train_labels,
        "y_val": val_labels,
        "y_test": test_labels,
    }
