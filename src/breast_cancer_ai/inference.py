from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from breast_cancer_ai.model import TabularMLP
from typing import TypedDict, Any

class Artifact(TypedDict):
    state_dict: dict[str, Any]
    model_params: dict[str, Any]
    feature_names: list[str]
    preprocessing: dict[str, Any]
    threshold: float
    model_version: str
    created_at_utc: str | None


class TorchPredictor:
    def __init__(self, artifact: Artifact) -> None:
        required = {
            "state_dict",
            "model_params",
            "feature_names",
            "preprocessing",
            "threshold",
            "model_version",
        }
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"Invalid artifact. Missing keys: {sorted(missing)}")

        model_params = artifact["model_params"]
        self.feature_names = list(artifact["feature_names"])
        self.threshold = float(artifact["threshold"])
        self.model_version = str(artifact["model_version"])
        self.created_at_utc = str(artifact.get("created_at_utc", ""))

        preprocessing = artifact["preprocessing"]
        self.imputer_median = np.asarray(preprocessing["imputer_median"], dtype=np.float32)
        self.scaler_mean = np.asarray(preprocessing["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.asarray(preprocessing["scaler_scale"], dtype=np.float32)

        self.imputer_median = np.where(np.isnan(self.imputer_median), 0.0, self.imputer_median)
        self.scaler_scale = np.where(self.scaler_scale == 0.0, 1.0, self.scaler_scale)

        self.model = TabularMLP(
            input_dim=int(model_params["input_dim"]),
            hidden_dims=tuple(int(x) for x in model_params["hidden_dims"]),
            dropout=float(model_params["dropout"]),
            use_batch_norm=bool(model_params.get("use_batch_norm", False)),
        )
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path) -> "TorchPredictor":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        artifact = torch.load(artifact_path, map_location="cpu")
        return cls(artifact)

    def _prepare(self, records: list[dict[str, float]]) -> np.ndarray:
        if not records:
            raise ValueError("At least one record is required.")

        frame = pd.DataFrame(records)
        missing = [col for col in self.feature_names if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        features = frame[self.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        nan_mask = np.isnan(features)
        if nan_mask.any():
            features[nan_mask] = np.take(self.imputer_median, np.where(nan_mask)[1])

        features = (features - self.scaler_mean) / self.scaler_scale
        return features

    def predict_from_records(self, records: list[dict[str, float]]) -> list[dict[str, object]]:
        features = self._prepare(records)
        with torch.no_grad():
            logits = self.model(torch.tensor(features, dtype=torch.float32))
            probabilities = torch.sigmoid(logits).numpy()

        predictions = (probabilities >= self.threshold).astype(int)

        outputs: list[dict[str, object]] = []
        for probability, prediction in zip(probabilities, predictions, strict=True):
            outputs.append(
                {
                    "probability_malignant": float(probability),
                    "prediction": int(prediction),
                    "prediction_label": "MALIGNANT" if prediction == 1 else "BENIGN",
                    "threshold": self.threshold,
                    "model_version": self.model_version,
                }
            )
        return outputs
