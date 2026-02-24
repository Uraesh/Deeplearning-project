from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SystemConfig:
    seed: int = 42
    num_threads: int = 4
    cleanup_every_epoch: bool = True


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    target_col: str = "diagnosis"
    id_col: str = "id"


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...] = (48, 24)
    dropout: float = 0.25
    use_batch_norm: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 220
    learning_rate: float = 0.001
    weight_decay: float = 0.0006
    early_stopping_patience: int = 16
    min_delta: float = 0.0001
    sensitivity_target: float = 0.97
    cv_folds: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 6
    min_learning_rate: float = 1e-5


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: str = "models"
    run_name_prefix: str = "wisconsin_torch"


@dataclass(frozen=True)
class TrainConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig(path="data.csv"))
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def _ensure_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return value  # type: ignore[return-value]


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    system_raw = _ensure_mapping(raw.get("system", {}), "system")
    dataset_raw = _ensure_mapping(raw.get("dataset", {}), "dataset")
    split_raw = _ensure_mapping(raw.get("split", {}), "split")
    model_raw = _ensure_mapping(raw.get("model", {}), "model")
    training_raw = _ensure_mapping(raw.get("training", {}), "training")
    artifacts_raw = _ensure_mapping(raw.get("artifacts", {}), "artifacts")

    if "path" not in dataset_raw:
        raise ValueError("dataset.path is required.")

    cfg = TrainConfig(
        system=SystemConfig(
            seed=int(system_raw.get("seed", 42)),
            num_threads=int(system_raw.get("num_threads", 4)),
            cleanup_every_epoch=bool(system_raw.get("cleanup_every_epoch", True)),
        ),
        dataset=DatasetConfig(
            path=str(dataset_raw["path"]),
            target_col=str(dataset_raw.get("target_col", "diagnosis")),
            id_col=str(dataset_raw.get("id_col", "id")),
        ),
        split=SplitConfig(
            test_size=float(split_raw.get("test_size", 0.2)),
        ),
        model=ModelConfig(
            hidden_dims=tuple(int(x) for x in model_raw.get("hidden_dims", [48, 24])),
            dropout=float(model_raw.get("dropout", 0.25)),
            use_batch_norm=bool(model_raw.get("use_batch_norm", True)),
        ),
        training=TrainingConfig(
            batch_size=int(training_raw.get("batch_size", 32)),
            max_epochs=int(training_raw.get("max_epochs", 220)),
            learning_rate=float(training_raw.get("learning_rate", 0.001)),
            weight_decay=float(training_raw.get("weight_decay", 0.0006)),
            early_stopping_patience=int(training_raw.get("early_stopping_patience", 16)),
            min_delta=float(training_raw.get("min_delta", 0.0001)),
            sensitivity_target=float(training_raw.get("sensitivity_target", 0.97)),
            cv_folds=int(training_raw.get("cv_folds", 5)),
            scheduler_factor=float(training_raw.get("scheduler_factor", 0.5)),
            scheduler_patience=int(training_raw.get("scheduler_patience", 6)),
            min_learning_rate=float(training_raw.get("min_learning_rate", 1e-5)),
        ),
        artifacts=ArtifactConfig(
            output_dir=str(artifacts_raw.get("output_dir", "models")),
            run_name_prefix=str(artifacts_raw.get("run_name_prefix", "wisconsin_torch")),
        ),
    )

    if not 0.0 < cfg.split.test_size < 1.0:
        raise ValueError("split.test_size must be in (0, 1).")
    if cfg.system.num_threads < 1:
        raise ValueError("system.num_threads must be >= 1.")
    if not 0.0 <= cfg.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1).")
    if cfg.training.batch_size < 1:
        raise ValueError("training.batch_size must be >= 1.")
    if cfg.training.max_epochs < 1:
        raise ValueError("training.max_epochs must be >= 1.")
    if cfg.training.cv_folds < 3:
        raise ValueError("training.cv_folds must be >= 3.")
    if cfg.training.early_stopping_patience < 1:
        raise ValueError("training.early_stopping_patience must be >= 1.")
    if not 0.0 < cfg.training.sensitivity_target <= 1.0:
        raise ValueError("training.sensitivity_target must be in (0, 1].")
    if not 0.0 < cfg.training.scheduler_factor < 1.0:
        raise ValueError("training.scheduler_factor must be in (0, 1).")

    return cfg
