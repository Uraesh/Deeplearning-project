"""Training pipeline with CV-OOF thresholding and cache-aware artifact reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from breast_cancer_ai.config import TrainConfig, load_train_config
from breast_cancer_ai.data import load_tabular_dataset, split_train_test
from breast_cancer_ai.metrics import (
    compute_binary_metrics,
    select_threshold_for_sensitivity,
)
from breast_cancer_ai.model import TabularMLP
from breast_cancer_ai.utils import cleanup_memory, configure_runtime, set_seed

FloatArray = npt.NDArray[np.floating[Any]]
IntArray = npt.NDArray[np.signedinteger[Any]]
TensorBatch = tuple[torch.Tensor, torch.Tensor]
BatchLoader = DataLoader[TensorBatch]


def _create_run_dir(output_dir: Path, prefix: str) -> Path:
    """Create a timestamped run directory under the artifact root."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize a mapping as formatted UTF-8 JSON."""
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _build_report(payload: Mapping[str, Any]) -> str:
    """Render a markdown report from metrics payload."""
    cv_metrics = cast(Mapping[str, Any], payload["cv_oof_metrics"])
    test_metrics = cast(Mapping[str, Any], payload["test_metrics"])
    trainval_metrics = cast(Mapping[str, Any], payload["trainval_metrics"])
    overfit_gap = cast(Mapping[str, Any], payload["overfit_gap"])
    threshold_details = cast(Mapping[str, Any], payload["threshold_details"])

    return (
        "# Training Report\n\n"
        f"- Model version: `{payload['model_version']}`\n"
        f"- Created at UTC: `{payload['created_at_utc']}`\n"
        f"- Threshold selected (OOF): `{float(payload['threshold']):.6f}`\n"
        f"- Threshold reason: `{threshold_details['reason']}`\n"
        f"- CV folds: `{int(payload['cv_folds'])}`\n"
        f"- Final training epochs: `{int(payload['final_training_epochs'])}`\n"
        f"- Training signature: `{payload['training_signature']}`\n\n"
        "## CV OOF\n"
        f"- Sensitivity: `{float(cv_metrics['sensitivity']):.4f}`\n"
        f"- Specificity: `{float(cv_metrics['specificity']):.4f}`\n"
        f"- ROC-AUC: `{float(cv_metrics['roc_auc']):.4f}`\n"
        f"- PR-AUC: `{float(cv_metrics['pr_auc']):.4f}`\n\n"
        "## Train+Val (Final model)\n"
        f"- Sensitivity: `{float(trainval_metrics['sensitivity']):.4f}`\n"
        f"- Specificity: `{float(trainval_metrics['specificity']):.4f}`\n"
        f"- ROC-AUC: `{float(trainval_metrics['roc_auc']):.4f}`\n"
        f"- PR-AUC: `{float(trainval_metrics['pr_auc']):.4f}`\n\n"
        "## Test\n"
        f"- Sensitivity: `{float(test_metrics['sensitivity']):.4f}`\n"
        f"- Specificity: `{float(test_metrics['specificity']):.4f}`\n"
        f"- ROC-AUC: `{float(test_metrics['roc_auc']):.4f}`\n"
        f"- PR-AUC: `{float(test_metrics['pr_auc']):.4f}`\n\n"
        "## Overfitting Gap (trainval - test)\n"
        f"- Sensitivity gap: `{float(overfit_gap['sensitivity_gap']):.4f}`\n"
        f"- Specificity gap: `{float(overfit_gap['specificity_gap']):.4f}`\n"
        f"- ROC-AUC gap: `{float(overfit_gap['roc_auc_gap']):.4f}`\n"
    )


def _dataset_signature(dataset_path: Path) -> dict[str, object]:
    """Return minimal filesystem metadata used by cache signature logic."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    stat = dataset_path.stat()
    return {
        "path": str(dataset_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _compute_training_signature(
    cfg: TrainConfig,
    dataset_signature: Mapping[str, object],
) -> str:
    """Compute stable hash signature from config and dataset metadata."""
    payload: dict[str, Any] = {
        "trainer": "torch_mlp_cv_oof_v2",
        "config": asdict(cfg),
        "dataset_signature": dict(dataset_signature),
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _try_reuse_cached_training(
    output_dir: Path,
    training_signature: str,
) -> dict[str, object] | None:
    """Reuse latest artifacts when their signature matches current training inputs."""
    latest_dir = output_dir / "latest"
    model_path = latest_dir / "model.pt"
    metrics_path = latest_dir / "metrics.json"
    manifest_path = latest_dir / "cache_manifest.json"

    if not (model_path.exists() and metrics_path.exists() and manifest_path.exists()):
        return None

    try:
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if not isinstance(manifest_raw, dict):
        return None

    manifest = cast(dict[str, Any], manifest_raw)
    if manifest.get("training_signature") != training_signature:
        return None

    return {
        "run_dir": str(manifest.get("run_dir", latest_dir)),
        "latest_model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "cache_hit": True,
        "training_signature": training_signature,
        "model_version": str(manifest.get("model_version", "unknown")),
    }


def _make_loader(
    features: FloatArray,
    targets: IntArray,
    batch_size: int,
    shuffle: bool,
) -> BatchLoader:
    """Build a CPU DataLoader from NumPy arrays."""
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
    return cast(BatchLoader, loader)


def _predict_proba(model: TabularMLP, features: FloatArray) -> FloatArray:
    """Return sigmoid probabilities for the positive class."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64)


def _evaluate(
    model: TabularMLP,
    loader: BatchLoader,
    criterion: nn.Module,
) -> tuple[float, IntArray, FloatArray]:
    """Evaluate loss and predictions over a loader."""
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_y)
            all_logits.append(logits.cpu())
            all_targets.append(batch_y.cpu())

    y_logits = torch.cat(all_logits)
    y_prob = np.asarray(torch.sigmoid(y_logits).numpy(), dtype=np.float64)
    y_true = np.asarray(torch.cat(all_targets).numpy().astype(np.int64), dtype=np.int64)
    avg_loss = total_loss / max(1, len(y_true))
    return avg_loss, y_true, y_prob


def _build_model(cfg: TrainConfig, input_dim: int) -> TabularMLP:
    """Instantiate the tabular MLP according to configuration."""
    return TabularMLP(
        input_dim=input_dim,
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm,
    )


def _train_with_early_stopping(
    cfg: TrainConfig,
    x_train: FloatArray,
    y_train: IntArray,
    x_val: FloatArray,
    y_val: IntArray,
) -> tuple[TabularMLP, int, float]:
    """Train one fold with early stopping and LR scheduling."""
    train_loader = _make_loader(x_train, y_train, cfg.training.batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, cfg.training.batch_size, shuffle=False)

    model = _build_model(cfg, input_dim=x_train.shape[1])

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer: Any = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.scheduler_patience,
        min_lr=cfg.training.min_learning_rate,
    )

    best_val_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    best_epoch = 1
    patience_count = 0

    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, _, _ = _evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < (best_val_loss - cfg.training.min_delta):
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        if cfg.system.cleanup_every_epoch:
            cleanup_memory()

        if patience_count >= cfg.training.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_val_loss


def _train_fixed_epochs(
    cfg: TrainConfig,
    x_train: FloatArray,
    y_train: IntArray,
    epochs: int,
) -> TabularMLP:
    """Train final model for a fixed number of epochs."""
    model = _build_model(cfg, input_dim=x_train.shape[1])

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight = torch.tensor(negatives / max(positives, 1.0), dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer: Any = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    train_loader = _make_loader(x_train, y_train, cfg.training.batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if cfg.system.cleanup_every_epoch:
            cleanup_memory()

    model.eval()
    return model


def _cross_validated_oof_predictions(
    cfg: TrainConfig,
    x_train_val_frame: pd.DataFrame,
    y_train_val_series: pd.Series,
) -> tuple[FloatArray, list[dict[str, object]], int]:
    """Compute out-of-fold probabilities and fold diagnostics."""
    y_values = np.asarray(y_train_val_series.to_numpy(dtype=np.int64), dtype=np.int64)
    oof_prob = np.zeros(len(y_values), dtype=np.float64)
    fold_reports: list[dict[str, object]] = []
    fold_best_epochs: list[int] = []

    skf = StratifiedKFold(
        n_splits=cfg.training.cv_folds,
        shuffle=True,
        random_state=cfg.system.seed,
    )

    for fold_idx, (fit_idx, val_idx) in enumerate(
        skf.split(x_train_val_frame, y_values), start=1
    ):
        x_fit_frame = cast(pd.DataFrame, x_train_val_frame.iloc[fit_idx])
        x_val_frame = cast(pd.DataFrame, x_train_val_frame.iloc[val_idx])
        y_fit = np.asarray(y_values[fit_idx], dtype=np.int64)
        y_val = np.asarray(y_values[val_idx], dtype=np.int64)

        imputer = SimpleImputer(strategy="median")
        x_fit_imputed = np.asarray(
            imputer.fit_transform(x_fit_frame),  # type: ignore[no-untyped-call]
            dtype=np.float32,
        )
        x_val_imputed = np.asarray(imputer.transform(x_val_frame), dtype=np.float32)

        scaler = StandardScaler()
        x_fit = np.asarray(
            scaler.fit_transform(x_fit_imputed),  # type: ignore[no-untyped-call]
            dtype=np.float32,
        )
        x_val = np.asarray(scaler.transform(x_val_imputed), dtype=np.float32)

        model, best_epoch, best_val_loss = _train_with_early_stopping(
            cfg,
            x_fit,
            y_fit,
            x_val,
            y_val,
        )
        val_prob = _predict_proba(model, x_val)
        oof_prob[val_idx] = val_prob

        fold_threshold, _ = select_threshold_for_sensitivity(
            y_true=y_val,
            y_prob=val_prob,
            target_sensitivity=cfg.training.sensitivity_target,
        )
        fold_metrics = compute_binary_metrics(
            y_true=y_val,
            y_prob=val_prob,
            threshold=fold_threshold,
        )

        fold_reports.append(
            {
                "fold": fold_idx,
                "fit_size": int(len(fit_idx)),
                "val_size": int(len(val_idx)),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "fold_threshold": float(fold_threshold),
                "fold_metrics": fold_metrics,
            }
        )
        fold_best_epochs.append(int(best_epoch))

    final_epochs = max(1, int(round(float(np.median(fold_best_epochs)))))
    return np.asarray(oof_prob, dtype=np.float64), fold_reports, final_epochs


def run_training(config_path: str, force_retrain: bool = False) -> dict[str, object]:
    """Run full training pipeline and persist model artifacts."""
    cfg = load_train_config(config_path)
    set_seed(cfg.system.seed)
    configure_runtime(cfg.system.num_threads)

    output_dir = Path(cfg.artifacts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_signature = _dataset_signature(Path(cfg.dataset.path))
    training_signature = _compute_training_signature(cfg, dataset_signature)

    if not force_retrain:
        cached = _try_reuse_cached_training(output_dir, training_signature)
        if cached is not None:
            return cached

    run_dir = _create_run_dir(output_dir, cfg.artifacts.run_name_prefix)

    feature_frame, labels, data_meta = load_tabular_dataset(
        path=cfg.dataset.path,
        target_col=cfg.dataset.target_col,
        id_col=cfg.dataset.id_col,
    )

    x_train_val_frame, x_test_frame, y_train_val_series, y_test_series = (
        split_train_test(
            features=feature_frame,
            labels=labels,
            test_size=cfg.split.test_size,
            random_state=cfg.system.seed,
        )
    )

    oof_prob, cv_folds_report, final_training_epochs = _cross_validated_oof_predictions(
        cfg=cfg,
        x_train_val_frame=x_train_val_frame,
        y_train_val_series=y_train_val_series,
    )
    y_train_val_np = np.asarray(
        y_train_val_series.to_numpy(dtype=np.int64), dtype=np.int64
    )

    threshold, threshold_details = select_threshold_for_sensitivity(
        y_true=y_train_val_np,
        y_prob=oof_prob,
        target_sensitivity=cfg.training.sensitivity_target,
    )
    cv_oof_metrics = compute_binary_metrics(
        y_true=y_train_val_np,
        y_prob=oof_prob,
        threshold=threshold,
    )

    final_imputer = SimpleImputer(strategy="median")
    x_train_val_imputed = np.asarray(
        final_imputer.fit_transform(x_train_val_frame),  # type: ignore[no-untyped-call]
        dtype=np.float32,
    )
    x_test_imputed = np.asarray(final_imputer.transform(x_test_frame), dtype=np.float32)

    final_scaler = StandardScaler()
    x_train_val = np.asarray(
        final_scaler.fit_transform(x_train_val_imputed),  # type: ignore[no-untyped-call]
        dtype=np.float32,
    )
    x_test = np.asarray(final_scaler.transform(x_test_imputed), dtype=np.float32)

    final_model = _train_fixed_epochs(
        cfg=cfg,
        x_train=x_train_val,
        y_train=y_train_val_np,
        epochs=final_training_epochs,
    )

    train_val_prob = _predict_proba(final_model, x_train_val)
    test_prob = _predict_proba(final_model, x_test)

    y_test_np = np.asarray(y_test_series.to_numpy(dtype=np.int64), dtype=np.int64)
    trainval_metrics = compute_binary_metrics(
        y_true=y_train_val_np,
        y_prob=train_val_prob,
        threshold=threshold,
    )
    test_metrics = compute_binary_metrics(
        y_true=y_test_np,
        y_prob=test_prob,
        threshold=threshold,
    )

    overfit_gap: dict[str, float] = {
        "sensitivity_gap": float(
            trainval_metrics["sensitivity"] - test_metrics["sensitivity"]
        ),
        "specificity_gap": float(
            trainval_metrics["specificity"] - test_metrics["specificity"]
        ),
        "roc_auc_gap": float(trainval_metrics["roc_auc"] - test_metrics["roc_auc"]),
    }

    if final_imputer.statistics_.size == 0:
        raise RuntimeError("Imputer statistics are missing after fit.")
    if final_scaler.mean_ is None or final_scaler.scale_ is None:
        raise RuntimeError("Scaler parameters are missing after fit.")

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at_utc = datetime.now(timezone.utc).isoformat()

    artifact: dict[str, object] = {
        "state_dict": final_model.state_dict(),
        "model_params": {
            "input_dim": int(x_train_val.shape[1]),
            "hidden_dims": list(cfg.model.hidden_dims),
            "dropout": float(cfg.model.dropout),
            "use_batch_norm": bool(cfg.model.use_batch_norm),
        },
        "feature_names": data_meta["feature_names"],
        "preprocessing": {
            "imputer_median": final_imputer.statistics_.tolist(),
            "scaler_mean": final_scaler.mean_.tolist(),
            "scaler_scale": final_scaler.scale_.tolist(),
        },
        "threshold": float(threshold),
        "model_version": model_version,
        "created_at_utc": created_at_utc,
        "target_definition": {"positive": "MALIGNANT", "negative": "BENIGN"},
        "config": asdict(cfg),
        "threshold_details": threshold_details,
        "dataset_meta": data_meta,
        "dataset_signature": dict(dataset_signature),
        "training_signature": training_signature,
        "final_training_epochs": int(final_training_epochs),
    }

    report_payload: dict[str, object] = {
        "model_version": model_version,
        "created_at_utc": created_at_utc,
        "threshold": float(threshold),
        "threshold_details": threshold_details,
        "cv_folds": int(cfg.training.cv_folds),
        "cv_oof_metrics": cv_oof_metrics,
        "cv_folds_report": cv_folds_report,
        "trainval_metrics": trainval_metrics,
        "test_metrics": test_metrics,
        "overfit_gap": overfit_gap,
        "final_training_epochs": int(final_training_epochs),
        "sample_counts": {
            "train_val": int(len(x_train_val_frame)),
            "test": int(len(x_test_frame)),
        },
        "dropped_columns": data_meta.get("dropped_columns", {}),
        "training_signature": training_signature,
        "cache_hit": False,
    }

    torch.save(artifact, run_dir / "model.pt")
    _write_json(run_dir / "metrics.json", report_payload)
    (run_dir / "report.md").write_text(_build_report(report_payload), encoding="utf-8")

    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, latest_dir / "model.pt")
    _write_json(latest_dir / "metrics.json", report_payload)
    (latest_dir / "report.md").write_text(
        _build_report(report_payload), encoding="utf-8"
    )

    cache_manifest: dict[str, str] = {
        "training_signature": training_signature,
        "model_version": model_version,
        "created_at_utc": created_at_utc,
        "run_dir": str(run_dir),
        "latest_model_path": str(latest_dir / "model.pt"),
        "metrics_path": str(latest_dir / "metrics.json"),
    }
    _write_json(run_dir / "cache_manifest.json", cache_manifest)
    _write_json(latest_dir / "cache_manifest.json", cache_manifest)

    cleanup_memory()

    return {
        "run_dir": str(run_dir),
        "latest_model_path": str(latest_dir / "model.pt"),
        "metrics_path": str(run_dir / "metrics.json"),
        "cache_hit": False,
        "training_signature": training_signature,
        "model_version": model_version,
    }


def main() -> None:
    """CLI entrypoint for model training."""
    parser = argparse.ArgumentParser(
        description="Train PyTorch breast cancer model on CPU."
    )
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force a new training run even when cache signature matches.",
    )
    args = parser.parse_args()

    result = run_training(config_path=args.config, force_retrain=args.force_retrain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
