from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        brier = float("nan")

    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
    }


def select_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_sensitivity: float,
) -> tuple[float, dict[str, object]]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    candidates = sorted(set(np.concatenate(([0.0], np.round(y_prob, 6), [1.0]))))
    evaluated = [(thr, compute_binary_metrics(y_true, y_prob, thr)) for thr in candidates]

    feasible = [(thr, met) for thr, met in evaluated if met["sensitivity"] >= target_sensitivity]
    if feasible:
        chosen = max(
            feasible,
            key=lambda item: (
                item[1]["specificity"],
                item[1]["ppv"],
                item[0],
            ),
        )
        reason = "target_sensitivity_met"
    else:
        chosen = max(
            evaluated,
            key=lambda item: item[1]["sensitivity"] + item[1]["specificity"] - 1.0,
        )
        reason = "fallback_max_youden"

    threshold, chosen_metrics = chosen
    return float(threshold), {
        "reason": reason,
        "target_sensitivity": float(target_sensitivity),
        "candidate_count": len(candidates),
        "chosen_metrics": chosen_metrics,
    }
