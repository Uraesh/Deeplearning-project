"""Unit tests for API utility endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from breast_cancer_ai import api as api_module
from breast_cancer_ai.api import app


class ApiMetrics(TypedDict):
    """Typed subset of metrics used by the test payload."""

    roc_auc: float
    sensitivity: float
    specificity: float


class PerformancePayload(TypedDict):
    """Typed metrics document returned by `/performance` in this test."""

    model_version: str
    test_metrics: ApiMetrics


def test_performance_endpoint_reads_metrics_payload(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """`/performance` should return the JSON content from METRICS_PATH."""
    metrics_path = tmp_path / "metrics.json"
    metrics_payload: PerformancePayload = {
        "model_version": "test-v1",
        "test_metrics": {"roc_auc": 0.99, "sensitivity": 0.95, "specificity": 0.96},
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    monkeypatch.setenv("METRICS_PATH", str(metrics_path))

    with TestClient(app) as client:
        response = client.get("/performance")

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics_path"] == str(metrics_path)
    assert payload["report"]["model_version"] == "test-v1"


def test_performance_dashboard_page_is_served() -> None:
    """`/performance-dashboard` should return the dedicated HTML page."""
    with TestClient(app) as client:
        response = client.get("/performance-dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Dashboard Clinique des Performances" in response.text


def test_resolve_metrics_path_prefers_model_neighbour(monkeypatch: MonkeyPatch) -> None:
    """Metrics path should default to sibling file of loaded MODEL_PATH."""
    monkeypatch.delenv("METRICS_PATH", raising=False)
    monkeypatch.setattr(
        api_module,
        "loaded_model_path",
        Path("models/runs/demo_run/model.pt"),
    )

    resolved_path: Path = api_module.resolve_metrics_path()

    assert resolved_path == Path("models/runs/demo_run/metrics.json")
    monkeypatch.setattr(api_module, "loaded_model_path", None)
