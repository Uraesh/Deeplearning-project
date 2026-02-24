from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypeAlias, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from breast_cancer_ai.inference import TorchPredictor

LOGGER = logging.getLogger("breast_cancer_ai.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

DEFAULT_MODEL_PATH = "models/latest/model.pt"
DEFAULT_METRICS_PATH = "models/latest/metrics.json"
WEB_DIR = Path(__file__).resolve().parent / "web"
predictor: TorchPredictor | None = None
loaded_model_path: Path | None = None

JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float] = Field(..., description="One patient feature dictionary.")


class PredictBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[dict[str, float]] = Field(..., min_length=1, max_length=256)


class PredictResponse(BaseModel):
    request_id: str
    probability_malignant: float
    prediction: int
    prediction_label: str
    threshold: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, loaded_model_path
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    predictor = TorchPredictor.load(model_path)
    loaded_model_path = model_path
    LOGGER.info("Model loaded from %s (version=%s)", model_path, predictor.model_version)
    yield
    predictor = None
    loaded_model_path = None


app = FastAPI(
    title="Breast Cancer Detection API",
    version="0.1.0",
    description="PyTorch CPU-first inference service for breast cancer triage.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the web app entry page."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/performance-dashboard", include_in_schema=False)
def performance_dashboard() -> FileResponse:
    """Serve the dedicated clinician-facing performance dashboard page."""
    return FileResponse(WEB_DIR / "performance.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Return empty favicon response to avoid browser 404 noise."""
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "ok",
        "model_version": predictor.model_version,
        "created_at_utc": predictor.created_at_utc,
    }


@app.get("/model_info")
def model_info() -> dict[str, Any]:
    """Expose model metadata for frontend form generation."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "feature_count": len(predictor.feature_names),
        "feature_names": predictor.feature_names,
    }


def _load_metrics_payload() -> JsonObject:
    """Load persisted metrics for dashboard rendering."""
    metrics_path = resolve_metrics_path()
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics file not found: {metrics_path}")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Metrics file is invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Metrics payload must be a JSON object.")
    if not all(isinstance(key, str) for key in payload):
        raise HTTPException(status_code=500, detail="Metrics object keys must be strings.")
    return cast(JsonObject, payload)


def resolve_metrics_path() -> Path:
    """Resolve metrics path with explicit override then model-neighbour fallback."""
    env_metrics_path = os.getenv("METRICS_PATH")
    if env_metrics_path:
        return Path(env_metrics_path)
    if loaded_model_path is not None:
        return loaded_model_path.with_name("metrics.json")
    return Path(DEFAULT_METRICS_PATH)


@app.get("/performance")
def performance() -> dict[str, Any]:
    """Expose latest model performance payload for Plotly dashboard."""
    report = _load_metrics_payload()
    return {
        "metrics_path": str(resolve_metrics_path()),
        "report": report,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    request_id = uuid.uuid4().hex
    try:
        result = predictor.predict_from_records([payload.features])[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    LOGGER.info("predict request_id=%s model_version=%s", request_id, predictor.model_version)
    return {"request_id": request_id, **result}


@app.post("/predict_batch")
def predict_batch(payload: PredictBatchRequest) -> dict[str, Any]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    request_id = uuid.uuid4().hex
    try:
        results = predictor.predict_from_records(payload.samples)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    LOGGER.info(
        "predict_batch request_id=%s model_version=%s size=%d",
        request_id,
        predictor.model_version,
        len(results),
    )
    return {
        "request_id": request_id,
        "model_version": predictor.model_version,
        "size": len(results),
        "results": results,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastAPI service for model inference.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    uvicorn.run("breast_cancer_ai.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
