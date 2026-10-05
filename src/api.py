"""FastAPI read-only interface for the Smart Healthcare System artifacts."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from src.gemini_service import explain_recommendation
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
MASTER_FILE = ROOT / "data" / "processed" / "master_dataset.csv"
REDISTRIBUTION_FILE = ROOT / "data" / "processed" / "final_redistribution_recommendations.csv"
DEMAND_MODEL_FILE = ROOT / "src" / "demand_model.pkl"
STOCKOUT_MODEL_FILE = ROOT / "src" / "stockout_model.pkl"

app = FastAPI(
    title="Smart Healthcare System API",
    description="Read-only API for existing demand, stockout, and redistribution artifacts.",
    version="1.0.0",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Required artifact is missing: {path.name}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not read {path.name}: {exc}") from exc


@lru_cache(maxsize=1)
def _master() -> pd.DataFrame:
    return _read_csv(MASTER_FILE)


@lru_cache(maxsize=1)
def _model_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Required model is missing: {path.name}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load {path.name}: {exc}") from exc
    if not isinstance(bundle, dict) or not {"model", "preprocessor", "features"}.issubset(bundle):
        raise HTTPException(status_code=503, detail=f"Unexpected saved model format: {path.name}")
    return bundle


def _latest_rows() -> pd.DataFrame:
    df = _master()
    if "month" not in df:
        raise HTTPException(status_code=503, detail="master_dataset.csv has no month column")
    latest = df["month"].max()
    return df.loc[df["month"] == latest].copy()


@lru_cache(maxsize=1)
def _demand_forecasts() -> pd.DataFrame:
    df = _latest_rows()
    bundle = _model_bundle(DEMAND_MODEL_FILE)
    features = bundle["features"]
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise HTTPException(status_code=503, detail=f"Demand model input columns missing: {missing}")
    try:
        df["predicted_demand"] = np.maximum(
            0, bundle["model"].predict(bundle["preprocessor"].transform(df[features]))
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Demand prediction failed: {exc}") from exc
    return df


@lru_cache(maxsize=1)
def _stockout_risks() -> pd.DataFrame:
    df = _latest_rows()
    bundle = _model_bundle(STOCKOUT_MODEL_FILE)
    features = bundle["features"]
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise HTTPException(status_code=503, detail=f"Stockout model input columns missing: {missing}")
    try:
        transformed = bundle["preprocessor"].transform(df[features])
        model = bundle["model"]
        df["stockout_prediction"] = model.predict(transformed)
        probabilities = model.predict_proba(transformed)
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
        df["stockout_probability"] = probabilities[:, positive_index]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Stockout scoring failed: {exc}") from exc
    return df


def _page(df: pd.DataFrame, limit: int, offset: int) -> dict[str, Any]:
    items = df.iloc[offset:offset + limit].to_dict(orient="records")
    return {"count": int(len(df)), "limit": limit, "offset": offset,
            "items": [_json_safe(item) for item in items]}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/facilities")
def facilities(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    df = _master()
    columns = [c for c in ["facility_id", "facility_name", "district", "latitude", "longitude",
                            "coordinate_usable"] if c in df.columns]
    result = df[columns].drop_duplicates(subset=[c for c in ["facility_id"] if c in columns])
    if "facility_name" in result:
        result = result.sort_values("facility_name")
    return _page(result, limit, offset)


@app.get("/medicines")
def medicines(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    df = _master()
    columns = [c for c in ["medicine_id", "medicine_name", "medicine_unit"] if c in df.columns]
    result = df[columns].drop_duplicates(subset=[c for c in ["medicine_id"] if c in columns])
    if "medicine_name" in result:
        result = result.sort_values("medicine_name")
    return _page(result, limit, offset)


def _filtered_latest(df: pd.DataFrame, facility_id: str | None, medicine_id: str | None) -> pd.DataFrame:
    if facility_id is not None:
        df = df.loc[df["facility_id"].astype(str) == facility_id]
    if medicine_id is not None:
        df = df.loc[df["medicine_id"].astype(str) == medicine_id]
    return df


@app.get("/demand-forecast")
def demand_forecast(facility_id: str | None = None, medicine_id: str | None = None,
                    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    df = _filtered_latest(_demand_forecasts(), facility_id, medicine_id)
    columns = [c for c in ["month", "facility_id", "facility_name", "district", "medicine_id",
                            "medicine_name", "demand_quantity", "predicted_demand", "closing_stock",
                            "reorder_level"] if c in df.columns]
    return _page(df[columns], limit, offset)


@app.get("/stockout-risk")
def stockout_risk(facility_id: str | None = None, medicine_id: str | None = None,
                  limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    df = _filtered_latest(_stockout_risks(), facility_id, medicine_id)
    columns = [c for c in ["month", "facility_id", "facility_name", "district", "medicine_id",
                            "medicine_name", "stockout_prediction", "stockout_probability",
                            "stockout_flag", "closing_stock", "reorder_level"] if c in df.columns]
    return _page(df[columns], limit, offset)


@app.get("/redistribution")
def redistribution(medicine_id: str | None = None, to_facility_id: str | None = None,
                   limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    df = _read_csv(REDISTRIBUTION_FILE)
    if medicine_id is not None and "medicine_id" in df:
        df = df.loc[df["medicine_id"].astype(str) == medicine_id]
    if to_facility_id is not None and "to_facility_id" in df:
        df = df.loc[df["to_facility_id"].astype(str) == to_facility_id]
    if "priority_score" in df:
        df = df.sort_values("priority_score", ascending=False)
    return _page(df, limit, offset)

class ExplainRequest(BaseModel):
    medicine_name: str
    to_facility: str
    from_facility: str
    current_stock: float
    reorder_level: float
    shortage: float
    predicted_demand: float
    transfer_quantity: float
    distance_km: float
    stockout_flag: int


@app.post("/explain")
def explain(request: ExplainRequest):
    """Generate a Gemini explanation for a redistribution recommendation."""

    try:
        data = request.model_dump()

        explanation = explain_recommendation(data)

        return {
            "status": "success",
            "explanation": explanation,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini explanation failed: {exc}",
        ) from exc
