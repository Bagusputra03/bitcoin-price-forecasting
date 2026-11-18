"""Operational live-data helpers.

This module is deliberately separate from the frozen research engine/parity path.
It reuses the same S4 feature engineering and PSO-ELM deployment routine, but data
are refreshed from Yahoo Finance up to the latest available daily BTC-USD candle.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

from .config import RESEARCH_SYMBOL, LIVE_DATA_START, RESEARCH_INTERVAL
from .data_loader import download_yfinance
from .deployment import train_deployment_model, predict_h1, load_artifact


def live_end_exclusive(now: datetime | None = None) -> str:
    """Yahoo Finance `end` is exclusive, therefore ask through tomorrow (UTC)."""
    now = now or datetime.now(timezone.utc)
    return (now.date() + timedelta(days=1)).isoformat()


def download_live_data(start: str = LIVE_DATA_START) -> pd.DataFrame:
    return download_yfinance(RESEARCH_SYMBOL, start, live_end_exclusive(), RESEARCH_INTERVAL)


def build_live_model(raw_df: pd.DataFrame, model_path: str | Path, N_pso=None, I_max=None):
    return train_deployment_model(
        raw_df, model_path, N_pso=N_pso, I_max=I_max, deployment_mode="live"
    )


def predict_live_h1(raw_df: pd.DataFrame, model_path: str | Path):
    result = predict_h1(raw_df, model_path)
    result["prediction_mode"] = "live"
    return result


def artifact_source_latest_date(model_path: str | Path):
    artifact = load_artifact(model_path)
    return artifact.get("source_latest_date")


def live_freshness_state(data_latest_date, model_source_date=None, prediction_data_date=None,
                         prediction_model_id=None, active_model_id=None):
    """Return a deterministic operational freshness state for the Live Mode UI.

    The research path never calls this helper.  It only guards operational actions:
    a prediction is allowed when an active live model was trained from the exact
    latest dataset candle.  A prediction is considered current only when it used
    that same data date *and* the currently active model.
    """
    data_date = str(data_latest_date) if data_latest_date else None
    model_date = str(model_source_date) if model_source_date else None
    pred_date = str(prediction_data_date) if prediction_data_date else None

    if not data_date:
        code = "NO_DATA"
        label = "Data live belum tersedia"
        can_predict = False
    elif not active_model_id:
        code = "NO_MODEL"
        label = "Model live belum tersedia"
        can_predict = False
    elif not model_date:
        code = "MODEL_DATE_UNKNOWN"
        label = "Tanggal sumber model tidak tersedia"
        can_predict = False
    elif model_date != data_date:
        code = "MODEL_STALE"
        label = f"Model usang • model {model_date} vs data {data_date}"
        can_predict = False
    else:
        code = "MODEL_FRESH"
        label = f"Model fresh • {data_date}"
        can_predict = True

    prediction_current = bool(
        can_predict and pred_date == data_date and prediction_model_id and active_model_id
        and str(prediction_model_id) == str(active_model_id)
    )
    if prediction_current:
        code = "PREDICTION_CURRENT"
        label = f"Prediksi terbaru tersedia • {data_date}"

    return {
        "code": code,
        "label": label,
        "data_latest_date": data_date,
        "model_source_date": model_date,
        "model_fresh": bool(can_predict),
        "can_predict": bool(can_predict),
        "prediction_current": prediction_current,
    }
