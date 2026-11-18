"""Champion evaluation dan deployment H+1.

Bagian `evaluate_source_champion` mereplikasi CELL 15 source (PSO-ELM S4, WFV 5-fold,
seed PSO 99). Bagian `train_deployment_model` adalah extension aplikasi: model champion
fit ulang pada seluruh data berlabel agar dapat melakukan inferensi untuk baris terbaru.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .config import (
    SCENARIOS, RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED,
    PSO_VAL_RATIO, ELM_FIXED_SEED, WFV_FOLDS, MODEL_ARTIFACT_VERSION,
    DEPLOYMENT_PROTOCOL, LIVE_DEPLOYMENT_PROTOCOL,
)
from .preprocessing import prepare_market_data, prepare_inference_features
from .scaling import minmax, inv_scale
from .pso_elm import PSO_ELM_Hyper
from .validation import calc_mae, calc_rmse, calc_mape


@dataclass
class DeploymentSummary:
    scenario: str
    n_hidden: int
    C: float
    trained_samples: int
    trained_until: str
    feature_count: int
    target_horizon: str = "H+1"


def _scale_with_bounds(a, mn, mx):
    return (a - mn) / (mx - mn + 1e-12)


def evaluate_source_champion(processed_df: pd.DataFrame, progress=None):
    """Replikasi analisis champion S4 pada CELL 15 source."""
    features = SCENARIOS[RESEARCH_CHAMPION_SCENARIO]
    X_sc, _, _ = minmax(processed_df[features].values)
    y_sc, ymn, ymx = minmax(processed_df["Target"].values.reshape(-1, 1))
    y_sc = y_sc.ravel()

    n = len(X_sc)
    seg = n // (WFV_FOLDS + 1)
    model = PSO_ELM_Hyper(val_ratio=PSO_VAL_RATIO, fixed_seed=ELM_FIXED_SEED)
    np.random.seed(CHAMPION_PSO_SEED)

    folds, plot_rows = [], []
    for k in range(1, WFV_FOLDS + 1):
        te = k * seg
        ts = te
        tt = ts + seg if k < WFV_FOLDS else n
        model.fit(X_sc[:te], y_sc[:te])
        yp_sc = model.predict(X_sc[ts:tt])
        yp = inv_scale(yp_sc, float(ymn), float(ymx))
        ya = inv_scale(y_sc[ts:tt], float(ymn), float(ymx))
        dates = processed_df.index[ts:tt]
        row = {
            "fold": k,
            "train_size": te,
            "test_size": tt-ts,
            "n_hidden": model.best_nh,
            "C": model.best_C,
            "mae": calc_mae(ya, yp),
            "rmse": calc_rmse(ya, yp),
            "mape": calc_mape(ya, yp),
        }
        folds.append(row)
        plot_rows.extend({
            "Date": pd.Timestamp(d), "Actual": float(a), "Predicted": float(p),
            "Residual": float(a-p), "Fold": k,
        } for d, a, p in zip(dates, ya, yp))
        if progress:
            progress(k / WFV_FOLDS, f"Champion S4 • Fold {k}/{WFV_FOLDS}")

    summary = {
        "mae": float(np.mean([r["mae"] for r in folds])),
        "rmse": float(np.mean([r["rmse"] for r in folds])),
        "mape": float(np.mean([r["mape"] for r in folds])),
    }
    return summary, pd.DataFrame(folds), pd.DataFrame(plot_rows)


def train_deployment_model(raw_df: pd.DataFrame, model_path: str | Path, N_pso=None, I_max=None, deployment_mode="research"):

    """Fit ulang champion S4 pada seluruh data berlabel untuk inferensi H+1.

    Ini adalah extension deployment aplikasi, bukan langkah evaluasi baru pada artikel.
    PSO tetap memakai konfigurasi champion source (val=25%, ELM seed=42, PSO seed=99).
    """
    labeled = prepare_market_data(raw_df)
    infer_df = prepare_inference_features(raw_df)
    features = SCENARIOS[RESEARCH_CHAMPION_SCENARIO]
    if labeled.empty or infer_df.empty:
        raise ValueError("Data tidak cukup untuk membangun model deployment.")

    X_raw = labeled[features].values
    y_raw = labeled["Target"].values.reshape(-1, 1)
    X_sc, xmn, xmx = minmax(X_raw)
    y_sc, ymn, ymx = minmax(y_raw)
    y_sc = y_sc.ravel()

    np.random.seed(CHAMPION_PSO_SEED)
    model = PSO_ELM_Hyper(val_ratio=PSO_VAL_RATIO, fixed_seed=ELM_FIXED_SEED)
    fit_kwargs = {}
    if N_pso is not None: fit_kwargs["N_pso"] = int(N_pso)
    if I_max is not None: fit_kwargs["I_max"] = int(I_max)
    model.fit(X_sc, y_sc, **fit_kwargs)

    protocol = LIVE_DEPLOYMENT_PROTOCOL if deployment_mode == "live" else DEPLOYMENT_PROTOCOL
    artifact = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "protocol": protocol,
        "deployment_mode": deployment_mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": RESEARCH_CHAMPION_SCENARIO,
        "features": features,
        "model": model,
        "x_min": xmn,
        "x_max": xmx,
        "y_min": ymn,
        "y_max": ymx,
        "trained_samples": len(labeled),
        "trained_until": pd.Timestamp(labeled.index.max()).strftime("%Y-%m-%d"),
        "source_latest_date": pd.Timestamp(infer_df.index.max()).strftime("%Y-%m-%d"),
        "pso_seed": CHAMPION_PSO_SEED,
        "fixed_elm_seed": ELM_FIXED_SEED,
        "val_ratio": PSO_VAL_RATIO,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    return DeploymentSummary(
        scenario=RESEARCH_CHAMPION_SCENARIO,
        n_hidden=model.best_nh,
        C=float(model.best_C),
        trained_samples=len(labeled),
        trained_until=artifact["trained_until"],
        feature_count=len(features),
    )


def load_artifact(model_path: str | Path):
    with Path(model_path).open("rb") as f:
        artifact = pickle.load(f)
    if artifact.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("Versi artifact model tidak didukung.")
    return artifact


def predict_h1(raw_df: pd.DataFrame, model_path: str | Path):
    artifact = load_artifact(model_path)
    infer_df = prepare_inference_features(raw_df)
    if infer_df.empty:
        raise ValueError("Tidak ada baris fitur yang siap untuk inferensi.")

    row = infer_df.iloc[-1]
    feature_values = row[artifact["features"]].values.astype(float).reshape(1, -1)
    x_sc = _scale_with_bounds(feature_values, artifact["x_min"], artifact["x_max"])
    yp_sc = artifact["model"].predict(x_sc)
    yp = inv_scale(yp_sc, float(np.asarray(artifact["y_min"]).ravel()[0]), float(np.asarray(artifact["y_max"]).ravel()[0]))

    input_date = pd.Timestamp(infer_df.index[-1])
    target_date = input_date + pd.Timedelta(days=1)
    last_close = float(row["Close"])
    predicted = float(np.asarray(yp).ravel()[0])
    return {
        "input_date": input_date.strftime("%Y-%m-%d"),
        "prediction_date": target_date.strftime("%Y-%m-%d"),
        "last_close": last_close,
        "predicted_close": predicted,
        "delta_usd": predicted - last_close,
        "delta_pct": ((predicted-last_close)/last_close*100) if last_close else 0.0,
        "scenario": artifact["scenario"],
        "n_hidden": artifact["model"].best_nh,
        "C": float(artifact["model"].best_C),
        "feature_snapshot": {f: float(row[f]) for f in artifact["features"]},
    }
