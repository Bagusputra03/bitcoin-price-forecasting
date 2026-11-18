"""Database-backed read model untuk hasil penelitian pada frontend sederhana.

Modul ini sengaja TIDAK mengimpor ``research_engine.reference_contract``.
Angka yang tampil ke pengguna hanya boleh berasal dari hasil komputasi yang
sudah tersimpan di SQLite: experiment/runs/folds atau parity report aktual.
Reference Colab tetap berada di Research Engine untuk diagnostic/parity saja.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from services.repository import (
    list_experiments,
    experiment_runs,
    experiment_folds,
    latest_parity_report,
    get_parity_report,
)


def _latest_completed(kind: str):
    for row in list_experiments():
        if row.get("experiment_type") == kind and row.get("status") == "completed":
            return row
    return None


def _latest_parity_payload():
    row = latest_parity_report()
    if not row:
        return None, None
    loaded = get_parity_report(row["id"])
    if not loaded:
        return None, None
    return loaded.get("report") or {}, loaded


def _source(kind: str, row: dict | None):
    if not row:
        return None
    return {
        "kind": kind,
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "completed_at": row.get("completed_at"),
    }


def get_sensitivity_results():
    """Return sensitivity table dari SQLite, tanpa fallback hard-coded.

    Prioritas:
    1) experiment sensitivity yang menyimpan n_hidden/c_value pada run;
    2) actual values dari latest parity report yang sudah tersimpan di SQLite.
    """
    exp = _latest_completed("sensitivity")
    if exp:
        runs = pd.DataFrame(experiment_runs(exp["id"]))
        if not runs.empty and {"n_hidden", "c_value"}.issubset(runs.columns):
            valid = runs.dropna(subset=["n_hidden", "c_value", "mae", "rmse", "mape"]).copy()
            if not valid.empty:
                table = (
                    valid.groupby(["n_hidden", "c_value"], as_index=False)
                    .agg(
                        **{
                            "MAE Mean": ("mae", "mean"),
                            "RMSE Mean": ("rmse", "mean"),
                            "MAPE Mean": ("mape", "mean"),
                            "MAPE Std": ("mape", lambda s: float(np.std(s.astype(float), ddof=1)) if len(s) > 1 else 0.0),
                        }
                    )
                    .rename(columns={"c_value": "C"})
                    .sort_values("n_hidden")
                    .reset_index(drop=True)
                )
                return table, _source("experiment", exp)

    report, parity_row = _latest_parity_payload()
    actual = (((report or {}).get("sections") or {}).get("sensitivity") or {}).get("actual") or []
    if actual:
        return pd.DataFrame(actual), _source("parity_report", parity_row)
    return pd.DataFrame(), None


def get_poc_results():
    exp = _latest_completed("poc")
    if exp:
        rows = experiment_runs(exp["id"])
        e_rows = [r for r in rows if r.get("model_type") == "ELM" and r.get("mape") is not None]
        p_rows = [r for r in rows if r.get("model_type") == "PSO-ELM" and r.get("mape") is not None]
        if e_rows and p_rows:
            e = np.array([float(r["mape"]) for r in e_rows], dtype=float)
            p = np.array([float(r["mape"]) for r in p_rows], dtype=float)
            _, pval = stats.mannwhitneyu(e, p, alternative="greater")
            pooled = np.sqrt((np.std(e, ddof=1) ** 2 + np.std(p, ddof=1) ** 2) / 2)
            out = {
                "elm_mae": float(np.mean([r["mae"] for r in e_rows])),
                "elm_rmse": float(np.mean([r["rmse"] for r in e_rows])),
                "elm_mape": float(np.mean(e)),
                "elm_std": float(np.std(e, ddof=1)),
                "pso_mae": float(np.mean([r["mae"] for r in p_rows])),
                "pso_rmse": float(np.mean([r["rmse"] for r in p_rows])),
                "pso_mape": float(np.mean(p)),
                "pso_std": float(np.std(p, ddof=1)),
                "delta_mape": float((np.mean(e) - np.mean(p)) / np.mean(e) * 100),
                "p_value": float(pval),
                "cohens_d": float((np.mean(e) - np.mean(p)) / pooled if pooled > 0 else 0.0),
            }
            return out, _source("experiment", exp)

    report, parity_row = _latest_parity_payload()
    summary = (((report or {}).get("sections") or {}).get("poc") or {}).get("summary")
    if summary:
        return dict(summary), _source("parity_report", parity_row)
    return None, None


def get_tournament_results():
    exp = _latest_completed("tournament")
    if exp:
        runs = pd.DataFrame(experiment_runs(exp["id"]))
        if not runs.empty and {"scenario", "model_type", "mape"}.issubset(runs.columns):
            rows = []
            for scenario in runs["scenario"].dropna().unique():
                sub = runs[runs["scenario"] == scenario]
                e = sub[sub["model_type"] == "ELM"]["mape"].dropna().astype(float).values
                p = sub[sub["model_type"] == "PSO-ELM"]["mape"].dropna().astype(float).values
                if len(e) == 0 or len(p) == 0:
                    continue
                _, pval = stats.mannwhitneyu(e, p, alternative="greater")
                em, pm = float(np.mean(e)), float(np.mean(p))
                rows.append({
                    "Skenario": scenario,
                    "ELM MAPE": em,
                    "ELM Std": float(np.std(e, ddof=1)),
                    "PSO MAPE": pm,
                    "PSO Std": float(np.std(p, ddof=1)),
                    "ΔMAPE": float((em - pm) / em * 100),
                    "p-value": float(pval),
                })
            if rows:
                return pd.DataFrame(rows).sort_values("PSO MAPE").reset_index(drop=True), _source("experiment", exp)

    report, parity_row = _latest_parity_payload()
    table = (((report or {}).get("sections") or {}).get("tournament") or {}).get("table") or []
    if table:
        return pd.DataFrame(table).sort_values("PSO MAPE").reset_index(drop=True), _source("parity_report", parity_row)
    return pd.DataFrame(), None


def _normalize_champion_folds(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "fold" in out.columns and "fold_number" not in out.columns:
        out["fold_number"] = out["fold"]
    if "C" in out.columns and "c_value" not in out.columns:
        out["c_value"] = out["C"]
    return out


def get_champion_results():
    exp = _latest_completed("champion_evaluation")
    if exp:
        runs = experiment_runs(exp["id"])
        folds = _normalize_champion_folds(pd.DataFrame(experiment_folds(exp["id"])))
        if runs:
            r = runs[0]
            out = {
                "scenario": r.get("scenario") or "S4 (Tren SMA)",
                "mae": r.get("mae"),
                "rmse": r.get("rmse"),
                "mape": r.get("mape"),
            }
            if all(out.get(k) is not None for k in ("mae", "rmse", "mape")):
                return out, folds, _source("experiment", exp)

    report, parity_row = _latest_parity_payload()
    section = (((report or {}).get("sections") or {}).get("champion") or {})
    summary = section.get("summary")
    folds = _normalize_champion_folds(pd.DataFrame(section.get("folds") or []))
    if summary:
        out = dict(summary)
        out.setdefault("scenario", "S4 (Tren SMA)")
        return out, folds, _source("parity_report", parity_row)
    return None, pd.DataFrame(), None


def research_results_status():
    sens, sens_src = get_sensitivity_results()
    poc, poc_src = get_poc_results()
    tour, tour_src = get_tournament_results()
    champ, _, champ_src = get_champion_results()
    return {
        "sensitivity": bool(not sens.empty),
        "poc": bool(poc),
        "tournament": bool(not tour.empty),
        "champion": bool(champ),
        "sources": {
            "sensitivity": sens_src,
            "poc": poc_src,
            "tournament": tour_src,
            "champion": champ_src,
        },
    }
