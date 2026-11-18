"""Parity validation antara Research Engine Streamlit dan output notebook final."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

from .config import (
    RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, RESEARCH_INTERVAL, SCENARIOS,
    SENSITIVITY_GRID, ELM_BASELINE_NH, ELM_BASELINE_C, POC_ELM_RUNS,
    POC_PSO_RUNS, TOURNAMENT_RUNS, WFV_FOLDS, PSO_PARTICLES, PSO_ITERATIONS,
    PSO_VAL_RATIO, ELM_FIXED_SEED, RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED,
)
from .reference_contract import (
    REFERENCE_ID, DATA_REFERENCE, WFV_REFERENCE, SENSITIVITY_REFERENCE,
    POC_REFERENCE, POC_PSO_RUN_MAPE_REFERENCE, POC_FOLD_HP_SUMMARY_REFERENCE,
    TOURNAMENT_REFERENCE, TOURNAMENT_RANKING_REFERENCE,
    TOURNAMENT_AVG_DELTA_REFERENCE, CHAMPION_REFERENCE,
)
from .preprocessing import prepare_market_data
from .experiments import run_sensitivity, run_poc, run_tournament
from .deployment import evaluate_source_champion


def _check(name: str, actual: Any, expected: Any, passed: bool, note: str = ""):
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "status": "PASS" if passed else "FAIL",
        "note": note,
    }


def _round_eq(actual, expected, decimals):
    try:
        return round(float(actual), decimals) == round(float(expected), decimals)
    except (TypeError, ValueError):
        return False


# MAE/RMSE dalam USD berasal dari operasi linear algebra floating-point.
# Hasil dapat berbeda sekitar 1 sen antar NumPy/BLAS/CPU walaupun pipeline,
# seed, data, ranking, MAPE, dan hyperparameter identik. Reference Colab tetap
# tidak diubah; hanya kriteria PASS yang memakai toleransi numerik yang eksplisit.
USD_ABS_TOL = 0.02


def _usd_eq(actual, expected, atol: float = USD_ABS_TOL):
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def _date_str(v):
    if v is None or (hasattr(pd, "isna") and pd.isna(v)):
        return None
    return pd.Timestamp(v).strftime("%Y-%m-%d")


def fold_plan(n: int, n_folds: int = WFV_FOLDS):
    seg = n // (n_folds + 1)
    rows = []
    for k in range(1, n_folds + 1):
        te = k * seg
        tt = te + seg if k < n_folds else n
        rows.append({"fold": k, "train_size": te, "test_size": tt - te})
    return rows


def structural_parity(raw_df: pd.DataFrame, dataset_meta: dict | None = None):
    processed = prepare_market_data(raw_df)
    checks = []
    meta = dataset_meta or {}

    checks.extend([
        _check("Symbol", meta.get("symbol", RESEARCH_SYMBOL), DATA_REFERENCE["symbol"], meta.get("symbol", RESEARCH_SYMBOL) == DATA_REFERENCE["symbol"]),
        _check("Start", meta.get("start_date", RESEARCH_START), DATA_REFERENCE["start"], meta.get("start_date", RESEARCH_START) == DATA_REFERENCE["start"]),
        _check("End (exclusive)", meta.get("end_date", RESEARCH_END), DATA_REFERENCE["end"], meta.get("end_date", RESEARCH_END) == DATA_REFERENCE["end"]),
        _check("Interval", meta.get("interval", RESEARCH_INTERVAL), DATA_REFERENCE["interval"], meta.get("interval", RESEARCH_INTERVAL) == DATA_REFERENCE["interval"]),
        _check("Jumlah sampel setelah preprocessing", len(processed), DATA_REFERENCE["processed_samples"], len(processed) == DATA_REFERENCE["processed_samples"]),
    ])

    if not raw_df.empty:
        checks.append(_check("Tanggal raw pertama", _date_str(raw_df.index.min()), DATA_REFERENCE["raw_first_date"], _date_str(raw_df.index.min()) == DATA_REFERENCE["raw_first_date"], "Derived dari periode source."))
        checks.append(_check("Tanggal raw terakhir", _date_str(raw_df.index.max()), DATA_REFERENCE["raw_last_date"], _date_str(raw_df.index.max()) == DATA_REFERENCE["raw_last_date"], "End yfinance bersifat exclusive."))
    if not processed.empty:
        checks.append(_check("Tanggal processed pertama", _date_str(processed.index.min()), DATA_REFERENCE["processed_first_date"], _date_str(processed.index.min()) == DATA_REFERENCE["processed_first_date"], "Warm-up SMA30 + target H+1."))
        checks.append(_check("Tanggal processed terakhir", _date_str(processed.index.max()), DATA_REFERENCE["processed_last_date"], _date_str(processed.index.max()) == DATA_REFERENCE["processed_last_date"], "Baris terakhir hilang karena Target=shift(-1)."))

    actual_folds = fold_plan(len(processed))
    for actual, expected in zip(actual_folds, WFV_REFERENCE):
        checks.append(_check(
            f"WFV Fold {expected['fold']} (train/test)",
            f"{actual['train_size']}/{actual['test_size']}",
            f"{expected['train_size']}/{expected['test_size']}",
            actual == expected,
        ))

    config_checks = [
        ("Skenario S1–S5", SCENARIOS, {
            "S1 (Baseline Univariat)": ["Close", "Lag_1", "Lag_2"],
            "S2 (Informasi OHLCV)": ["Open", "High", "Low", "Close", "Volume"],
            "S3 (Momentum - Teori)": ["Close", "Lag_1", "Lag_2", "RSI_14", "MACD"],
            "S4 (Tren SMA)": ["Close", "Lag_1", "Lag_2", "SMA_10", "SMA_30"],
            "S5 (Volatilitas BB)": ["Close", "Lag_1", "Lag_2", "BB_Up", "BB_Low"],
        }),
        ("Sensitivity grid", SENSITIVITY_GRID, [(5,1e1),(10,1e2),(50,1e4),(100,1e6),(150,1e8)]),
        ("ELM baseline", (ELM_BASELINE_NH, ELM_BASELINE_C), (10,1e2)),
        ("Run POC ELM/PSO", (POC_ELM_RUNS, POC_PSO_RUNS), (30,10)),
        ("Run tournament", TOURNAMENT_RUNS, 10),
        ("PSO N/I", (PSO_PARTICLES, PSO_ITERATIONS), (20,50)),
        ("PSO val_ratio", PSO_VAL_RATIO, 0.25),
        ("Fixed ELM seed", ELM_FIXED_SEED, 42),
        ("Champion scenario/seed", (RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED), ("S4 (Tren SMA)",99)),
    ]
    for name, actual, expected in config_checks:
        checks.append(_check(name, str(actual), str(expected), actual == expected))

    return {
        "reference_id": REFERENCE_ID,
        "processed_df": processed,
        "checks": checks,
        "passed": all(c["status"] == "PASS" for c in checks),
        "fold_plan": actual_folds,
    }


def compare_sensitivity(table: pd.DataFrame):
    checks = []
    actual_records = []
    for ref in SENSITIVITY_REFERENCE:
        row = table[(table["n_hidden"] == ref["n_hidden"]) & np.isclose(table["C"], ref["C"])]
        if row.empty:
            checks.append(_check(f"Sensitivity NH={ref['n_hidden']}, C={ref['C']:.0e}", "missing", "present", False))
            continue
        a = row.iloc[0].to_dict(); actual_records.append(a)
        for metric, decimals in [("MAE Mean",2),("RMSE Mean",2),("MAPE Mean",4),("MAPE Std",4)]:
            is_usd = metric in ("MAE Mean", "RMSE Mean")
            passed = _usd_eq(a[metric], ref[metric]) if is_usd else _round_eq(a[metric], ref[metric], decimals)
            note = (
                f"Toleransi numerik ±{USD_ABS_TOL:.2f} USD untuk floating-point/BLAS; reference Colab tetap {ref[metric]:.2f}."
                if is_usd else f"Parity pada {decimals} desimal seperti output notebook."
            )
            checks.append(_check(
                f"NH={ref['n_hidden']} C={ref['C']:.0e} • {metric}",
                round(float(a[metric]), decimals), ref[metric], passed, note
            ))
    return {"checks": checks, "passed": all(c["status"] == "PASS" for c in checks), "actual": actual_records}


def compare_poc(summary: dict, runs: list[dict], params: pd.DataFrame):
    checks = []
    for metric, decimals in [
        ("elm_mae",2),("elm_rmse",2),("elm_mape",4),("elm_std",4),
        ("pso_mae",2),("pso_rmse",2),("pso_mape",4),("pso_std",4),
        ("delta_mape",2),("cohens_d",4),
    ]:
        is_usd = metric in ("elm_mae", "elm_rmse", "pso_mae", "pso_rmse")
        passed = _usd_eq(summary[metric], POC_REFERENCE[metric]) if is_usd else _round_eq(summary[metric], POC_REFERENCE[metric], decimals)
        note = f"Toleransi numerik ±{USD_ABS_TOL:.2f} USD untuk floating-point/BLAS." if is_usd else ""
        checks.append(_check(metric, round(float(summary[metric]),decimals), POC_REFERENCE[metric], passed, note))
    checks.append(_check("p_value (printed 0.0000)", f"{summary['p_value']:.4f}", "0.0000", f"{summary['p_value']:.4f}" == "0.0000"))

    pso_runs = sorted([r for r in runs if r.get("model_type") == "PSO-ELM"], key=lambda r: r.get("run_number",0))
    for i, expected in enumerate(POC_PSO_RUN_MAPE_REFERENCE, start=1):
        if i <= len(pso_runs):
            actual = pso_runs[i-1]["mape"]
            checks.append(_check(f"PSO POC Run {i} MAPE", round(float(actual),4), expected, _round_eq(actual, expected, 4)))
        else:
            checks.append(_check(f"PSO POC Run {i} MAPE", "missing", expected, False))

    hp_actual = []
    if params is not None and not params.empty:
        g = (params.groupby("Fold").agg(
            n_train=("n_train","first"), nh_mean=("n_hidden","mean"), nh_std=("n_hidden","std"),
            lc_mean=("log10C","mean"), lc_std=("log10C","std")
        ).reset_index())
        for ref in POC_FOLD_HP_SUMMARY_REFERENCE:
            row = g[g["Fold"] == ref["Fold"]]
            if row.empty:
                checks.append(_check(f"POC HP Fold {ref['Fold']}", "missing", "present", False)); continue
            a = row.iloc[0].to_dict(); hp_actual.append(a)
            checks.append(_check(f"POC Fold {ref['Fold']} n_train", int(a["n_train"]), ref["n_train"], int(a["n_train"]) == ref["n_train"]))
            for metric, decimals in [("nh_mean",1),("nh_std",1),("lc_mean",2),("lc_std",2)]:
                checks.append(_check(f"POC Fold {ref['Fold']} {metric}", round(float(a[metric]),decimals), ref[metric], _round_eq(a[metric],ref[metric],decimals)))
    return {"checks": checks, "passed": all(c["status"] == "PASS" for c in checks), "summary": summary, "hp_summary": hp_actual}


def compare_tournament(table: pd.DataFrame):
    checks = []
    for scenario, ref in TOURNAMENT_REFERENCE.items():
        row = table[table["Skenario"] == scenario]
        if row.empty:
            checks.append(_check(f"Tournament {scenario}", "missing", "present", False)); continue
        a = row.iloc[0]
        checks.append(_check(f"{scenario} • Dim", int(a["Dim"]), ref["Dim"], int(a["Dim"]) == ref["Dim"]))
        for metric, decimals in [("ELM MAPE",4),("ELM Std",4),("PSO MAPE",4),("PSO Std",4),("ΔMAPE",2),("p-value",4)]:
            checks.append(_check(f"{scenario} • {metric}", round(float(a[metric]),decimals), ref[metric], _round_eq(a[metric],ref[metric],decimals)))
    ranking = list(table.sort_values("PSO MAPE")["Skenario"])
    checks.append(_check("Ranking tournament", " > ".join(ranking), " > ".join(TOURNAMENT_RANKING_REFERENCE), ranking == TOURNAMENT_RANKING_REFERENCE))
    avg_delta = float(table["ΔMAPE"].mean())
    checks.append(_check("Rata-rata Δ MAPE", round(avg_delta,2), TOURNAMENT_AVG_DELTA_REFERENCE, _round_eq(avg_delta,TOURNAMENT_AVG_DELTA_REFERENCE,2)))
    return {"checks": checks, "passed": all(c["status"] == "PASS" for c in checks), "table": table.to_dict("records")}


def compare_champion(summary: dict, folds: pd.DataFrame):
    checks = []
    for metric, decimals in [("mae",2),("rmse",2),("mape",4)]:
        is_usd = metric in ("mae", "rmse")
        passed = _usd_eq(summary[metric], CHAMPION_REFERENCE[metric]) if is_usd else _round_eq(summary[metric],CHAMPION_REFERENCE[metric],decimals)
        note = f"Toleransi numerik ±{USD_ABS_TOL:.2f} USD untuk floating-point/BLAS." if is_usd else ""
        checks.append(_check(f"Champion average {metric}", round(float(summary[metric]),decimals), CHAMPION_REFERENCE[metric], passed, note))
    fold_actual = []
    for ref in CHAMPION_REFERENCE["folds"]:
        row = folds[folds["fold"] == ref["fold"]]
        if row.empty:
            checks.append(_check(f"Champion Fold {ref['fold']}", "missing", "present", False)); continue
        a = row.iloc[0].to_dict(); fold_actual.append(a)
        checks.append(_check(f"Champion F{ref['fold']} NH", int(a["n_hidden"]), ref["n_hidden"], int(a["n_hidden"]) == ref["n_hidden"]))
        logc = math.log10(float(a["C"]))
        checks.append(_check(f"Champion F{ref['fold']} log10(C)", round(logc,2), ref["log10C"], _round_eq(logc,ref["log10C"],2)))
        for metric, decimals in [("mae",2),("rmse",2),("mape",4)]:
            is_usd = metric in ("mae", "rmse")
            passed = _usd_eq(a[metric], ref[metric]) if is_usd else _round_eq(a[metric],ref[metric],decimals)
            note = f"Toleransi numerik ±{USD_ABS_TOL:.2f} USD untuk floating-point/BLAS." if is_usd else ""
            checks.append(_check(f"Champion F{ref['fold']} {metric}", round(float(a[metric]),decimals), ref[metric], passed, note))
    return {"checks": checks, "passed": all(c["status"] == "PASS" for c in checks), "summary": summary, "folds": fold_actual}


def run_full_parity(raw_df: pd.DataFrame, dataset_meta: dict | None = None, progress=None):
    """Jalankan parity dari struktur hingga champion.

    Numeric stages hanya dijalankan jika structural contract lolos, supaya hasil dari
    dataset yang berbeda tidak disalahartikan sebagai reproduksi artikel.
    """
    report = {"reference_id": REFERENCE_ID, "sections": {}}
    if progress: progress(0.01, "Memvalidasi dataset & WFV...")
    structural = structural_parity(raw_df, dataset_meta)
    processed = structural.pop("processed_df")
    report["sections"]["structure"] = structural
    if not structural["passed"]:
        report["passed"] = False
        report["aborted"] = True
        report["reason"] = "Structural parity gagal; tahap numerik tidak dijalankan agar hasil tidak menyesatkan."
        if progress: progress(1.0, "Parity berhenti: dataset/struktur tidak identik.")
        return report

    if progress: progress(0.08, "Menjalankan sensitivity parity...")
    sens_table, _ = run_sensitivity(processed)
    report["sections"]["sensitivity"] = compare_sensitivity(sens_table)

    def poc_cb(v, text):
        if progress: progress(0.18 + v * 0.27, f"POC parity • {text}")
    if progress: progress(0.18, "Menjalankan POC parity...")
    poc_summary, poc_runs, poc_params = run_poc(processed, poc_cb)
    report["sections"]["poc"] = compare_poc(poc_summary, poc_runs, poc_params)

    def tour_cb(v, text):
        if progress: progress(0.45 + v * 0.45, f"Tournament parity • {text}")
    if progress: progress(0.45, "Menjalankan tournament parity...")
    tour_table, _ = run_tournament(processed, tour_cb)
    report["sections"]["tournament"] = compare_tournament(tour_table)

    def champ_cb(v, text):
        if progress: progress(0.90 + v * 0.09, text)
    if progress: progress(0.90, "Menjalankan champion S4 parity...")
    champ_summary, champ_folds, _ = evaluate_source_champion(processed, champ_cb)
    report["sections"]["champion"] = compare_champion(champ_summary, champ_folds)

    report["passed"] = all(s.get("passed", False) for s in report["sections"].values())
    report["aborted"] = False
    if progress: progress(1.0, "Parity validation selesai.")
    return report


def flatten_checks(report: dict):
    rows = []
    for section, payload in report.get("sections", {}).items():
        for c in payload.get("checks", []):
            rows.append({"section": section, **c})
    return rows
