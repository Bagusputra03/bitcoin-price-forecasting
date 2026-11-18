import time
import numpy as np
import pandas as pd
from scipy import stats
from .config import *
from .scaling import minmax
from .elm import ELM
from .pso_elm import PSO_ELM_Hyper
from .validation import wfv5

def prepare_arrays(df, features):
    X, _, _ = minmax(df[features].values)
    y, ymn, ymx = minmax(df["Target"].values.reshape(-1, 1))
    return X, y.ravel(), ymn, ymx

def run_sensitivity(df):
    X, y, ymn, ymx = prepare_arrays(df, SCENARIOS["S3 (Momentum - Teori)"])
    rows, runs = [], []
    for nh, c in SENSITIVITY_GRID:
        maes, rmses, mapes = [], [], []
        for seed in range(SENSITIVITY_RUNS):
            res = wfv5(X, y, ELM(nh, c, seed), ymn, ymx, WFV_FOLDS)
            maes.append(res["mae"]); rmses.append(res["rmse"]); mapes.append(res["mape"])
            runs.append({"scenario":"S3 (Momentum - Teori)", "model_type":"ELM",
                         "run_number":seed+1, "seed":seed, "n_hidden":nh, "C":c, **res})
        rows.append({"n_hidden":nh, "C":c, "MAE Mean":np.mean(maes),
                     "RMSE Mean":np.mean(rmses), "MAPE Mean":np.mean(mapes),
                     "MAPE Std":np.std(mapes, ddof=1)})
    return pd.DataFrame(rows), runs

def run_poc(df, progress=None):
    X, y, ymn, ymx = prepare_arrays(df, SCENARIOS["S3 (Momentum - Teori)"])
    elm_runs, pso_runs, all_params = [], [], []
    total = POC_ELM_RUNS + POC_PSO_RUNS
    done = 0
    for seed in range(POC_ELM_RUNS):
        t0 = time.perf_counter()
        res = wfv5(X, y, ELM(ELM_BASELINE_NH, ELM_BASELINE_C, seed), ymn, ymx, WFV_FOLDS)
        elm_runs.append({"scenario":"S3 (Momentum - Teori)", "model_type":"ELM",
                         "run_number":seed+1, "seed":seed, "execution_time":time.perf_counter()-t0, **res})
        done += 1
        if progress: progress(done/total, f"ELM run {seed+1}/{POC_ELM_RUNS}")
    for run in range(POC_PSO_RUNS):
        seed = POC_PSO_SEED(run)
        np.random.seed(seed)
        t0 = time.perf_counter()
        res = wfv5(X, y, PSO_ELM_Hyper(PSO_VAL_RATIO, ELM_FIXED_SEED), ymn, ymx, WFV_FOLDS, True)
        pso_runs.append({"scenario":"S3 (Momentum - Teori)", "model_type":"PSO-ELM",
                         "run_number":run+1, "seed":seed, "execution_time":time.perf_counter()-t0, **res})
        all_params.extend(res["params"])
        done += 1
        if progress: progress(done/total, f"PSO-ELM run {run+1}/{POC_PSO_RUNS}")

    e = np.array([r["mape"] for r in elm_runs]); p = np.array([r["mape"] for r in pso_runs])
    e_mae = np.mean([r["mae"] for r in elm_runs]); p_mae = np.mean([r["mae"] for r in pso_runs])
    e_rmse = np.mean([r["rmse"] for r in elm_runs]); p_rmse = np.mean([r["rmse"] for r in pso_runs])
    _, pval = stats.mannwhitneyu(e, p, alternative="greater")
    pooled = np.sqrt((np.std(e, ddof=1)**2 + np.std(p, ddof=1)**2)/2)
    d = (np.mean(e)-np.mean(p))/pooled if pooled > 0 else 0.0
    summary = {
        "elm_mae":float(e_mae), "elm_rmse":float(e_rmse), "elm_mape":float(np.mean(e)), "elm_std":float(np.std(e,ddof=1)),
        "pso_mae":float(p_mae), "pso_rmse":float(p_rmse), "pso_mape":float(np.mean(p)), "pso_std":float(np.std(p,ddof=1)),
        "delta_mape":float((np.mean(e)-np.mean(p))/np.mean(e)*100),
        "p_value":float(pval), "cohens_d":float(d), "significant":bool(pval < 0.05)
    }
    return summary, elm_runs + pso_runs, pd.DataFrame(all_params)

def run_tournament(df, progress=None):
    rows, runs = [], []
    total = len(SCENARIOS) * TOURNAMENT_RUNS * 2
    done = 0
    for scenario, features in SCENARIOS.items():
        X, y, ymn, ymx = prepare_arrays(df, features)
        em, pm = [], []
        for seed in range(TOURNAMENT_RUNS):
            res = wfv5(X, y, ELM(ELM_BASELINE_NH, ELM_BASELINE_C, seed), ymn, ymx, WFV_FOLDS)
            em.append(res["mape"]); runs.append({"scenario":scenario,"model_type":"ELM","run_number":seed+1,"seed":seed,**res})
            done += 1
            if progress: progress(done/total, f"{scenario} • ELM {seed+1}/{TOURNAMENT_RUNS}")
        for r in range(TOURNAMENT_RUNS):
            seed = TOURNAMENT_PSO_SEED(r); np.random.seed(seed)
            res = wfv5(X, y, PSO_ELM_Hyper(PSO_VAL_RATIO, ELM_FIXED_SEED), ymn, ymx, WFV_FOLDS)
            pm.append(res["mape"]); runs.append({"scenario":scenario,"model_type":"PSO-ELM","run_number":r+1,"seed":seed,**res})
            done += 1
            if progress: progress(done/total, f"{scenario} • PSO-ELM {r+1}/{TOURNAMENT_RUNS}")
        _, pv = stats.mannwhitneyu(em, pm, alternative="greater")
        e, p = float(np.mean(em)), float(np.mean(pm))
        rows.append({"Skenario":scenario,"Dim":len(features),"ELM MAPE":e,"ELM Std":float(np.std(em,ddof=1)),
                     "PSO MAPE":p,"PSO Std":float(np.std(pm,ddof=1)),"ΔMAPE":(e-p)/e*100,"p-value":float(pv),
                     "Sig":"✅" if pv<0.05 else "⚠"})
    return pd.DataFrame(rows).sort_values("PSO MAPE").reset_index(drop=True), runs
