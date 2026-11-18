import numpy as np
from .scaling import inv_scale

def calc_mae(yt, yp):
    return float(np.mean(np.abs(yt - yp)))

def calc_rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def calc_mape(yt, yp):
    m = yt != 0
    return float(np.mean(np.abs((yt[m] - yp[m]) / yt[m])) * 100)

def wfv5(X, y, model, ymn, ymx, n_folds=5, collect_params=False):
    N, seg = len(X), len(X) // (n_folds + 1)
    fm, fr, fp, params, folds = [], [], [], [], []
    for k in range(1, n_folds + 1):
        te = k * seg
        ts = te
        tt = ts + seg if k < n_folds else N
        model.fit(X[:te], y[:te])
        yp = inv_scale(model.predict(X[ts:tt]), float(ymn), float(ymx))
        ya = inv_scale(y[ts:tt], float(ymn), float(ymx))
        f_mae, f_rmse, f_mape = calc_mae(ya, yp), calc_rmse(ya, yp), calc_mape(ya, yp)
        fm.append(f_mae); fr.append(f_rmse); fp.append(f_mape)
        row = {"fold": k, "train_size": te, "test_size": tt-ts,
               "mae": f_mae, "rmse": f_rmse, "mape": f_mape}
        if collect_params and hasattr(model, "best_nh"):
            param = {"Fold": k, "n_train": te, "n_hidden": model.best_nh,
                     "C": model.best_C, "log10C": round(np.log10(model.best_C), 2)}
            params.append(param)
            row.update({"n_hidden": model.best_nh, "C": model.best_C})
        folds.append(row)
    return {"mae": float(np.mean(fm)), "rmse": float(np.mean(fr)),
            "mape": float(np.mean(fp)), "params": params, "folds": folds}
