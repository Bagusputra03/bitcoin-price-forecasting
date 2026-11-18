"""Reference contract dari output notebook final yang diberikan pengguna.

Sumber numerik:
- blok final "JALUR KEMENANGAN (BASELINE DEFAULT 10/100)" pada notebook,
- CELL 15 final "CHAMPION MODEL: S4".

Nilai referensi disimpan pada presisi yang memang dicetak oleh notebook. Parity
numerik membandingkan hasil sistem pada presisi publikasi tersebut, bukan memaksa
bitwise equality lintas BLAS/OS.
"""

REFERENCE_ID = "COLAB_FINAL_CELL13_CELL28"
REFERENCE_LABEL = "Notebook final — baseline 10/100 + champion S4"

DATA_REFERENCE = {
    "symbol": "BTC-USD",
    "start": "2021-01-01",
    "end": "2026-04-01",
    "interval": "1d",
    "processed_samples": 1886,
    "raw_first_date": "2021-01-01",
    "raw_last_date": "2026-03-31",
    "processed_first_date": "2021-01-30",
    "processed_last_date": "2026-03-30",
}

WFV_REFERENCE = [
    {"fold": 1, "train_size": 314,  "test_size": 314},
    {"fold": 2, "train_size": 628,  "test_size": 314},
    {"fold": 3, "train_size": 942,  "test_size": 314},
    {"fold": 4, "train_size": 1256, "test_size": 314},
    {"fold": 5, "train_size": 1570, "test_size": 316},
]

SENSITIVITY_REFERENCE = [
    {"n_hidden": 5,   "C": 1e1, "MAE Mean": 5020.52, "RMSE Mean": 5956.22, "MAPE Mean": 11.6028, "MAPE Std": 4.0065},
    {"n_hidden": 10,  "C": 1e2, "MAE Mean": 1716.18, "RMSE Mean": 2224.95, "MAPE Mean": 3.1191,  "MAPE Std": 0.2242},
    {"n_hidden": 50,  "C": 1e4, "MAE Mean": 1279.79, "RMSE Mean": 1731.09, "MAPE Mean": 2.3564,  "MAPE Std": 0.0688},
    {"n_hidden": 100, "C": 1e6, "MAE Mean": 1561.89, "RMSE Mean": 2069.66, "MAPE Mean": 3.2198,  "MAPE Std": 0.2322},
    {"n_hidden": 150, "C": 1e8, "MAE Mean": 2695.83, "RMSE Mean": 3688.33, "MAPE Mean": 5.9091,  "MAPE Std": 0.2742},
]

POC_REFERENCE = {
    "elm_mae": 1674.38,
    "elm_rmse": 2175.30,
    "elm_mape": 3.0999,
    "elm_std": 0.3639,
    "pso_mae": 1284.96,
    "pso_rmse": 1739.31,
    "pso_mape": 2.2730,
    "pso_std": 0.0219,
    "delta_mape": 26.67,
    "cohens_d": 3.2074,
    "p_value_display": 0.0000,
}

POC_PSO_RUN_MAPE_REFERENCE = [
    2.2822, 2.2822, 2.2625, 2.2527, 2.2794,
    2.2873, 2.2795, 2.3142, 2.2378, 2.2526,
]

POC_FOLD_HP_SUMMARY_REFERENCE = [
    {"Fold": 1, "n_train": 314,  "nh_mean": 33.8,  "nh_std": 3.8,  "lc_mean": 4.79, "lc_std": 0.04},
    {"Fold": 2, "n_train": 628,  "nh_mean": 62.9,  "nh_std": 45.4, "lc_mean": 2.46, "lc_std": 0.60},
    {"Fold": 3, "n_train": 942,  "nh_mean": 128.5, "nh_std": 16.9, "lc_mean": 4.09, "lc_std": 0.00},
    {"Fold": 4, "n_train": 1256, "nh_mean": 53.0,  "nh_std": 9.7,  "lc_mean": 2.13, "lc_std": 0.15},
    {"Fold": 5, "n_train": 1570, "nh_mean": 6.0,   "nh_std": 0.0,  "lc_mean": 7.00, "lc_std": 0.00},
]

TOURNAMENT_REFERENCE = {
    "S1 (Baseline Univariat)": {"Dim": 3, "ELM MAPE": 2.5413, "ELM Std": 0.2907, "PSO MAPE": 2.1915, "PSO Std": 0.0632, "ΔMAPE": 13.77, "p-value": 0.0057},
    "S2 (Informasi OHLCV)":    {"Dim": 5, "ELM MAPE": 2.7167, "ELM Std": 0.2977, "PSO MAPE": 2.7088, "PSO Std": 0.1960, "ΔMAPE": 0.29,  "p-value": 0.7146},
    "S3 (Momentum - Teori)":   {"Dim": 5, "ELM MAPE": 3.1191, "ELM Std": 0.2242, "PSO MAPE": 2.2686, "PSO Std": 0.0537, "ΔMAPE": 27.27, "p-value": 0.0001},
    "S4 (Tren SMA)":           {"Dim": 5, "ELM MAPE": 2.9384, "ELM Std": 0.2896, "PSO MAPE": 2.1553, "PSO Std": 0.0443, "ΔMAPE": 26.65, "p-value": 0.0001},
    "S5 (Volatilitas BB)":     {"Dim": 5, "ELM MAPE": 2.9446, "ELM Std": 0.2527, "PSO MAPE": 2.1980, "PSO Std": 0.0777, "ΔMAPE": 25.35, "p-value": 0.0001},
}
TOURNAMENT_RANKING_REFERENCE = [
    "S4 (Tren SMA)",
    "S1 (Baseline Univariat)",
    "S5 (Volatilitas BB)",
    "S3 (Momentum - Teori)",
    "S2 (Informasi OHLCV)",
]
TOURNAMENT_AVG_DELTA_REFERENCE = 18.67

CHAMPION_REFERENCE = {
    "scenario": "S4 (Tren SMA)",
    "seed": 99,
    "val_ratio": 0.25,
    "mae": 1189.06,
    "rmse": 1626.43,
    "mape": 2.1323,
    "folds": [
        {"fold": 1, "n_hidden": 148, "log10C": 1.89, "mae": 911.03,  "rmse": 1236.16, "mape": 3.0725},
        {"fold": 2, "n_hidden": 33,  "log10C": 2.25, "mae": 383.51,  "rmse": 592.84,  "mape": 1.6072},
        {"fold": 3, "n_hidden": 65,  "log10C": 1.58, "mae": 1006.70, "rmse": 1496.07, "mape": 1.8939},
        {"fold": 4, "n_hidden": 7,   "log10C": 3.87, "mae": 1845.59, "rmse": 2463.01, "mape": 2.2158},
        {"fold": 5, "n_hidden": 5,   "log10C": 4.79, "mae": 1798.48, "rmse": 2344.10, "mape": 1.8720},
    ],
}
