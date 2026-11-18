"""Konfigurasi reproduksi berdasarkan blok executable paling akhir skrip penelitian."""
from dataclasses import dataclass

RESEARCH_SYMBOL = "BTC-USD"
RESEARCH_START = "2021-01-01"
RESEARCH_END = "2026-04-01"
RESEARCH_INTERVAL = "1d"

SCENARIOS = {
    "S1 (Baseline Univariat)": ["Close", "Lag_1", "Lag_2"],
    "S2 (Informasi OHLCV)": ["Open", "High", "Low", "Close", "Volume"],
    "S3 (Momentum - Teori)": ["Close", "Lag_1", "Lag_2", "RSI_14", "MACD"],
    "S4 (Tren SMA)": ["Close", "Lag_1", "Lag_2", "SMA_10", "SMA_30"],
    "S5 (Volatilitas BB)": ["Close", "Lag_1", "Lag_2", "BB_Up", "BB_Low"],
}

SENSITIVITY_GRID = [(5, 1e1), (10, 1e2), (50, 1e4), (100, 1e6), (150, 1e8)]
SENSITIVITY_RUNS = 10

ELM_BASELINE_NH = 10
ELM_BASELINE_C = 1e2
POC_ELM_RUNS = 30
POC_PSO_RUNS = 10
TOURNAMENT_RUNS = 10

WFV_FOLDS = 5
PSO_PARTICLES = 20
PSO_ITERATIONS = 50
PSO_C1 = 1.5
PSO_C2 = 1.5
PSO_WMAX = 0.9
PSO_WMIN = 0.4
PSO_NH_BOUNDS = (5, 150)
PSO_LOG10C_BOUNDS = (-1, 7)
PSO_VAL_RATIO = 0.25
ELM_FIXED_SEED = 42

POC_PSO_SEED = lambda run: run * 11 + 3
TOURNAMENT_PSO_SEED = lambda run: run * 13 + 42

# Penting: mode reproduksi mempertahankan normalisasi global seperti skrip sumber.
REPRODUCTION_SCALING = "global_minmax_before_wfv"

# Champion model eksplisit pada CELL 15 bagian akhir source.
RESEARCH_CHAMPION_SCENARIO = "S4 (Tren SMA)"
CHAMPION_PSO_SEED = 99

# Deployment H+1 adalah lapisan aplikasi di atas metodologi penelitian.
# Pemilihan skenario/seed tetap mengacu pada champion analysis source,
# sedangkan fit ulang pada seluruh data berlabel dilakukan khusus untuk inferensi.
DEPLOYMENT_PROTOCOL = "source_champion_s4_full_labeled_refit"
MODEL_ARTIFACT_VERSION = 1

# Live prediction is an operational extension. It NEVER changes the frozen research/parity contract.
LIVE_DATASET_NAME = "Live BTC-USD"
LIVE_DATA_START = RESEARCH_START
LIVE_MODEL_MODE = "live"
RESEARCH_MODEL_MODE = "research"
LIVE_DEPLOYMENT_PROTOCOL = "live_s4_full_labeled_refit"
