from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.database import init_db, connect
from research_engine.config import (
    ELM_BASELINE_NH, ELM_BASELINE_C, PSO_PARTICLES, PSO_ITERATIONS,
    PSO_VAL_RATIO, RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED,
    LIVE_MODEL_MODE, RESEARCH_MODEL_MODE,
)
from research_engine.parity import fold_plan
from research_engine.reference_contract import DATA_REFERENCE, WFV_REFERENCE, REFERENCE_ID

init_db()
assert ELM_BASELINE_NH == 10 and ELM_BASELINE_C == 1e2
assert PSO_PARTICLES == 20 and PSO_ITERATIONS == 50 and PSO_VAL_RATIO == 0.25
assert RESEARCH_CHAMPION_SCENARIO == 'S4 (Tren SMA)' and CHAMPION_PSO_SEED == 99
assert fold_plan(DATA_REFERENCE['processed_samples']) == WFV_REFERENCE
assert LIVE_MODEL_MODE == 'live' and RESEARCH_MODEL_MODE == 'research'
with connect() as con:
    assert con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parity_reports'").fetchone()
    ds_cols = {r[1] for r in con.execute('PRAGMA table_info(datasets)').fetchall()}
    model_cols = {r[1] for r in con.execute('PRAGMA table_info(models)').fetchall()}
    pred_cols = {r[1] for r in con.execute('PRAGMA table_info(predictions)').fetchall()}
    assert {'mode','last_synced_at'} <= ds_cols
    assert {'model_mode','source_latest_date'} <= model_cols
    assert {'prediction_mode','data_freshness_date'} <= pred_cols
print(f"PRE-FLIGHT PASS • {REFERENCE_ID} • Research frozen • Live mode ready • champion S4")
