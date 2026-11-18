from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.database import init_db, connect
from research_engine.config import (
    ELM_BASELINE_NH, ELM_BASELINE_C, PSO_PARTICLES, PSO_ITERATIONS,
    PSO_VAL_RATIO, RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED,
)
from research_engine.live import live_freshness_state
from services.repository import active_model, latest_prediction, latest_dataset_by_mode, dataset_market_stats

checks = []
def check(name, ok, detail=''):
    checks.append((name, bool(ok), detail))

init_db()
check('Research baseline', ELM_BASELINE_NH == 10 and ELM_BASELINE_C == 1e2, f'NH={ELM_BASELINE_NH}, C={ELM_BASELINE_C}')
check('PSO contract', PSO_PARTICLES == 20 and PSO_ITERATIONS == 50 and PSO_VAL_RATIO == .25, f'{PSO_PARTICLES}x{PSO_ITERATIONS}, val={PSO_VAL_RATIO}')
check('Champion S4', RESEARCH_CHAMPION_SCENARIO == 'S4 (Tren SMA)' and CHAMPION_PSO_SEED == 99, RESEARCH_CHAMPION_SCENARIO)

with connect() as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
check('SQLite schema', {'datasets','experiments','models','predictions','parity_reports','activity_logs'} <= tables, ', '.join(sorted(tables)))

model_dir = ROOT / 'storage' / 'models'
model_dir.mkdir(parents=True, exist_ok=True)
probe = model_dir / '.write_test'
try:
    probe.write_text('ok'); probe.unlink(); writable = True
except Exception as exc:
    writable = False
check('Model storage writable', writable, str(model_dir))

stale = live_freshness_state('2026-08-27','2026-08-26',active_model_id='M')
fresh = live_freshness_state('2026-08-27','2026-08-27',active_model_id='M')
check('Live freshness guard', stale['code']=='MODEL_STALE' and not stale['can_predict'] and fresh['can_predict'], f"stale={stale['code']}, fresh={fresh['code']}")

research_model = active_model('research')
live_model = active_model('live')
live_ds = latest_dataset_by_mode('live')
live_pred = latest_prediction('live')
if live_ds:
    stats = dataset_market_stats(live_ds['id'])
    state = live_freshness_state(stats.get('last_date'), live_model.get('source_latest_date') if live_model else None,
                                 live_pred.get('data_freshness_date') if live_pred else None,
                                 live_pred.get('model_id') if live_pred else None,
                                 live_model.get('id') if live_model else None)
    check('Runtime live state', True, json.dumps(state))
else:
    check('Runtime live state', True, 'Belum ada live dataset; bukan error untuk instalasi baru.')

print('\nBITCOIN FORECAST INTELLIGENCE • FINAL ACCEPTANCE CHECK')
print('='*72)
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL':4}  {name:<28} {detail}")
passed = sum(ok for _,ok,_ in checks)
print('-'*72)
print(f'{passed}/{len(checks)} checks passed')
if passed != len(checks):
    raise SystemExit(1)
print('ACCEPTANCE CHECK PASS')
