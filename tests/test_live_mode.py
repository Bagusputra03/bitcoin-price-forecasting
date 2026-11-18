from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from research_engine.live import (
    live_end_exclusive, build_live_model, predict_live_h1, live_freshness_state
)


def synthetic_market(n=90):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    base = 60000 + np.linspace(0, 3000, n) + np.sin(np.arange(n)/5)*400
    return pd.DataFrame({
        'Open': base-70, 'High': base+200, 'Low': base-220,
        'Close': base, 'Volume': 1200+np.arange(n)*4,
    }, index=idx)


def test_live_end_is_tomorrow_utc():
    now = datetime(2026, 8, 28, 1, 0, tzinfo=timezone.utc)
    assert live_end_exclusive(now) == '2026-08-29'


def test_live_model_roundtrip(tmp_path: Path):
    raw = synthetic_market()
    path = tmp_path / 'live.pkl'
    summary = build_live_model(raw, path, N_pso=3, I_max=2)
    result = predict_live_h1(raw, path)
    assert summary.scenario == 'S4 (Tren SMA)'
    assert result['prediction_mode'] == 'live'
    assert result['prediction_date'] == '2026-04-01'
    assert np.isfinite(result['predicted_close'])


def test_live_freshness_guard():
    missing = live_freshness_state(None)
    assert missing['code'] == 'NO_DATA' and not missing['can_predict']

    no_model = live_freshness_state('2026-08-27')
    assert no_model['code'] == 'NO_MODEL' and not no_model['can_predict']

    stale = live_freshness_state('2026-08-27', '2026-08-26', active_model_id='MDL-A')
    assert stale['code'] == 'MODEL_STALE' and not stale['can_predict']

    fresh = live_freshness_state('2026-08-27', '2026-08-27', active_model_id='MDL-A')
    assert fresh['code'] == 'MODEL_FRESH' and fresh['can_predict']

    current = live_freshness_state(
        '2026-08-27', '2026-08-27', prediction_data_date='2026-08-27',
        prediction_model_id='MDL-A', active_model_id='MDL-A'
    )
    assert current['code'] == 'PREDICTION_CURRENT'
    assert current['prediction_current']
