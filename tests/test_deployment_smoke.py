from pathlib import Path
import numpy as np
import pandas as pd

from research_engine.deployment import train_deployment_model, predict_h1


def synthetic_market(n=90):
    idx = pd.date_range('2025-01-01', periods=n, freq='D')
    base = 50000 + np.linspace(0, 4000, n) + np.sin(np.arange(n)/4)*600
    return pd.DataFrame({
        'Open': base - 80,
        'High': base + 250,
        'Low': base - 260,
        'Close': base,
        'Volume': 1000 + np.arange(n)*3,
    }, index=idx)


def test_deployment_artifact_roundtrip(tmp_path: Path):
    raw = synthetic_market()
    path = tmp_path / 'model.pkl'
    summary = train_deployment_model(raw, path, N_pso=3, I_max=2)
    assert path.exists()
    assert summary.scenario == 'S4 (Tren SMA)'
    result = predict_h1(raw, path)
    assert result['prediction_date'] == '2025-04-01'
    assert np.isfinite(result['predicted_close'])
    assert set(result['feature_snapshot']) == {'Close','Lag_1','Lag_2','SMA_10','SMA_30'}
