"""Orkestrasi operasional untuk prediksi harian terbaru.

Research engine tetap terpisah dan tidak diubah. Modul ini hanya mengatur sinkronisasi
snapshot live, freshness model, retraining saat diperlukan, dan penyimpanan prediksi.
"""
from pathlib import Path
import pandas as pd

from services.repository import (
    latest_dataset_by_mode, list_datasets, create_dataset, save_market_data,
    update_dataset_sync, load_market_data, create_model, active_model,
    create_prediction, latest_prediction, create_experiment, finish_experiment,
)
from research_engine.config import (
    RESEARCH_SYMBOL, RESEARCH_INTERVAL, LIVE_DATASET_NAME, LIVE_DATA_START,
    LIVE_MODEL_MODE, LIVE_DEPLOYMENT_PROTOCOL, SCENARIOS,
    RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED, PSO_VAL_RATIO, ELM_FIXED_SEED,
)
from research_engine.live import (
    live_end_exclusive, download_live_data, build_live_model, predict_live_h1,
    artifact_source_latest_date, live_freshness_state,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'storage' / 'models'


def ensure_live_dataset():
    d = latest_dataset_by_mode(LIVE_MODEL_MODE)
    if d:
        return d
    did = create_dataset(
        LIVE_DATASET_NAME, RESEARCH_SYMBOL, LIVE_DATA_START, live_end_exclusive(),
        RESEARCH_INTERVAL,
        description='Data harian BTC-USD untuk prediksi terbaru.',
        is_locked=False, mode=LIVE_MODEL_MODE, source='Yahoo Finance',
    )
    return next(x for x in list_datasets(LIVE_MODEL_MODE) if x['id'] == did)


def refresh_live_data():
    d = ensure_live_dataset()
    raw = download_live_data(LIVE_DATA_START)
    save_market_data(d['id'], raw)
    update_dataset_sync(d['id'], live_end_exclusive())
    return latest_dataset_by_mode(LIVE_MODEL_MODE), raw


def model_source_date(model):
    if not model:
        return None
    if model.get('source_latest_date'):
        return model['source_latest_date']
    try:
        return artifact_source_latest_date(model['model_path'])
    except Exception:
        return None


def get_live_state(dataset=None, raw=None, model=None):
    dataset = dataset or latest_dataset_by_mode(LIVE_MODEL_MODE)
    if raw is None:
        raw = load_market_data(dataset['id']) if dataset else pd.DataFrame()
    model = model or active_model(LIVE_MODEL_MODE)
    latest_date = pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d') if raw is not None and not raw.empty else None
    pred = latest_prediction(LIVE_MODEL_MODE)
    state = live_freshness_state(
        latest_date,
        model_source_date(model),
        prediction_data_date=pred.get('data_freshness_date') if pred else None,
        prediction_model_id=pred.get('model_id') if pred else None,
        active_model_id=model.get('id') if model else None,
    )
    return state, pred, model, raw


def train_live_model(dataset, raw, notes='Pembaruan model prediksi harian.'):
    eid = create_experiment(
        'Pembaruan Model Prediksi Live', 'live_deployment', dataset['id'],
        notes=notes,
        config={
            'scenario': RESEARCH_CHAMPION_SCENARIO,
            'protocol': LIVE_DEPLOYMENT_PROTOCOL,
            'pso_seed': CHAMPION_PSO_SEED,
            'val_ratio': PSO_VAL_RATIO,
            'fixed_seed': ELM_FIXED_SEED,
        },
    )
    stamp = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
    path = MODEL_DIR / f'live_pso_elm_s4_{stamp}.pkl'
    try:
        summary = build_live_model(raw, path)
        latest_date = pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
        mid = create_model(
            dataset_id=dataset['id'], scenario=summary.scenario, algorithm='PSO-ELM',
            n_hidden=summary.n_hidden, c_value=summary.C,
            feature_set=SCENARIOS[summary.scenario], model_path=path,
            mape=None, experiment_id=eid, trained_until=summary.trained_until,
            status='active',
            config={
                'protocol': LIVE_DEPLOYMENT_PROTOCOL,
                'pso_seed': CHAMPION_PSO_SEED,
                'val_ratio': PSO_VAL_RATIO,
                'fixed_seed': ELM_FIXED_SEED,
                'trained_samples': summary.trained_samples,
            },
            notes='Model operasional untuk prediksi BTC-USD H+1.',
            model_mode=LIVE_MODEL_MODE, source_latest_date=latest_date,
        )
        finish_experiment(eid)
        return active_model(LIVE_MODEL_MODE), summary
    except Exception:
        finish_experiment(eid, 'failed')
        raise


def create_live_prediction(dataset, raw, model):
    latest_date = pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
    result = predict_live_h1(raw, model['model_path'])
    pid = create_prediction(
        model['id'], result['input_date'], result['prediction_date'],
        result['last_close'], result['predicted_close'], result['delta_usd'],
        result['delta_pct'], result['feature_snapshot'],
        prediction_mode=LIVE_MODEL_MODE, data_freshness_date=latest_date,
    )
    return result | {'id': pid, 'model_id': model['id'], 'n_hidden': model['n_hidden'], 'C': model['c_value']}
