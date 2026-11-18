"""Generate demo-friendly historical prediction history using the ACTIVE live model.

IMPORTANT:
- This is HISTORICAL REPLAY, not a true walk-forward backtest.
- The current active live model/scaler is replayed on older daily candles.
- Rows are stored with source_type='historical_replay' so they remain distinguishable
  from genuine operational predictions.
- Existing research results, parity, models, and operational predictions are not changed.

Usage from project root:
    python scripts/generate_backtest_history.py
    python scripts/generate_backtest_history.py --days 60
    python scripts/generate_backtest_history.py --days 90 --replace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from database.database import init_db, connect
from services.repository import latest_dataset_by_mode, load_market_data, active_model, create_prediction
from research_engine.config import LIVE_MODEL_MODE
from research_engine.preprocessing import prepare_inference_features
from research_engine.deployment import predict_h1

SOURCE_TYPE = 'historical_replay'


def _existing(model_id: str, input_date: str) -> bool:
    with connect() as con:
        r = con.execute(
            "SELECT 1 FROM predictions WHERE model_id=? AND input_date=? "
            "AND prediction_mode=? AND COALESCE(source_type,'operational')=? LIMIT 1",
            (model_id, input_date, LIVE_MODEL_MODE, SOURCE_TYPE),
        ).fetchone()
        return bool(r)


def _delete_existing_replay(model_id: str | None = None) -> int:
    with connect() as con:
        if model_id:
            cur = con.execute(
                "DELETE FROM predictions WHERE prediction_mode=? AND source_type=? AND model_id=?",
                (LIVE_MODEL_MODE, SOURCE_TYPE, model_id),
            )
        else:
            cur = con.execute(
                "DELETE FROM predictions WHERE prediction_mode=? AND source_type=?",
                (LIVE_MODEL_MODE, SOURCE_TYPE),
            )
        return int(cur.rowcount or 0)


def main():
    ap = argparse.ArgumentParser(description='Generate historical replay prediction history.')
    ap.add_argument('--days', type=int, default=60, help='Jumlah tanggal replay (default: 60).')
    ap.add_argument('--replace', action='store_true', help='Hapus replay lama untuk model aktif lalu generate ulang.')
    args = ap.parse_args()

    if args.days < 1 or args.days > 365:
        raise SystemExit('--days harus antara 1 dan 365.')

    init_db()
    dataset = latest_dataset_by_mode(LIVE_MODEL_MODE)
    model = active_model(LIVE_MODEL_MODE)
    if not dataset:
        raise SystemExit('ERROR: Dataset live belum tersedia. Jalankan Prediksi Bitcoin terlebih dahulu.')
    if not model:
        raise SystemExit('ERROR: Live model aktif belum tersedia. Jalankan Prediksi Bitcoin terlebih dahulu.')

    raw = load_market_data(dataset['id'])
    if raw.empty:
        raise SystemExit('ERROR: Dataset live kosong.')

    infer = prepare_inference_features(raw)
    if len(infer) < 35:
        raise SystemExit('ERROR: Data tidak cukup untuk historical replay.')

    # Exclude latest inference date so current operational prediction remains the latest real prediction.
    candidate_dates = list(infer.index[:-1])[-args.days:]
    if not candidate_dates:
        raise SystemExit('ERROR: Tidak ada tanggal replay yang tersedia.')

    if args.replace:
        removed = _delete_existing_replay(model['id'])
        print(f'Replay lama yang dihapus: {removed}')

    created = skipped = failed = 0
    print('=' * 76)
    print('BITCOIN FORECAST INTELLIGENCE — HISTORICAL REPLAY GENERATOR')
    print('=' * 76)
    print(f'Dataset      : {dataset["name"]} ({dataset["id"]})')
    print(f'Live model   : {model["id"]} • NH={model["n_hidden"]} • C={model["c_value"]:.4g}')
    print(f'Jumlah replay: {len(candidate_dates)} hari')
    print('Catatan      : historical replay, bukan true walk-forward backtest.')
    print('-' * 76)

    for i, dt in enumerate(candidate_dates, 1):
        input_date = pd.Timestamp(dt).strftime('%Y-%m-%d')
        if _existing(model['id'], input_date):
            skipped += 1
            print(f'[{i:02d}/{len(candidate_dates)}] SKIP {input_date} (sudah ada)')
            continue
        try:
            cutoff = raw.loc[:pd.Timestamp(dt)].copy()
            result = predict_h1(cutoff, model['model_path'])

            # For transparency only: include the realized next close when available.
            next_date = pd.Timestamp(result['prediction_date'])
            actual_next_close = None
            if next_date in raw.index:
                actual_next_close = float(raw.loc[next_date, 'Close'])
            features = dict(result['feature_snapshot'])
            features['_history_source'] = SOURCE_TYPE
            features['_replay_model_id'] = model['id']
            if actual_next_close is not None:
                features['_actual_next_close'] = actual_next_close

            pid = create_prediction(
                model['id'], result['input_date'], result['prediction_date'],
                result['last_close'], result['predicted_close'], result['delta_usd'],
                result['delta_pct'], features,
                prediction_mode=LIVE_MODEL_MODE,
                data_freshness_date=result['input_date'],
                source_type=SOURCE_TYPE,
            )
            created += 1
            print(
                f'[{i:02d}/{len(candidate_dates)}] OK   {result["input_date"]} -> '
                f'{result["prediction_date"]}  ${result["predicted_close"]:,.2f}  {pid}'
            )
        except Exception as exc:
            failed += 1
            print(f'[{i:02d}/{len(candidate_dates)}] FAIL {input_date}: {exc}')

    print('-' * 76)
    print(f'Selesai. Dibuat={created}, dilewati={skipped}, gagal={failed}.')
    print("Buka aplikasi -> 'Riwayat & Laporan' atau 'Prediksi Bitcoin' untuk melihat history.")
    print("Kolom 'Sumber' akan menandai record ini sebagai 'Replay Historis'.")
    print('=' * 76)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
