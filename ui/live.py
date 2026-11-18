from pathlib import Path
import json
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    list_datasets, latest_dataset_by_mode, create_dataset, save_market_data,
    update_dataset_sync, load_market_data, dataset_market_stats,
    create_model, active_model, list_models, set_active_model, delete_model,
    create_prediction, list_predictions, latest_prediction, delete_prediction,
    create_experiment, finish_experiment,
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


def _fig_style(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=45, b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', legend=dict(orientation='h', y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(148,163,184,.15)')
    return fig


def _ensure_live_dataset():
    d = latest_dataset_by_mode(LIVE_MODEL_MODE)
    if d:
        return d
    did = create_dataset(
        LIVE_DATASET_NAME, RESEARCH_SYMBOL, LIVE_DATA_START, live_end_exclusive(),
        RESEARCH_INTERVAL,
        description='Snapshot operasional BTC-USD yang diperbarui untuk Mode Prediksi Live.',
        is_locked=False, mode=LIVE_MODEL_MODE, source='Yahoo Finance',
    )
    return next(x for x in list_datasets(LIVE_MODEL_MODE) if x['id'] == did)


def _refresh_live_data():
    d = _ensure_live_dataset()
    raw = download_live_data(LIVE_DATA_START)
    save_market_data(d['id'], raw)
    update_dataset_sync(d['id'], live_end_exclusive())
    return d['id'], raw


def _model_source_date(model):
    if not model:
        return None
    model_date = model.get('source_latest_date')
    if model_date:
        return model_date
    try:
        return artifact_source_latest_date(model['model_path'])
    except Exception:
        return None


def _page_state(d, raw, model):
    latest_date = None
    if d and raw is not None and not raw.empty:
        latest_date = pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
    pred = latest_prediction(LIVE_MODEL_MODE)
    return live_freshness_state(
        latest_date,
        _model_source_date(model),
        prediction_data_date=pred.get('data_freshness_date') if pred else None,
        prediction_model_id=pred.get('model_id') if pred else None,
        active_model_id=model.get('id') if model else None,
    ), pred


def _status_callout(state):
    code = state['code']
    if code == 'NO_DATA':
        st.markdown('<div class="callout extension"><b>Live belum siap</b><br>Sinkronkan data BTC-USD terlebih dahulu.</div>', unsafe_allow_html=True)
    elif code == 'NO_MODEL':
        st.markdown('<div class="callout extension"><b>Data tersedia, model belum ada</b><br>Bangun model live S4 sebelum melakukan prediksi.</div>', unsafe_allow_html=True)
    elif code in ('MODEL_STALE', 'MODEL_DATE_UNKNOWN'):
        st.markdown(
            f'<div class="callout danger"><b>Model live perlu diperbarui</b><br>{state["label"]}. '
            'Tombol prediksi dikunci sampai model dilatih ulang pada snapshot terbaru.</div>',
            unsafe_allow_html=True,
        )
    elif code == 'PREDICTION_CURRENT':
        st.markdown(
            f'<div class="callout live"><b>Live pipeline sudah current</b><br>Data, model, dan prediksi memakai snapshot {state["data_latest_date"]}. '
            'Prediksi baru tidak diperlukan sampai data berubah atau model dibangun ulang.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="callout live"><b>Model live fresh</b><br>Model aktif sudah dilatih pada snapshot {state["data_latest_date"]}. '
            'Prediksi H+1 siap dijalankan.</div>',
            unsafe_allow_html=True,
        )


def _live_status_cards(d, raw, model, state):
    stats = dataset_market_stats(d['id']) if d else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card('Data Terbaru', stats.get('last_date') or '—', 'candle harian terbaru')
    with c2:
        last_close = stats.get('last_close')
        metric_card('Close Terbaru', f"${last_close:,.2f}" if last_close is not None else '—', RESEARCH_SYMBOL)
    with c3:
        metric_card('Model Live', model['id'] if model else '—', f"NH {model['n_hidden']}" if model else 'belum dibangun')
    with c4:
        label = 'FRESH' if state['model_fresh'] else ('STALE' if state['code'] == 'MODEL_STALE' else '—')
        metric_card('Kesegaran Model', label, state.get('model_source_date') or 'belum tersedia')


def _train_live_model(d, raw, notes='Operational refit; bukan evaluasi artikel.'):
    eid = create_experiment(
        'Live Deployment Refit', 'live_deployment', d['id'],
        notes=notes,
        config={'scenario': RESEARCH_CHAMPION_SCENARIO, 'protocol': LIVE_DEPLOYMENT_PROTOCOL,
                'pso_seed': CHAMPION_PSO_SEED, 'val_ratio': PSO_VAL_RATIO,
                'fixed_seed': ELM_FIXED_SEED},
    )
    stamp = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
    path = MODEL_DIR / f'live_pso_elm_s4_{stamp}.pkl'
    try:
        summary = build_live_model(raw, path)
        latest_date = pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
        mid = create_model(
            dataset_id=d['id'], scenario=summary.scenario, algorithm='PSO-ELM',
            n_hidden=summary.n_hidden, c_value=summary.C,
            feature_set=SCENARIOS[summary.scenario], model_path=path,
            mape=None, experiment_id=eid, trained_until=summary.trained_until,
            status='active',
            config={'protocol': LIVE_DEPLOYMENT_PROTOCOL, 'pso_seed': CHAMPION_PSO_SEED,
                    'val_ratio': PSO_VAL_RATIO, 'fixed_seed': ELM_FIXED_SEED,
                    'trained_samples': summary.trained_samples},
            notes='Live operational model. MAPE penelitian tidak dipindahkan sebagai evaluasi live.',
            model_mode=LIVE_MODEL_MODE, source_latest_date=latest_date,
        )
        finish_experiment(eid)
        return mid, summary, path, latest_date
    except Exception:
        finish_experiment(eid, 'failed')
        raise


def _create_live_prediction(d, raw, model):
    latest_date = pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
    result = predict_live_h1(raw, model['model_path'])
    pid = create_prediction(
        model['id'], result['input_date'], result['prediction_date'],
        result['last_close'], result['predicted_close'], result['delta_usd'],
        result['delta_pct'], result['feature_snapshot'],
        prediction_mode=LIVE_MODEL_MODE, data_freshness_date=latest_date,
    )
    st.session_state['latest_live_prediction'] = result | {'id': pid}
    return result | {'id': pid}


def render():
    hero(
        'Mode Prediksi Live',
        'Sinkronkan data BTC-USD terbaru, latih ulang champion PSO–ELM S4 dengan pipeline yang sama, lalu prediksi Close H+1 dari candle harian terbaru.',
        'MODE OPERASIONAL • DATA HARIAN TERBARU'
    )
    st.markdown(
        '<div class="callout live"><b>Terpisah dari Research Mode</b><br>'
        'Mode Live tidak mengubah dataset penelitian, parity contract, POC, tournament, atau champion evaluation. '
        'Karena penelitian memakai interval <b>1d</b>, “live” berarti <b>daily candle terbaru yang tersedia dari Yahoo Finance</b>, bukan tick/intraday real-time.</div>',
        unsafe_allow_html=True,
    )

    d = latest_dataset_by_mode(LIVE_MODEL_MODE)
    raw = load_market_data(d['id']) if d else pd.DataFrame()
    model = active_model(LIVE_MODEL_MODE)
    state, latest_db_pred = _page_state(d, raw, model)
    _live_status_cards(d, raw, model, state)
    _status_callout(state)

    st.markdown('### Alur Live')
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown('<div class="step-card"><span>01</span><b>Sinkronkan Data</b><small>Yahoo Finance • BTC-USD • 1d</small></div>', unsafe_allow_html=True)
        if st.button('Sinkronkan data sekarang', type='primary', use_container_width=True, key='live_refresh'):
            try:
                with st.status('Mengambil candle BTC-USD terbaru...', expanded=True) as status:
                    _, live_raw = _refresh_live_data()
                    st.write(f'{len(live_raw):,} baris tersedia sampai {live_raw.index.max().strftime("%Y-%m-%d")}.')
                    status.update(label='Snapshot live diperbarui', state='complete', expanded=False)
                st.success('Data live berhasil disinkronkan.'); st.rerun()
            except Exception as exc:
                st.error(f'Gagal sinkronisasi data live: {exc}')

    with s2:
        st.markdown('<div class="step-card"><span>02</span><b>Perbarui Model</b><small>PSO 20×50 • S4 • full labeled refit</small></div>', unsafe_allow_html=True)
        if st.button('Build / refresh live model', use_container_width=True, key='live_build'):
            d = latest_dataset_by_mode(LIVE_MODEL_MODE)
            if not d:
                st.warning('Sinkronkan data live terlebih dahulu.')
            else:
                raw = load_market_data(d['id'])
                if raw.empty:
                    st.warning('Snapshot live masih kosong.')
                else:
                    try:
                        with st.status('Melatih live deployment model...', expanded=True) as status:
                            st.write('Feature engineering S4 dan Target H+1...')
                            t0 = time.perf_counter()
                            mid, summary, _, _ = _train_live_model(d, raw)
                            st.write(f'PSO memilih NH={summary.n_hidden}, C={summary.C:.3e}.')
                            status.update(label=f'Live model {mid} aktif', state='complete', expanded=False)
                        st.success(f'Live model selesai dalam {time.perf_counter()-t0:.1f} detik.'); st.rerun()
                    except Exception as exc:
                        st.error(f'Build live model gagal: {exc}')

    with s3:
        st.markdown('<div class="step-card"><span>03</span><b>Prediksi H+1</b><small>latest candle → next calendar day</small></div>', unsafe_allow_html=True)
        predict_disabled = not state['can_predict'] or state['prediction_current']
        predict_help = (
            'Model harus fresh terhadap candle terbaru.' if not state['can_predict']
            else ('Prediksi untuk model dan snapshot ini sudah tersimpan.' if state['prediction_current'] else None)
        )
        if st.button('Prediksi live H+1', use_container_width=True, key='live_predict',
                     disabled=predict_disabled, help=predict_help):
            try:
                d = latest_dataset_by_mode(LIVE_MODEL_MODE)
                raw = load_market_data(d['id'])
                model = active_model(LIVE_MODEL_MODE)
                _create_live_prediction(d, raw, model)
                st.success('Prediksi live tersimpan ke SQLite.'); st.rerun()
            except Exception as exc:
                st.error(f'Prediksi live gagal: {exc}')

    with st.expander('One-click: Sinkronkan → Train → Predict'):
        st.caption('Menjalankan PSO 20×50 ulang sehingga dapat membutuhkan waktu. Research/parity data tidak disentuh.')
        confirm = st.checkbox('Saya memahami proses ini akan melatih ulang live model.', key='live_oneclick_confirm')
        if st.button('Jalankan Live Pipeline Lengkap', type='primary', use_container_width=True,
                     disabled=not confirm, key='live_oneclick_run'):
            try:
                with st.status('Live pipeline berjalan...', expanded=True) as status:
                    _, raw = _refresh_live_data(); st.write(f'1/3 Data synced sampai {raw.index.max().strftime("%Y-%m-%d")}.')
                    d = latest_dataset_by_mode(LIVE_MODEL_MODE)
                    mid, summary, path, latest_date = _train_live_model(d, raw, notes='One-click operational live pipeline.')
                    st.write(f'2/3 Model {mid}: NH={summary.n_hidden}, C={summary.C:.3e}.')
                    result = predict_live_h1(raw, path)
                    pid = create_prediction(mid, result['input_date'], result['prediction_date'], result['last_close'],
                                            result['predicted_close'], result['delta_usd'], result['delta_pct'],
                                            result['feature_snapshot'], prediction_mode=LIVE_MODEL_MODE,
                                            data_freshness_date=latest_date)
                    st.session_state['latest_live_prediction'] = result | {'id': pid}
                    st.write(f'3/3 Prediksi {result["prediction_date"]}: ${result["predicted_close"]:,.2f}.')
                    status.update(label='Live pipeline selesai', state='complete', expanded=False)
                st.success('Data, model, dan prediction history berhasil diperbarui.'); st.rerun()
            except Exception as exc:
                st.error(f'Live pipeline gagal: {exc}')

    d = latest_dataset_by_mode(LIVE_MODEL_MODE)
    raw = load_market_data(d['id']) if d else pd.DataFrame()
    model = active_model(LIVE_MODEL_MODE)
    if d and not raw.empty:
        st.markdown('### Konteks pasar terbaru')
        recent = raw.tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], mode='lines', name='BTC Close', line=dict(width=2.2)))
        result = st.session_state.get('latest_live_prediction')
        if result:
            fig.add_trace(go.Scatter(x=[pd.Timestamp(result['prediction_date'])], y=[result['predicted_close']],
                                     mode='markers', name='Prediksi H+1', marker=dict(size=11, symbol='diamond')))
        fig.update_layout(title='BTC-USD • 120 Daily Candles Terakhir', yaxis_title='USD')
        st.plotly_chart(_fig_style(fig, 390), use_container_width=True)

    result = st.session_state.get('latest_live_prediction')
    if result:
        st.markdown('### Prediksi live terbaru')
        up = result['delta_usd'] >= 0
        a, b, c = st.columns(3)
        with a: metric_card('Close Terbaru', f"${result['last_close']:,.2f}", result['input_date'])
        with b: metric_card('Prediksi H+1', f"${result['predicted_close']:,.2f}", result['prediction_date'])
        with c: metric_card('Perubahan Prediksi', f"{'▲' if up else '▼'} {result['delta_pct']:+.2f}%", f"{result['delta_usd']:+,.2f} USD")
        st.caption(f"Prediction ID: {result['id']} • LIVE PSO–ELM S4 • NH={result['n_hidden']} • C={result['C']:.3e}")
        with st.expander('Feature snapshot live'):
            snap = pd.DataFrame({'Feature': list(result['feature_snapshot']), 'Value': list(result['feature_snapshot'].values())})
            st.dataframe(snap, use_container_width=True, hide_index=True)

    st.markdown('### Riwayat prediksi live')
    rows = list_predictions(mode=LIVE_MODEL_MODE)
    if rows:
        df = pd.DataFrame(rows)
        cols = ['id','input_date','prediction_date','last_close','predicted_close','delta_pct','data_freshness_date','created_at']
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.download_button('Export Live Predictions CSV', df.to_csv(index=False).encode('utf-8'),
                           'live_predictions.csv', 'text/csv', use_container_width=True, key='live_predictions_export')
        labels = {f"{r['id']} — {r['prediction_date']}": r for r in rows}
        target = labels[st.selectbox('Kelola histori live', list(labels), key='live_manage_prediction')]
        confirm_delete = st.checkbox('Konfirmasi hapus histori terpilih', key='live_delete_confirm')
        if st.button('Hapus histori live', use_container_width=True, disabled=not confirm_delete, key='live_delete_prediction'):
            delete_prediction(target['id']); st.success('Histori live dihapus.'); st.rerun()
    else:
        st.info('Belum ada live prediction tersimpan.')

    with st.expander('Live model registry'):
        models = list_models(LIVE_MODEL_MODE)
        if not models:
            st.info('Belum ada live model.')
        else:
            st.dataframe(pd.DataFrame(models)[['id','status','n_hidden','c_value','trained_until','source_latest_date','created_at']],
                         use_container_width=True, hide_index=True)
            labels = {f"{m['id']} — {m['status']} — source {m.get('source_latest_date') or '—'}": m for m in models}
            row = labels[st.selectbox('Kelola live model', list(labels), key='manage_live_model')]
            a,b = st.columns(2)
            if row['status'] != 'active':
                if a.button('Jadikan live model aktif', use_container_width=True, key='live_activate_model'):
                    set_active_model(row['id']); st.rerun()
            else:
                a.info('Live model ini aktif.')
            confirm_model_delete = st.checkbox('Konfirmasi hapus model', key='live_model_delete_confirm')
            if b.button('Hapus live model', use_container_width=True, disabled=not confirm_model_delete, key='live_delete_model'):
                delete_model(row['id']); st.success('Live model dihapus.'); st.rerun()
