from pathlib import Path
import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    list_datasets, load_market_data, create_experiment, save_runs, finish_experiment,
    latest_scenario_mape, create_model, list_models, get_model, set_active_model,
    delete_model, active_model, create_prediction, list_predictions, delete_prediction,
)
from research_engine.preprocessing import prepare_market_data
from research_engine.deployment import evaluate_source_champion, train_deployment_model, predict_h1
from research_engine.config import (
    RESEARCH_CHAMPION_SCENARIO, CHAMPION_PSO_SEED, PSO_VAL_RATIO,
    ELM_FIXED_SEED, SCENARIOS, DEPLOYMENT_PROTOCOL, RESEARCH_MODEL_MODE,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "storage" / "models"


def _dataset_select(key='model_dataset'):
    ds = [d for d in list_datasets() if (d.get('mode') or 'research') != 'live']
    if not ds:
        st.info('Belum ada dataset. Buat snapshot BTC-USD terlebih dahulu.')
        return None
    labels = {f"{d['id']} — {d['name']}": d for d in ds}
    return labels[st.selectbox('Dataset sumber', list(labels), key=key)]


def _transparent_layout(fig, height=400):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=46, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', legend=dict(orientation='h', y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(148,163,184,.16)')
    return fig


def models_page():
    hero(
        'Champion Model & Registry',
        'Replikasi champion PSO–ELM S4 dari source dipisahkan dari proses deployment H+1 agar evaluasi penelitian dan inferensi aplikasi tetap dapat diaudit.',
        'MODEL LIFECYCLE • SOURCE CHAMPION S4'
    )

    st.markdown(
        '<div class="callout source"><b>Source-locked champion</b><br>'
        'CELL 15 pada source menetapkan PSO–ELM + S4 (Tren SMA), val_ratio 25%, fixed ELM seed 42, dan PSO seed 99 untuk evaluasi WFV 5-fold.</div>', unsafe_allow_html=True
    )

    tab_eval, tab_build, tab_registry = st.tabs(['Champion Evaluation', 'Build Deployment Model', 'Model Registry'])

    with tab_eval:
        st.markdown('### Replikasi analisis champion')
        st.caption('Bagian ini tetap berupa evaluasi WFV. Tidak menghasilkan model deployment aktif kecuali Anda membangunnya pada tab berikutnya.')
        d = _dataset_select('champ_eval_ds')
        if d and st.button('Jalankan Champion Evaluation', type='primary', use_container_width=True):
            raw = load_market_data(d['id'])
            if raw.empty:
                st.error('Snapshot dataset belum berisi data market.')
            else:
                eid = create_experiment(
                    'Champion Evaluation — PSO-ELM S4', 'champion_evaluation', d['id'],
                    notes='Replikasi CELL 15 source.',
                    config={'scenario': RESEARCH_CHAMPION_SCENARIO, 'pso_seed': CHAMPION_PSO_SEED,
                            'val_ratio': PSO_VAL_RATIO, 'fixed_seed': ELM_FIXED_SEED}
                )
                bar = st.progress(0, text='Menyiapkan champion evaluation...')
                try:
                    processed = prepare_market_data(raw)
                    def cb(v, t): bar.progress(min(v, 1.0), text=t)
                    summary, folds, plot_df = evaluate_source_champion(processed, cb)
                    run = {
                        'scenario': RESEARCH_CHAMPION_SCENARIO, 'model_type': 'PSO-ELM',
                        'run_number': 1, 'seed': CHAMPION_PSO_SEED,
                        'mae': summary['mae'], 'rmse': summary['rmse'], 'mape': summary['mape'],
                        'folds': [
                            {'fold': int(r.fold), 'train_size': int(r.train_size), 'test_size': int(r.test_size),
                             'n_hidden': int(r.n_hidden), 'C': float(r.C), 'mae': float(r.mae),
                             'rmse': float(r.rmse), 'mape': float(r.mape)}
                            for r in folds.itertuples(index=False)
                        ]
                    }
                    save_runs(eid, [run]); finish_experiment(eid)
                    st.session_state['champ_summary'] = summary
                    st.session_state['champ_folds'] = folds
                    st.session_state['champ_plot'] = plot_df
                    bar.empty(); st.success(f'Champion evaluation selesai dan tersimpan sebagai {eid}.')
                except Exception as exc:
                    finish_experiment(eid, 'failed'); bar.empty(); st.error(str(exc))

        summary = st.session_state.get('champ_summary')
        folds = st.session_state.get('champ_folds')
        plot_df = st.session_state.get('champ_plot')
        if summary:
            a, b, c, dcol = st.columns(4)
            with a: metric_card('Champion', 'S4', 'Tren SMA')
            with b: metric_card('MAPE', f"{summary['mape']:.4f}%", 'Mean WFV 5-fold')
            with c: metric_card('RMSE', f"${summary['rmse']:,.2f}", 'Mean WFV')
            with dcol: metric_card('MAE', f"${summary['mae']:,.2f}", 'Mean WFV')

        if folds is not None and not folds.empty:
            show = folds.copy()
            show['C'] = show['C'].map(lambda x: f"{x:.3e}")
            show['mae'] = show['mae'].map(lambda x: f"{x:,.2f}")
            show['rmse'] = show['rmse'].map(lambda x: f"{x:,.2f}")
            show['mape'] = show['mape'].map(lambda x: f"{x:.4f}%")
            st.dataframe(show, use_container_width=True, hide_index=True)

        if plot_df is not None and not plot_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Actual'], mode='lines', name='Aktual', line=dict(width=2)))
            fig.add_trace(go.Scatter(x=plot_df['Date'], y=plot_df['Predicted'], mode='lines', name='Prediksi', line=dict(width=1.7, dash='dash')))
            fig.update_layout(title='Aktual vs Prediksi • Champion PSO–ELM S4', yaxis_title='USD')
            st.plotly_chart(_transparent_layout(fig, 430), use_container_width=True)

            fig2 = go.Figure(go.Scatter(
                x=plot_df['Date'], y=plot_df['Residual'], mode='markers', name='Residual',
                marker=dict(size=5, opacity=.65)
            ))
            fig2.add_hline(y=0, line_width=1.5, line_dash='dash')
            fig2.update_layout(title='Residual Champion Model', yaxis_title='Aktual − Prediksi (USD)')
            st.plotly_chart(_transparent_layout(fig2, 330), use_container_width=True)

    with tab_build:
        st.markdown('### Build model untuk inferensi H+1')
        st.markdown(
            '<div class="callout extension"><b>Application Deployment Extension</b><br>'
            'Source berhenti pada evaluasi champion WFV. Untuk kebutuhan sistem, champion S4 fit ulang pada seluruh data yang sudah memiliki Target, lalu dipakai memprediksi baris fitur terbaru yang Target H+1-nya belum diketahui. Algoritma PSO/ELM, fitur, val_ratio dan seed tetap mengikuti champion source.</div>', unsafe_allow_html=True
        )
        d = _dataset_select('deploy_ds')
        if d:
            c1, c2, c3 = st.columns(3)
            c1.metric('Scenario', 'S4', 'Tren SMA')
            c2.metric('PSO Seed', CHAMPION_PSO_SEED, 'source champion')
            c3.metric('Internal Validation', f'{PSO_VAL_RATIO:.0%}', 'fitness PSO')
            if st.button('Build & Activate Deployment Model', type='primary', use_container_width=True):
                raw = load_market_data(d['id'])
                if raw.empty:
                    st.error('Snapshot dataset kosong.')
                else:
                    stamp = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
                    path = MODEL_DIR / f"pso_elm_s4_{stamp}.pkl"
                    with st.status('Membangun model deployment...', expanded=True) as status:
                        try:
                            st.write('Menyiapkan feature engineering dan target H+1...')
                            t0 = time.perf_counter()
                            summary = train_deployment_model(raw, path)
                            st.write(f'PSO memilih NH={summary.n_hidden}, C={summary.C:.3e}.')
                            source_metric = latest_scenario_mape(RESEARCH_CHAMPION_SCENARIO, 'PSO-ELM')
                            mape = source_metric['mean_mape'] if source_metric else None
                            mid = create_model(
                                dataset_id=d['id'], scenario=summary.scenario, algorithm='PSO-ELM',
                                n_hidden=summary.n_hidden, c_value=summary.C,
                                feature_set=SCENARIOS[summary.scenario], model_path=path, mape=mape,
                                experiment_id=source_metric['experiment_id'] if source_metric else None,
                                trained_until=summary.trained_until, status='active',
                                config={'protocol': DEPLOYMENT_PROTOCOL, 'pso_seed': CHAMPION_PSO_SEED,
                                        'val_ratio': PSO_VAL_RATIO, 'fixed_seed': ELM_FIXED_SEED,
                                        'trained_samples': summary.trained_samples},
                                notes='Deployment extension dari champion S4 source.',
                                model_mode=RESEARCH_MODEL_MODE,
                                source_latest_date=pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
                            )
                            status.update(label=f'Model {mid} aktif', state='complete', expanded=False)
                            st.success(f'Model tersimpan dan aktif. Waktu build {time.perf_counter()-t0:.1f} detik.')
                            st.rerun()
                        except Exception as exc:
                            status.update(label='Build model gagal', state='error')
                            st.error(str(exc))

    with tab_registry:
        rows = list_models()
        if not rows:
            st.info('Belum ada model deployment. Bangun model dari tab sebelumnya.')
        else:
            cards = st.columns(min(3, len(rows)))
            for col, row in zip(cards, rows[:3]):
                with col:
                    status_class = 'active-dot' if row['status'] == 'active' else 'inactive-dot'
                    mape_txt = f"{row['mape']:.4f}%" if row.get('mape') is not None else '—'
                    st.markdown(
                        f'<div class="model-card"><div><span class="{status_class}"></span>'
                        f'<span class="model-status">{row["status"].upper()}</span></div>'
                        f'<div class="model-id">{row["id"]}</div>'
                        f'<div class="model-title">PSO–ELM • S4 • {(row.get("model_mode") or "research").upper()}</div>'
                        f'<div class="model-meta">NH {row["n_hidden"]} · C {row["c_value"]:.2e} · MAPE {mape_txt}</div>'
                        f'<div class="model-meta">trained until {row.get("trained_until") or "—"}</div></div>',
                        unsafe_allow_html=True
                    )

            table = pd.DataFrame(rows)
            st.dataframe(
                table[['id','model_mode','status','scenario','n_hidden','c_value','mape','trained_until','source_latest_date','dataset_name','created_at']],
                use_container_width=True, hide_index=True
            )
            labels = {f"{r['id']} — {r['status']} — NH {r['n_hidden']}": r for r in rows}
            row = labels[st.selectbox('Kelola model', list(labels), key='manage_model')]
            a, b = st.columns(2)
            if row['status'] != 'active':
                if a.button('Jadikan model aktif', use_container_width=True):
                    set_active_model(row['id']); st.success('Model diaktifkan.'); st.rerun()
            else:
                a.info('Model ini sedang aktif.')
            if b.button('Hapus model', use_container_width=True):
                delete_model(row['id']); st.success('Model dan artifact dihapus.'); st.rerun()


def prediction_page():
    hero(
        'Prediksi Harga Bitcoin H+1',
        'Inferensi menggunakan deployment model champion PSO–ELM S4. Hasil prediksi disimpan sebagai immutable machine output pada SQLite.',
        'DEPLOYMENT • H+1 INFERENCE'
    )

    models = list_models(mode=RESEARCH_MODEL_MODE)
    if not models:
        st.warning('Belum ada model. Buka Model Registry → Build Deployment Model terlebih dahulu.')
        return

    current = active_model(RESEARCH_MODEL_MODE)
    default_id = current['id'] if current else models[0]['id']
    options = {f"{m['id']} — {m['status']} — NH {m['n_hidden']}": m for m in models}
    keys = list(options)
    default_idx = next((i for i, k in enumerate(keys) if options[k]['id'] == default_id), 0)
    model_row = options[st.selectbox('Model deployment', keys, index=default_idx)]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Status', model_row['status'].upper())
    c2.metric('Scenario', 'S4')
    c3.metric('Hidden Neuron', model_row['n_hidden'])
    c4.metric('C', f"{model_row['c_value']:.2e}")

    st.markdown(
        '<div class="callout extension"><b>Interpretasi</b><br>'
        'Prediksi ini adalah implementasi deployment dari champion penelitian, bukan metrik evaluasi baru. Nilai H+1 tidak dapat diedit manual setelah tersimpan.</div>', unsafe_allow_html=True
    )

    if st.button('Jalankan Prediksi H+1', type='primary', use_container_width=True):
        raw = load_market_data(model_row['dataset_id'])
        if raw.empty:
            st.error('Dataset yang terhubung ke model tidak tersedia.')
        else:
            try:
                result = predict_h1(raw, model_row['model_path'])
                pid = create_prediction(
                    model_row['id'], result['input_date'], result['prediction_date'],
                    result['last_close'], result['predicted_close'], result['delta_usd'],
                    result['delta_pct'], result['feature_snapshot'],
                    prediction_mode=RESEARCH_MODEL_MODE, data_freshness_date=result['input_date']
                )
                st.session_state['latest_prediction'] = result | {'id': pid}
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.get('latest_prediction')
    if result:
        up = result['delta_usd'] >= 0
        trend = '▲' if up else '▼'
        a, b, c = st.columns([1, 1, 1])
        with a: metric_card('Close Terakhir', f"${result['last_close']:,.2f}", result['input_date'])
        with b: metric_card('Prediksi H+1', f"${result['predicted_close']:,.2f}", result['prediction_date'])
        with c: metric_card('Perubahan Prediksi', f"{trend} {result['delta_pct']:+.2f}%", f"{result['delta_usd']:+,.2f} USD")
        st.caption(f"Prediction ID: {result['id']} • PSO–ELM S4 • NH={result['n_hidden']} • C={result['C']:.3e}")

        snap = pd.DataFrame([result['feature_snapshot']]).T.reset_index()
        snap.columns = ['Feature', 'Value']
        with st.expander('Feature snapshot yang masuk ke model'):
            st.dataframe(snap, use_container_width=True, hide_index=True)

    st.markdown('### Riwayat prediksi')
    rows = list_predictions(mode=RESEARCH_MODEL_MODE)
    if not rows:
        st.info('Belum ada prediksi tersimpan.')
        return

    df = pd.DataFrame(rows)
    show_cols = ['id','input_date','prediction_date','last_close','predicted_close','delta_pct','scenario','created_at']
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    chart_df = df.sort_values('input_date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df['input_date'], y=chart_df['last_close'], mode='lines+markers', name='Close terakhir'))
    fig.add_trace(go.Scatter(x=chart_df['prediction_date'], y=chart_df['predicted_close'], mode='lines+markers', name='Prediksi H+1'))
    fig.update_layout(title='Riwayat Prediksi', yaxis_title='USD')
    st.plotly_chart(_transparent_layout(fig, 340), use_container_width=True)

    labels = {f"{r['id']} — {r['prediction_date']}": r for r in rows}
    target = labels[st.selectbox('Kelola histori prediksi', list(labels), key='manage_prediction')]
    if st.button('Hapus histori prediksi', use_container_width=True):
        delete_prediction(target['id']); st.success('Histori prediksi dihapus.'); st.rerun()
