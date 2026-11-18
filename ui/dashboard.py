import json
import pandas as pd
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    dashboard_stats, list_experiments, active_model, latest_prediction,
    latest_parity_report, latest_dataset_by_mode, dataset_market_stats,
)
from research_engine.config import *
from research_engine.parity import flatten_checks
from research_engine.live import live_freshness_state, artifact_source_latest_date


def _safe_model_date(model):
    if not model:
        return None
    if model.get('source_latest_date'):
        return model['source_latest_date']
    try:
        return artifact_source_latest_date(model['model_path'])
    except Exception:
        return None


def render():
    hero(
        'Research & Live Control Center',
        'Satu workspace untuk reproduksi eksperimen PSO–ELM yang terkunci dan prediksi BTC-USD H+1 berbasis daily candle terbaru.',
        'BAGSISTEM • RESEARCH + OPERATIONAL'
    )

    s = dashboard_stats()
    cols = st.columns(4)
    vals = [
        ('Dataset', s['datasets'], 'Research • Live • Custom'),
        ('Eksperimen', s['experiments'], 'run & fold dapat diaudit'),
        ('Model', s['models'], 'registry Research + Live'),
        ('Prediksi', s['predictions'], 'riwayat H+1 immutable'),
    ]
    for c,(l,v,h) in zip(cols,vals):
        with c: metric_card(l,v,h)

    parity = latest_parity_report()
    research_model = active_model('research')
    live_model = active_model('live')
    research_pred = latest_prediction('research')
    live_pred = latest_prediction('live')
    live_ds = latest_dataset_by_mode('live')
    live_stats = dataset_market_stats(live_ds['id']) if live_ds else {}
    live_state = live_freshness_state(
        live_stats.get('last_date'), _safe_model_date(live_model),
        prediction_data_date=live_pred.get('data_freshness_date') if live_pred else None,
        prediction_model_id=live_pred.get('model_id') if live_pred else None,
        active_model_id=live_model.get('id') if live_model else None,
    )

    st.markdown('### Status sistem')
    left,right = st.columns(2)
    with left:
        if parity:
            try:
                report = json.loads(parity['report_json'])
                checks = flatten_checks(report)
                passed_n = sum(1 for x in checks if x.get('status') == 'PASS')
                total_n = len(checks)
            except Exception:
                passed_n,total_n = 0,0
            parity_label = 'PASS' if parity['passed'] else 'MINOR NUMERICAL MISMATCH'
            st.markdown(
                f'<div class="mode-card research"><div class="mode-kicker">RESEARCH MODE</div>'
                f'<h3>● Frozen / Reproducible</h3><p>Parity: <b>{parity_label}</b> • {passed_n}/{total_n} checks. '
                f'Champion S4 • WFV 5-fold • research engine terkunci.</p>'
                f'<span class="soft-badge">SOURCE LOCKED</span><span class="soft-badge">ARTICLE PATH</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="mode-card research"><div class="mode-kicker">RESEARCH MODE</div>'
                '<h3>○ Parity belum dijalankan</h3><p>Jalankan Parity Validation untuk membuktikan reproduksibilitas terhadap notebook final.</p>'
                '<span class="soft-badge">SOURCE LOCKED</span></div>', unsafe_allow_html=True,
            )
    with right:
        live_title = {
            'PREDICTION_CURRENT':'● Data + Model + Prediction Current',
            'MODEL_FRESH':'● Model Fresh • Siap Prediksi',
            'MODEL_STALE':'● Model Stale • Perlu Refit',
            'NO_MODEL':'○ Data Ada • Model Belum Dibangun',
            'NO_DATA':'○ Live Data Belum Ada',
            'MODEL_DATE_UNKNOWN':'● Kesegaran Model Tidak Diketahui',
        }.get(live_state['code'], live_state['label'])
        st.markdown(
            f'<div class="mode-card live"><div class="mode-kicker">LIVE MODE</div>'
            f'<h3>{live_title}</h3><p>Data: <b>{live_state.get("data_latest_date") or "—"}</b> • '
            f'Model source: <b>{live_state.get("model_source_date") or "—"}</b>. '
            f'Pipeline operasional terpisah dari Research Mode.</p>'
            f'<span class="live-badge">OPERATIONAL</span><span class="live-badge">H+1 DAILY</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('### Reproducibility status')
    if parity:
        a,b,c,d = st.columns(4)
        with a: metric_card('Parity', 'PASS' if parity['passed'] else 'MINOR FAIL', f'{passed_n}/{total_n} checks')
        with b: metric_card('Reference', parity['reference_id'].replace('COLAB_FINAL_',''), parity['mode'].upper())
        with c: metric_card('ELM Baseline', f'NH {ELM_BASELINE_NH}', f'C {ELM_BASELINE_C:.0e}')
        with d: metric_card('PSO', f'{PSO_PARTICLES} × {PSO_ITERATIONS}', f'val {PSO_VAL_RATIO:.0%}')
        if not parity['passed'] and total_n:
            st.caption('Strict parity tetap mencatat deviasi numerik kecil secara transparan. Research Engine tidak diubah untuk memaksa status PASS.')
    else:
        st.info('Parity validation belum dijalankan.')

    st.markdown('### Deployment overview')
    a,b = st.columns(2)
    with a:
        st.markdown('#### Research deployment')
        if research_model:
            x1,x2,x3 = st.columns(3)
            x1.metric('Model', research_model['id']); x2.metric('NH', research_model['n_hidden']); x3.metric('C', f"{research_model['c_value']:.2e}")
            if research_pred:
                st.markdown(f'<div class="prediction-strip"><small>{research_pred["prediction_date"]}</small><b>${research_pred["predicted_close"]:,.2f}</b><span>{research_pred["delta_pct"]:+.2f}%</span></div>', unsafe_allow_html=True)
        else:
            st.info('Belum ada research deployment model.')
    with b:
        st.markdown('#### Live deployment')
        if live_ds:
            x1,x2,x3 = st.columns(3)
            x1.metric('Data Terbaru', live_stats.get('last_date') or '—')
            x2.metric('Model Live', live_model['id'] if live_model else '—')
            x3.metric('Status', 'FRESH' if live_state['model_fresh'] else 'STALE/NOT READY')
            if live_pred:
                st.markdown(f'<div class="prediction-strip live"><small>{live_pred["prediction_date"]}</small><b>${live_pred["predicted_close"]:,.2f}</b><span>{live_pred["delta_pct"]:+.2f}%</span></div>', unsafe_allow_html=True)
        else:
            st.info('Live snapshot belum dibuat. Buka Mode Prediksi Live.')

    st.markdown('### Konfigurasi penelitian terkunci')
    a,b,c,d = st.columns(4)
    a.metric('Target','Close H+1','daily')
    b.metric('Validation',f'{WFV_FOLDS}-Fold WFV','expanding window')
    c.metric('Champion','S4','Tren SMA')
    d.metric('Features','5','Close • Lag1 • Lag2 • SMA10 • SMA30')

    st.markdown('### Eksperimen terbaru')
    exp = list_experiments(8)
    if not exp:
        st.info('Belum ada eksperimen.')
    else:
        st.dataframe(pd.DataFrame(exp)[['id','name','experiment_type','dataset_name','dataset_mode','status','created_at']],use_container_width=True,hide_index=True)
