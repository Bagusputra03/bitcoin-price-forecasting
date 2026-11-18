import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    latest_dataset_by_mode, dataset_market_stats, load_market_data,
    active_model, latest_prediction, latest_parity_report,
)
from research_engine.config import LIVE_MODEL_MODE, RESEARCH_MODEL_MODE
from services.research_results_service import get_champion_results
from services.live_service import get_live_state


def _chart(raw, pred=None):
    recent = raw.tail(180) if raw is not None and not raw.empty else raw
    fig = go.Figure()
    if recent is not None and not recent.empty:
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent['Close'], mode='lines', name='Harga Penutupan',
            line=dict(width=2.2, color='#F2A900')
        ))
    if pred:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(pred['prediction_date'])], y=[pred['predicted_close']],
            mode='markers', name='Prediksi H+1', marker=dict(size=12, color='#111111', symbol='diamond')
        ))
    fig.update_layout(
        height=405, margin=dict(l=8,r=8,t=45,b=8),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified', title='Pergerakan Bitcoin Terbaru', yaxis_title='USD',
        legend=dict(orientation='h', y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(120,105,80,.12)')
    return fig


def render():
    hero(
        'Dashboard Prediksi Bitcoin',
        'Ringkasan harga BTC-USD, hasil model terbaik, dan prediksi penutupan satu hari ke depan dalam satu tampilan.',
        'BITCOIN FORECAST INTELLIGENCE'
    )

    live_ds = latest_dataset_by_mode(LIVE_MODEL_MODE)
    raw = load_market_data(live_ds['id']) if live_ds else pd.DataFrame()
    stats = dataset_market_stats(live_ds['id']) if live_ds else {}
    live_model = active_model(LIVE_MODEL_MODE)
    live_pred = latest_prediction(LIVE_MODEL_MODE)
    research_pred = latest_prediction(RESEARCH_MODEL_MODE)
    state, _, _, _ = get_live_state(live_ds, raw, live_model)
    research_model = active_model(RESEARCH_MODEL_MODE)
    champion, _, _ = get_champion_results()

    last_close = stats.get('last_close')
    current_pred = live_pred or research_pred
    champion_mape = None
    champion_scenario = None
    if champion:
        champion_mape = champion.get('mape')
        champion_scenario = champion.get('scenario')
    if champion_mape is None and research_model:
        champion_mape = research_model.get('mape')
    if not champion_scenario and research_model:
        champion_scenario = research_model.get('scenario')
    champion_short = (champion_scenario or 'Belum tersedia').split('(')[0].strip()
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        metric_card('Harga BTC Terbaru', f'${last_close:,.2f}' if last_close is not None else '—', stats.get('last_date') or 'data belum diperbarui')
    with c2:
        metric_card('Prediksi H+1', f'${current_pred["predicted_close"]:,.2f}' if current_pred else '—', current_pred['prediction_date'] if current_pred else 'belum ada prediksi')
    with c3:
        metric_card('Model Terbaik', f'PSO–ELM {champion_short}' if champion_scenario else '—', 'hasil tersimpan di database')
    with c4:
        metric_card('MAPE Evaluasi', f'{float(champion_mape):.4f}%' if champion_mape is not None else '—', 'Walk-Forward 5-Fold')

    st.markdown('### Status Sistem')
    a,b = st.columns([1,1.35])
    with a:
        parity = latest_parity_report()
        research_ok = bool(parity)
        st.markdown(
            f'<div class="status-row"><i class="status-dot"></i><b>Model penelitian</b><span>PSO–ELM S4</span></div>'
            f'<div class="status-row"><i class="status-dot{("" if research_ok else " warn")}"></i><b>Validasi penelitian</b><span>{"Terverifikasi" if research_ok else "Belum tersedia"}</span></div>'
            f'<div class="status-row"><i class="status-dot{("" if state.get("model_fresh") else " warn")}"></i><b>Data & model live</b><span>{"Siap digunakan" if state.get("model_fresh") else "Perlu diperbarui"}</span></div>',
            unsafe_allow_html=True,
        )
        
        st.markdown(
            '<div class="panel"><b>Ringkasan Metode</b><p style="font-size:11px;color:#766F64;line-height:1.65;margin-bottom:0">'
            'Model menggunakan PSO untuk memilih hyperparameter ELM, kemudian diuji dengan Walk-Forward Validation 5-fold. '
            'Skenario S4 (Close, Lag 1, Lag 2, SMA 10, SMA 30) memberikan hasil terbaik dan digunakan untuk prediksi H+1.</p></div>',
            unsafe_allow_html=True,
        )
    with b:
        if raw is not None and not raw.empty:
            st.plotly_chart(_chart(raw, live_pred), use_container_width=True)
        else:
            st.info('Belum ada data live. Buka menu Prediksi Bitcoin untuk memperbarui data terbaru.')

    if live_pred:
        st.markdown('### Prediksi Terbaru')
        up = live_pred['delta_usd'] >= 0
        st.markdown(
            f'<div class="prediction-hero"><small>Prediksi harga penutupan • {live_pred["prediction_date"]}</small>'
            f'<div class="price">${live_pred["predicted_close"]:,.2f}</div>'
            f'<div class="{"delta-up" if up else "delta-down"}">{"▲" if up else "▼"} {live_pred["delta_pct"]:+.2f}% '
            f'({live_pred["delta_usd"]:+,.2f} USD)</div></div>',
            unsafe_allow_html=True,
        )
