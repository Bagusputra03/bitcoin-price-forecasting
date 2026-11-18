import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    latest_dataset_by_mode, load_market_data, dataset_market_stats,
    active_model, latest_prediction, list_predictions,
)
from services.live_service import (
    refresh_live_data, get_live_state, train_live_model, create_live_prediction,
)
from research_engine.config import LIVE_MODEL_MODE, RESEARCH_MODEL_MODE
from services.research_results_service import get_champion_results


def _chart(raw, pred):
    recent=raw.tail(120)
    fig=go.Figure(go.Scatter(x=recent.index,y=recent['Close'],mode='lines',name='Harga Penutupan',line=dict(width=2.2,color='#F2A900')))
    if pred:
        fig.add_trace(go.Scatter(x=[pd.Timestamp(pred['prediction_date'])],y=[pred['predicted_close']],mode='markers',name='Prediksi H+1',marker=dict(size=12,color='#111111',symbol='diamond')))
    fig.update_layout(height=390,margin=dict(l=8,r=8,t=45,b=8),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',title='Harga Bitcoin dan Titik Prediksi Terbaru',yaxis_title='USD',hovermode='x unified',legend=dict(orientation='h',y=1.08,x=0))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='rgba(120,105,80,.12)')
    return fig


def _result_block(pred):
    if not pred: return
    up=pred['delta_usd']>=0
    a,b,c=st.columns(3)
    with a: metric_card('Harga Terakhir',f"${pred['last_close']:,.2f}",pred['input_date'])
    with b: metric_card('Prediksi H+1',f"${pred['predicted_close']:,.2f}",pred['prediction_date'])
    with c: metric_card('Perubahan',f"{'▲' if up else '▼'} {pred['delta_pct']:+.2f}%",f"{pred['delta_usd']:+,.2f} USD")
    st.markdown(
        f'<div class="prediction-hero"><small>Prediksi harga penutupan Bitcoin • {pred["prediction_date"]}</small>'
        f'<div class="price">${pred["predicted_close"]:,.2f}</div>'
        f'<div class="{"delta-up" if up else "delta-down"}">{"▲" if up else "▼"} {pred["delta_pct"]:+.2f}% dari harga terakhir</div></div>',
        unsafe_allow_html=True,
    )


def render():
    hero(
        'Prediksi Harga Bitcoin',
        'Perbarui data harian terbaru dan hasilkan prediksi harga penutupan Bitcoin untuk satu hari berikutnya dengan model PSO–ELM S4.',
        'BITCOIN H+1 FORECAST'
    )
    st.markdown('<div class="callout source"><b>Prediksi harian</b><br>Data menggunakan interval 1 hari. Tombol di bawah akan memperbarui data, memperbarui model bila diperlukan, lalu menghasilkan prediksi H+1 secara otomatis.</div>',unsafe_allow_html=True)

    d=latest_dataset_by_mode(LIVE_MODEL_MODE)
    raw=load_market_data(d['id']) if d else pd.DataFrame()
    state,pred,model,_=get_live_state(d,raw,active_model(LIVE_MODEL_MODE))
    stats=dataset_market_stats(d['id']) if d else {}
    champion,_,_=get_champion_results()
    research_model=active_model(RESEARCH_MODEL_MODE)
    research_mape=(champion or {}).get('mape')
    if research_mape is None and research_model:
        research_mape=research_model.get('mape')

    s1,s2,s3=st.columns(3)
    with s1: metric_card('Data Terbaru',stats.get('last_date') or '—','BTC-USD daily')
    with s2: metric_card('Status Model','Siap' if state.get('model_fresh') else 'Perlu Diperbarui','PSO–ELM S4')
    with s3: metric_card('MAPE Penelitian',f'{float(research_mape):.4f}%' if research_mape is not None else '—','hasil evaluasi tersimpan')

    if st.button('Perbarui Data & Prediksi',type='primary',use_container_width=True,key='simple_live_predict'):
        try:
            with st.status('Menyiapkan prediksi terbaru...',expanded=True) as status:
                st.write('1/3 Memperbarui data BTC-USD...')
                d,raw=refresh_live_data()
                latest=pd.Timestamp(raw.index.max()).strftime('%Y-%m-%d')
                model=active_model(LIVE_MODEL_MODE)
                current_state,_,_,_=get_live_state(d,raw,model)
                if not current_state.get('model_fresh'):
                    st.write('2/3 Menyesuaikan model dengan data terbaru...')
                    model,summary=train_live_model(d,raw)
                    st.write(f'Model siap • NH={summary.n_hidden}, C={summary.C:.3e}')
                else:
                    st.write('2/3 Model masih sesuai dengan data terbaru.')
                current_pred=latest_prediction(LIVE_MODEL_MODE)
                current_state,_,_,_=get_live_state(d,raw,model)
                if current_state.get('prediction_current') and current_pred:
                    st.write('3/3 Prediksi untuk snapshot terbaru sudah tersedia.')
                    result=current_pred
                else:
                    st.write('3/3 Menghitung prediksi H+1...')
                    result=create_live_prediction(d,raw,model)
                st.session_state['simple_live_result_id']=result['id']
                status.update(label=f'Prediksi terbaru siap • data {latest}',state='complete',expanded=False)
            st.success('Prediksi berhasil diperbarui.'); st.rerun()
        except Exception as exc:
            st.error(f'Proses prediksi belum dapat diselesaikan: {exc}')

    # reload after possible update
    d=latest_dataset_by_mode(LIVE_MODEL_MODE)
    raw=load_market_data(d['id']) if d else pd.DataFrame()
    pred=latest_prediction(LIVE_MODEL_MODE)
    model=active_model(LIVE_MODEL_MODE)
    state,_,_,_=get_live_state(d,raw,model)
    if pred:
        st.markdown('### Hasil Prediksi Terbaru')
        _result_block(pred)
        if not raw.empty:
            st.plotly_chart(_chart(raw,pred),use_container_width=True)
        with st.expander('Lihat detail model dan fitur input'):
            if model:
                st.markdown(f'**Model:** PSO–ELM S4  \n**Hidden Neuron:** {model["n_hidden"]}  \n**C:** {model["c_value"]:.4g}  \n**Data model sampai:** {model.get("source_latest_date") or model.get("trained_until") or "—"}')
            import json
            obj=json.loads(pred.get('feature_json') or '{}')
            snap=pd.DataFrame({'Fitur':list(obj.keys()),'Nilai':list(obj.values())})
            if not snap.empty: st.dataframe(snap,use_container_width=True,hide_index=True)
    else:
        st.info('Belum ada prediksi live. Klik “Perbarui Data & Prediksi”.')

    with st.expander('Prediksi berdasarkan dataset penelitian'):
        rp=latest_prediction(RESEARCH_MODEL_MODE)
        if rp:
            a,b,c=st.columns(3)
            a.metric('Harga Terakhir',f"${rp['last_close']:,.2f}",rp['input_date'])
            b.metric('Prediksi H+1',f"${rp['predicted_close']:,.2f}",rp['prediction_date'])
            c.metric('Perubahan',f"{rp['delta_pct']:+.2f}%")
            st.caption('Bagian ini menggunakan deployment model dari dataset penelitian yang telah dikunci.')
        else:
            st.caption('Belum ada prediksi tersimpan dari dataset penelitian.')

    rows=list_predictions(limit=5,mode=LIVE_MODEL_MODE)
    if rows:
        st.markdown('### Prediksi Sebelumnya')
        df=pd.DataFrame(rows)
        if 'source_type' not in df.columns:
            df['source_type'] = 'operational'
        df['source_label'] = df['source_type'].map({
            'historical_replay': 'Replay Historis',
            'operational': 'Prediksi Aktual',
        }).fillna(df['source_type'])
        show=df[['prediction_date','last_close','predicted_close','delta_pct','source_label','created_at']].copy()
        show.columns=['Tanggal Prediksi','Harga Terakhir','Prediksi H+1','Perubahan (%)','Sumber','Dibuat']
        st.dataframe(show,use_container_width=True,hide_index=True)
