import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui.theme import hero, metric_card
from services.repository import list_datasets, load_market_data
from research_engine.preprocessing import prepare_market_data, prepare_inference_features
from research_engine.config import SCENARIOS


def render():
    hero(
        'Preprocessing & Feature Engineering',
        'Pipeline transparan dari OHLCV hingga fitur S1–S5. Research target H+1 dan inference feature rows ditampilkan terpisah.',
        'PIPELINE INSPECTOR • FEATURE AUDIT',
    )
    ds = list_datasets()
    if not ds:
        st.info('Tambahkan dataset terlebih dahulu.'); return

    labels = {f"{d['id']} — {d['name']} [{(d.get('mode') or 'research').upper()}]": d for d in ds}
    d = labels[st.selectbox('Dataset', list(labels), key='prep_dataset')]
    raw = load_market_data(d['id'])
    if raw.empty:
        st.warning('Snapshot market_data kosong.'); return

    processed = prepare_market_data(raw)
    infer = prepare_inference_features(raw)
    steps = ['OHLCV','Interpolasi','Lag 1 & 2','RSI • MACD • SMA • BB','Target H+1','Drop NaN','Min-Max']
    cols = st.columns(len(steps))
    for c,step in zip(cols,steps): c.markdown(f'<div class="flow-step">{step}</div>',unsafe_allow_html=True)

    st.write('')
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card('Raw Samples',f'{len(raw):,}',(d.get('mode') or 'research').upper())
    with c2: metric_card('Labeled Samples',f'{len(processed):,}','Target tersedia')
    with c3: metric_card('Inference Rows',f'{len(infer):,}','latest row retained')
    with c4: metric_card('Feature Columns',len(processed.columns)-1,'excluding Target')

    tabs = st.tabs(['Processed Data','Latest Inference','Skenario Fitur','WFV Structure','Export'])
    with tabs[0]:
        st.dataframe(processed.tail(40),use_container_width=True)
    with tabs[1]:
        row = infer.tail(1).T.reset_index(); row.columns=['Feature','Latest Value']
        st.dataframe(row,use_container_width=True,hide_index=True)
        st.caption(f'Latest inference date: {infer.index.max().strftime("%Y-%m-%d")}')
    with tabs[2]:
        rows=[{'Scenario':name,'Dimensi':len(features),'Features':' + '.join(features)} for name,features in SCENARIOS.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    with tabs[3]:
        N=len(processed); seg=N//6; rows=[]
        for k in range(1,6):
            te=k*seg; tt=te+seg if k<5 else N
            rows.append({'Fold':k,'Train size':te,'Test size':tt-te,'Train mode':'Expanding window'})
        st.dataframe(rows,use_container_width=True,hide_index=True)
        fig=go.Figure()
        for r in rows:
            fig.add_trace(go.Bar(y=[f"Fold {r['Fold']}"],x=[r['Train size']],orientation='h',name='Train',showlegend=r['Fold']==1))
            fig.add_trace(go.Bar(y=[f"Fold {r['Fold']}"],x=[r['Test size']],orientation='h',base=[r['Train size']],name='Test',showlegend=r['Fold']==1))
        fig.update_layout(barmode='stack',height=330,margin=dict(l=8,r=8,t=30,b=8),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',title='Expanding Walk-Forward Validation')
        st.plotly_chart(fig,use_container_width=True)
    with tabs[4]:
        a,b=st.columns(2)
        with a:
            st.download_button('Processed Labeled CSV',processed.reset_index().to_csv(index=False).encode('utf-8'),f"{d['id']}_processed.csv",'text/csv',use_container_width=True,key=f"prep_export_{d['id']}")
        with b:
            st.download_button('Inference Features CSV',infer.reset_index().to_csv(index=False).encode('utf-8'),f"{d['id']}_inference.csv",'text/csv',use_container_width=True,key=f"infer_export_{d['id']}")
