import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    list_datasets, latest_dataset_by_mode, create_dataset, save_market_data,
    load_market_data, dataset_market_stats, update_dataset, delete_dataset,
)
from research_engine.data_loader import download_yfinance
from research_engine.preprocessing import prepare_market_data, prepare_inference_features
from research_engine.config import RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, RESEARCH_INTERVAL, SCENARIOS


def _price_chart(df):
    fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='BTC Close', line=dict(width=2.2, color='#F2A900')))
    fig.update_layout(height=350, margin=dict(l=8,r=8,t=38,b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title='USD', hovermode='x unified')
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='rgba(120,105,80,.12)')
    return fig


def render():
    hero(
        'Data & Proses Prediksi',
        'Lihat sumber data Bitcoin, tahapan pengolahan, fitur yang digunakan, serta kelola snapshot data dari satu halaman.',
        'DATA • PREPROCESSING • FEATURE ENGINEERING'
    )

    research_ds = latest_dataset_by_mode('research')
    live_ds = latest_dataset_by_mode('live')
    rstats = dataset_market_stats(research_ds['id']) if research_ds else {}
    lstats = dataset_market_stats(live_ds['id']) if live_ds else {}
    a,b,c,d = st.columns(4)
    with a: metric_card('Sumber Data', 'Yahoo Finance', RESEARCH_SYMBOL)
    with b: metric_card('Data Penelitian', f"{rstats.get('rows_count',0):,}", f"sampai {rstats.get('last_date') or '—'}")
    with c: metric_card('Data Live', f"{lstats.get('rows_count',0):,}", f"sampai {lstats.get('last_date') or '—'}")
    with d: metric_card('Interval', '1 Hari', 'prediksi Close H+1')

    st.markdown('### Alur Sistem')
    steps = ['Data BTC-USD','Pembersihan Data','Fitur Teknikal','Normalisasi','Optimasi PSO–ELM','Prediksi H+1']
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        col.markdown(f'<div class="flow-step">{step}</div>', unsafe_allow_html=True)

    tab_overview, tab_process, tab_manage = st.tabs(['Ringkasan Data','Proses & Fitur','Kelola Dataset'])

    with tab_overview:
        datasets = list_datasets()
        if not datasets:
            st.info('Belum ada dataset tersimpan.')
        else:
            labels = {f"{d['name']} • {(d.get('mode') or 'research').title()}": d for d in datasets}
            dset = labels[st.selectbox('Pilih dataset', list(labels), key='simple_dataset_view')]
            raw = load_market_data(dset['id'])
            stats = dataset_market_stats(dset['id'])
            x1,x2,x3,x4 = st.columns(4)
            x1.metric('Jumlah Data', f"{stats.get('rows_count',0):,}")
            x2.metric('Tanggal Awal', stats.get('first_date') or '—')
            x3.metric('Tanggal Terakhir', stats.get('last_date') or '—')
            lc = stats.get('last_close'); x4.metric('Close Terakhir', f'${lc:,.2f}' if lc is not None else '—')
            if not raw.empty:
                st.plotly_chart(_price_chart(raw), use_container_width=True)
                st.dataframe(raw.tail(20), use_container_width=True)
                st.download_button('Unduh Data CSV', raw.reset_index().to_csv(index=False).encode('utf-8'), f"{dset['id']}_btc.csv", 'text/csv', use_container_width=True, key=f"simple_data_export_{dset['id']}")

    with tab_process:
        source = research_ds or live_ds
        if not source:
            st.info('Tambahkan dataset terlebih dahulu untuk melihat hasil preprocessing.')
        else:
            raw = load_market_data(source['id'])
            if raw.empty:
                st.warning('Dataset belum berisi data.')
            else:
                processed = prepare_market_data(raw)
                infer = prepare_inference_features(raw)
                p1,p2,p3 = st.columns(3)
                with p1: metric_card('Data Mentah', f'{len(raw):,}', 'OHLCV')
                with p2: metric_card('Data Siap Uji', f'{len(processed):,}', 'Target H+1 tersedia')
                with p3: metric_card('Fitur Champion', '5', 'Close • Lag • SMA')
                st.markdown('#### Skenario fitur penelitian')
                rows = [{'Skenario': k.split('(')[0].strip(), 'Nama': k[k.find('(')+1:k.find(')')], 'Dimensi': len(v), 'Fitur': ' + '.join(v)} for k,v in SCENARIOS.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.markdown('<div class="callout source"><b>Skenario terbaik: S4 — Tren SMA</b><br>Fitur yang digunakan untuk model prediksi: Close, Lag_1, Lag_2, SMA_10, dan SMA_30.</div>', unsafe_allow_html=True)
                with st.expander('Lihat detail proses teknis'):
                    st.markdown('''
**Tahapan preprocessing**
1. Memuat OHLCV harian BTC-USD.
2. Interpolasi linear dan pembersihan nilai kosong.
3. Membentuk Lag_1 dan Lag_2.
4. Menghitung RSI 14, MACD, SMA 10, SMA 30, dan Bollinger Bands.
5. Membentuk target `Close(t+1)` untuk evaluasi.
6. Menyelaraskan seluruh skenario pada rentang temporal yang sama.
7. Normalisasi Min-Max sebelum pengujian model.

**Evaluasi** menggunakan Walk-Forward Validation 5-fold dengan expanding window.
''')
                    latest = infer.tail(1).T.reset_index(); latest.columns=['Fitur','Nilai Terbaru']
                    st.dataframe(latest, use_container_width=True, hide_index=True)

    with tab_manage:
        st.markdown('#### Tambah snapshot data')
        with st.form('simple_add_dataset'):
            c1,c2 = st.columns(2)
            name = c1.text_input('Nama dataset', 'Dataset Penelitian Final')
            mode_label = c2.selectbox('Jenis dataset', ['Research','Custom'])
            c3,c4 = st.columns(2)
            start = c3.text_input('Tanggal awal', RESEARCH_START)
            end = c4.text_input('Tanggal akhir (exclusive)', RESEARCH_END)
            desc = st.text_area('Catatan', 'Snapshot BTC-USD untuk pengujian dan dokumentasi.')
            submit = st.form_submit_button('Simpan Dataset', type='primary', use_container_width=True)
        if submit:
            try:
                with st.status('Mengunduh dan menyimpan data...', expanded=False):
                    df = download_yfinance(RESEARCH_SYMBOL, start, end, RESEARCH_INTERVAL)
                    mode = mode_label.lower()
                    did = create_dataset(name, RESEARCH_SYMBOL, start, end, RESEARCH_INTERVAL, desc, mode=='research', mode=mode)
                    save_market_data(did, df)
                st.success(f'Dataset {did} berhasil disimpan.'); st.rerun()
            except Exception as exc:
                st.error(str(exc))

        datasets = list_datasets()
        if datasets:
            st.markdown('#### Ubah atau hapus dataset')
            labels = {f"{d['id']} — {d['name']}": d for d in datasets}
            dset = labels[st.selectbox('Pilih dataset', list(labels), key='simple_manage_dataset')]
            with st.form(f'simple_edit_{dset["id"]}'):
                new_name = st.text_input('Nama', dset['name'])
                new_desc = st.text_area('Catatan', dset.get('description') or '')
                if st.form_submit_button('Simpan Perubahan', use_container_width=True):
                    update_dataset(dset['id'], new_name, new_desc); st.success('Perubahan disimpan.'); st.rerun()
            if dset.get('is_locked'):
                st.info('Dataset penelitian dikunci untuk menjaga konsistensi hasil pengujian.')
            else:
                confirm = st.checkbox('Saya yakin ingin menghapus dataset ini', key=f'simple_delete_confirm_{dset["id"]}')
                if st.button('Hapus Dataset', disabled=not confirm, use_container_width=True, key=f'simple_delete_{dset["id"]}'):
                    delete_dataset(dset['id']); st.success('Dataset dihapus.'); st.rerun()
