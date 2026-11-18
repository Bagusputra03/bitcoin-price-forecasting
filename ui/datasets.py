import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    create_dataset, list_datasets, update_dataset, delete_dataset,
    save_market_data, load_market_data, dataset_market_stats,
)
from research_engine.data_loader import download_yfinance
from research_engine.config import RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, RESEARCH_INTERVAL


def _plot_close(df):
    fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close', line=dict(width=2.2)))
    fig.update_layout(height=340, margin=dict(l=8,r=8,t=30,b=8), paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', yaxis_title='USD', hovermode='x unified')
    fig.update_xaxes(showgrid=False); fig.update_yaxes(gridcolor='rgba(148,163,184,.15)')
    return fig


def render():
    hero(
        'Dataset & Market Snapshots',
        'CRUD dataset dengan snapshot OHLCV tersimpan di SQLite. Research snapshot dapat dikunci, sedangkan Live snapshot dikelola otomatis dari Mode Prediksi Live.',
        'DATA LAYER • CRUD • SNAPSHOT'
    )

    items = list_datasets()
    research_n = sum(1 for d in items if (d.get('mode') or 'research') == 'research')
    live_n = sum(1 for d in items if d.get('mode') == 'live')
    custom_n = sum(1 for d in items if d.get('mode') == 'custom')
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card('Total Dataset', len(items), 'SQLite snapshots')
    with c2: metric_card('Research', research_n, 'frozen / reproducible')
    with c3: metric_card('Live', live_n, 'operational latest data')
    with c4: metric_card('Custom', custom_n, 'exploratory snapshots')

    tab_list, tab_create = st.tabs(['Dataset Registry', 'Create Snapshot'])

    with tab_create:
        st.markdown('### Buat snapshot baru')
        st.caption('Untuk data terbaru operasional, gunakan menu Live Prediction agar research dataset tidak tercampur.')
        with st.form('new_ds'):
            c1,c2 = st.columns([1.4,1])
            name = c1.text_input('Nama dataset', 'Dataset Penelitian Final')
            mode_label = c2.selectbox('Mode', ['Research', 'Custom'])
            c3,c4 = st.columns(2)
            symbol = c3.text_input('Symbol', RESEARCH_SYMBOL)
            interval = c4.selectbox('Interval', ['1d'], index=0)
            c5,c6 = st.columns(2)
            start = c5.text_input('Start (YYYY-MM-DD)', RESEARCH_START)
            end = c6.text_input('End exclusive (YYYY-MM-DD)', RESEARCH_END)
            desc = st.text_area('Deskripsi', 'Snapshot untuk reproduksi eksperimen penelitian.')
            locked_default = mode_label == 'Research'
            locked = st.checkbox('Kunci dataset dari penghapusan UI', value=locked_default)
            submit = st.form_submit_button('Download & Save Snapshot', type='primary', use_container_width=True)
        if submit:
            try:
                with st.status('Membuat snapshot...', expanded=True) as status:
                    df = download_yfinance(symbol, start, end, interval)
                    mode = 'research' if mode_label == 'Research' else 'custom'
                    did = create_dataset(name, symbol, start, end, interval, desc, locked, mode=mode)
                    save_market_data(did, df)
                    status.update(label=f'Snapshot {did} tersimpan', state='complete', expanded=False)
                st.success(f'{len(df):,} baris OHLCV berhasil disimpan.'); st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with tab_list:
        if not items:
            st.info('Belum ada dataset. Buat snapshot dari tab Create Snapshot.'); return

        f1,f2 = st.columns([1,2])
        mode_filter = f1.selectbox('Filter mode', ['Semua','Research','Live','Custom'])
        query = f2.text_input('Cari dataset', placeholder='ID, nama, atau symbol...')
        filtered = items
        if mode_filter != 'Semua':
            filtered = [d for d in filtered if (d.get('mode') or 'research') == mode_filter.lower()]
        if query.strip():
            q = query.lower().strip()
            filtered = [d for d in filtered if q in f"{d['id']} {d['name']} {d['symbol']}".lower()]
        if not filtered:
            st.info('Tidak ada dataset sesuai filter.'); return

        registry = pd.DataFrame(filtered)
        show_cols = ['id','name','mode','symbol','start_date','end_date','interval','is_locked','last_synced_at','created_at']
        st.dataframe(registry[[c for c in show_cols if c in registry.columns]], use_container_width=True, hide_index=True)

        labels = {f"{d['id']} — {d['name']} [{(d.get('mode') or 'research').upper()}]": d for d in filtered}
        d = labels[st.selectbox('Buka dataset', list(labels), key='dataset_registry_select')]
        stats = dataset_market_stats(d['id'])
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card('Rows', f"{stats.get('rows_count',0):,}", d['symbol'])
        with c2: metric_card('First Date', stats.get('first_date') or '—', 'snapshot range')
        with c3: metric_card('Latest Date', stats.get('last_date') or '—', 'available candle')
        last_close = stats.get('last_close')
        with c4: metric_card('Last Close', f"${last_close:,.2f}" if last_close is not None else '—', (d.get('mode') or 'research').upper())

        df = load_market_data(d['id'])
        tab_overview, tab_data, tab_manage = st.tabs(['Overview','Market Data','Manage'])
        with tab_overview:
            if not df.empty:
                st.plotly_chart(_plot_close(df), use_container_width=True)
            st.markdown(
                f'<div class="detail-grid"><div><small>Dataset ID</small><b>{d["id"]}</b></div>'
                f'<div><small>Mode</small><b>{(d.get("mode") or "research").upper()}</b></div>'
                f'<div><small>Source</small><b>{d.get("source") or "Yahoo Finance"}</b></div>'
                f'<div><small>Protection</small><b>{"LOCKED" if d["is_locked"] else "EDITABLE"}</b></div></div>',
                unsafe_allow_html=True,
            )
            if d.get('description'): st.caption(d['description'])

        with tab_data:
            if df.empty:
                st.warning('Snapshot belum memiliki market data.')
            else:
                st.dataframe(df.tail(50), use_container_width=True)
                st.download_button('Export Dataset CSV', df.reset_index().to_csv(index=False).encode('utf-8'),
                                   f"{d['id']}_{d['symbol']}.csv", 'text/csv', use_container_width=True,
                                   key=f"dataset_export_{d['id']}")

        with tab_manage:
            st.markdown('#### Update metadata')
            with st.form(f"edit_ds_{d['id']}"):
                n = st.text_input('Nama', d['name'])
                ds = st.text_area('Deskripsi', d.get('description') or '')
                if st.form_submit_button('Save Metadata', use_container_width=True):
                    update_dataset(d['id'], n, ds); st.success('Metadata diperbarui.'); st.rerun()

            st.markdown('#### Delete dataset')
            if d['is_locked']:
                st.info('Dataset ini LOCKED. Penghapusan dari UI dinonaktifkan untuk melindungi snapshot penelitian.')
            else:
                st.caption('Menghapus dataset akan menghapus market rows dan eksperimen terkait melalui foreign-key cascade.')
                confirm = st.checkbox(f"Saya yakin menghapus {d['id']}", key=f"delete_ds_confirm_{d['id']}")
                if st.button('Delete Dataset', use_container_width=True, disabled=not confirm, key=f"delete_ds_{d['id']}"):
                    delete_dataset(d['id']); st.success('Dataset dihapus.'); st.rerun()
