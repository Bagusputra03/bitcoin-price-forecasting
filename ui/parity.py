import json
import pandas as pd
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    list_datasets, load_market_data, save_parity_report, list_parity_reports,
    get_parity_report, delete_parity_report, create_dataset, save_market_data,
)
from research_engine.data_loader import download_yfinance
from research_engine.preprocessing import prepare_market_data
from research_engine.parity import structural_parity, run_full_parity, flatten_checks
from research_engine.reference_contract import (
    REFERENCE_ID, REFERENCE_LABEL, DATA_REFERENCE, WFV_REFERENCE,
    SENSITIVITY_REFERENCE, POC_REFERENCE, TOURNAMENT_REFERENCE,
    TOURNAMENT_RANKING_REFERENCE, CHAMPION_REFERENCE,
)
from research_engine.config import RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, RESEARCH_INTERVAL


def _dataset_select():
    ds = [d for d in list_datasets() if (d.get('mode') or 'research') != 'live']
    if not ds:
        return None
    labels = {f"{d['id']} — {d['name']}": d for d in ds}
    return labels[st.selectbox('Dataset untuk parity', list(labels), key='parity_dataset')]


def _status_badge(passed):
    if passed:
        return '<span style="display:inline-flex;align-items:center;gap:7px;background:#ECFDF3;color:#027A48;border:1px solid #ABEFC6;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:800">● PASS</span>'
    return '<span style="display:inline-flex;align-items:center;gap:7px;background:#FEF3F2;color:#B42318;border:1px solid #FECDCA;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:800">● FAIL</span>'


def _checks_df(checks):
    if not checks:
        return pd.DataFrame(columns=['name','actual','expected','status','note'])
    df = pd.DataFrame(checks)
    cols = [c for c in ['name','actual','expected','status','note'] if c in df.columns]
    return df[cols]


def _render_section(title, payload):
    passed = bool(payload.get('passed'))
    st.markdown(f"### {title} &nbsp; {_status_badge(passed)}", unsafe_allow_html=True)
    checks = _checks_df(payload.get('checks', []))
    if not checks.empty:
        st.dataframe(checks, use_container_width=True, hide_index=True)
    return checks


def _render_report(report, report_id=None, key_prefix='parity_report'):
    passed = bool(report.get('passed'))
    total_checks = len(flatten_checks(report))
    pass_checks = sum(1 for r in flatten_checks(report) if r.get('status') == 'PASS')

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card('Overall', 'PASS' if passed else 'FAIL', 'Colab ↔ Streamlit')
    with c2: metric_card('Checks', f'{pass_checks}/{total_checks}', 'pemeriksaan lulus')
    with c3: metric_card('Reference', REFERENCE_ID.replace('COLAB_FINAL_',''), 'contract terkunci')
    with c4: metric_card('Status', 'ABORTED' if report.get('aborted') else 'COMPLETE', 'numerical parity')

    if passed:
        st.success('Parity validation lulus. Pipeline sistem mereproduksi output notebook final pada presisi yang dipublikasikan.')
    elif report.get('aborted'):
        st.error(report.get('reason','Structural parity gagal.'))
    else:
        st.warning('Ada satu atau lebih nilai yang tidak sama dengan output notebook final. Buka setiap bagian untuk melihat titik mismatch.')

    sections = report.get('sections', {})
    tabs = st.tabs(['Data & WFV','Sensitivity','POC','Tournament','Champion S4'])
    keys = ['structure','sensitivity','poc','tournament','champion']
    titles = ['Data & Walk-Forward Validation','Sensitivity ELM','Proof of Concept S3','Tournament S1–S5','Champion PSO–ELM S4']
    for tab,key,title in zip(tabs,keys,titles):
        with tab:
            if key not in sections:
                st.info('Bagian ini tidak dijalankan karena structural parity belum lulus.')
            else:
                _render_section(title, sections[key])

    flat = pd.DataFrame(flatten_checks(report))
    if not flat.empty:
        st.download_button(
            'Export parity checks CSV',
            flat.to_csv(index=False).encode('utf-8'),
            f'parity_{report_id or "latest"}.csv', 'text/csv', use_container_width=True,
            key=f'{key_prefix}_csv_{report_id or "latest"}'
        )
        st.download_button(
            'Export parity report JSON',
            json.dumps(report, indent=2, default=str).encode('utf-8'),
            f'parity_{report_id or "latest"}.json', 'application/json', use_container_width=True,
            key=f'{key_prefix}_json_{report_id or "latest"}'
        )


def render():
    hero(
        'Parity Validation — Colab ↔ Sistem',
        'Validasi reproduksibilitas menggunakan angka output notebook final sebagai reference contract. Struktur harus exact; metrik numerik harus sama pada presisi yang dicetak artikel/notebook.',
        'REPRODUCIBILITY • SOURCE OF TRUTH'
    )
    st.markdown(
        '<div class="callout source"><b>Reference contract terkunci</b><br>'
        'Dataset BTC-USD 2021-01-01 s.d. 2026-04-01, 1.886 sampel processed, WFV 5-fold, '
        'ELM baseline NH=10/C=1e2, PSO N=20/I=50/val=25%, dan champion S4 seed=99. '
        'Angka pembanding diambil dari output notebook yang Anda kirim.</div>',
        unsafe_allow_html=True
    )
    st.caption('Catatan metodologis: notebook tidak menyimpan seluruh 1.886 baris OHLCV sebagai file CSV. Karena itu snapshot exact direkonstruksi melalui Yahoo Finance dengan periode yang sama. Jika provider merevisi historical data, structural/numerical parity akan FAIL dan sistem tidak memaksakan kelulusan.')

    tab_run, tab_reference, tab_history = st.tabs(['Run Parity','Reference Contract','Riwayat'])

    with tab_run:
        d = _dataset_select()
        if d is None:
            st.info('Belum ada dataset. Buat snapshot penelitian langsung dari sini.')
            if st.button('Buat Snapshot Penelitian Final', type='primary', use_container_width=True):
                try:
                    with st.spinner('Mengambil BTC-USD sesuai periode source...'):
                        raw = download_yfinance(RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, RESEARCH_INTERVAL)
                        did = create_dataset('Dataset Penelitian Final', RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, RESEARCH_INTERVAL,
                                             'Snapshot terkunci untuk parity validation.', True)
                        save_market_data(did, raw)
                    st.success(f'Snapshot {did} dibuat dengan {len(raw):,} baris raw.'); st.rerun()
                except Exception as e:
                    st.error(str(e))
            return

        raw = load_market_data(d['id'])
        if raw.empty:
            st.error('Dataset terpilih belum memiliki market data.'); return

        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card('Dataset', d['symbol'], d['id'])
        with c2: metric_card('Raw rows', f'{len(raw):,}', f"{d['start_date']} → {d['end_date']}")
        try:
            processed = prepare_market_data(raw)
            processed_count = len(processed)
        except Exception:
            processed_count = 0
        with c3: metric_card('Processed', f'{processed_count:,}', 'reference: 1,886')
        with c4: metric_card('Reference', 'FINAL', REFERENCE_ID)

        st.markdown('#### 1. Structural pre-check')
        if st.button('Cek Struktur & WFV', use_container_width=True):
            try:
                r = structural_parity(raw, d); r.pop('processed_df', None)
                report = {'reference_id': REFERENCE_ID, 'sections': {'structure': r}, 'passed': r['passed'], 'aborted': False}
                pid = save_parity_report(d['id'], REFERENCE_ID, 'structural', report)
                st.session_state['parity_report'] = report; st.session_state['parity_report_id'] = pid
            except Exception as e:
                st.error(str(e))

        st.markdown('#### 2. Full numerical parity')
        st.caption('Full parity menjalankan ulang sensitivity, POC, tournament, dan champion S4 dengan konfigurasi source. Ini adalah pengujian komputasi penuh, bukan pembacaan hasil tersimpan.')
        if st.button('Jalankan FULL PARITY', type='primary', use_container_width=True):
            bar = st.progress(0, text='Memulai parity...')
            def cb(v,t): bar.progress(min(max(float(v),0.0),1.0), text=t)
            try:
                report = run_full_parity(raw, d, cb)
                pid = save_parity_report(d['id'], REFERENCE_ID, 'full', report)
                st.session_state['parity_report'] = report; st.session_state['parity_report_id'] = pid
                bar.empty()
            except Exception as e:
                bar.empty(); st.exception(e)

        report = st.session_state.get('parity_report')
        if report:
            st.divider()
            _render_report(report, st.session_state.get('parity_report_id'), key_prefix='current')

    with tab_reference:
        st.markdown('### Dataset & WFV')
        st.json(DATA_REFERENCE)
        st.dataframe(pd.DataFrame(WFV_REFERENCE), use_container_width=True, hide_index=True)

        st.markdown('### Sensitivity reference')
        st.dataframe(pd.DataFrame(SENSITIVITY_REFERENCE), use_container_width=True, hide_index=True)

        st.markdown('### POC reference')
        st.json(POC_REFERENCE)

        st.markdown('### Tournament reference')
        tour = pd.DataFrame([{'Skenario':k, **v} for k,v in TOURNAMENT_REFERENCE.items()])
        st.dataframe(tour, use_container_width=True, hide_index=True)
        st.caption('Ranking: ' + ' → '.join(TOURNAMENT_RANKING_REFERENCE))

        st.markdown('### Champion S4 reference')
        st.json({k:v for k,v in CHAMPION_REFERENCE.items() if k != 'folds'})
        st.dataframe(pd.DataFrame(CHAMPION_REFERENCE['folds']), use_container_width=True, hide_index=True)

    with tab_history:
        rows = list_parity_reports()
        if not rows:
            st.info('Belum ada parity report tersimpan.')
        else:
            view = pd.DataFrame(rows)
            st.dataframe(view[['id','dataset_name','reference_id','mode','status','passed','created_at']], use_container_width=True, hide_index=True)
            labels = {f"{r['id']} — {r['mode']} — {'PASS' if r['passed'] else 'FAIL'}": r for r in rows}
            choice = st.selectbox('Buka report', list(labels), key='parity_history_choice')
            selected = labels[choice]
            loaded = get_parity_report(selected['id'])
            if loaded:
                _render_report(loaded.get('report',{}), selected['id'], key_prefix='history')
                if st.button('Hapus report ini', use_container_width=True):
                    delete_parity_report(selected['id']); st.success('Parity report dihapus.'); st.rerun()
