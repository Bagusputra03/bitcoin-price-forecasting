import json
from datetime import datetime
import pandas as pd
import streamlit as st

from ui.theme import hero, metric_card, APP_NAME
from services.repository import (
    list_predictions, delete_prediction, list_experiments, update_experiment,
    delete_experiment, experiment_runs, experiment_folds, list_datasets,
    list_models, list_activity_logs,
)
from services.research_results_service import get_poc_results, get_tournament_results, get_champion_results
from research_engine.config import RESEARCH_SYMBOL, RESEARCH_START, RESEARCH_END, WFV_FOLDS, PSO_PARTICLES, PSO_ITERATIONS


def _html_table(df):
    if df is None or df.empty:
        return '<p><em>Belum ada data.</em></p>'
    return df.to_html(index=False, border=0, classes='tbl')


def _report_html():
    predictions = pd.DataFrame(list_predictions())
    experiments = pd.DataFrame(list_experiments())
    poc, _ = get_poc_results()
    tournament, _ = get_tournament_results()
    champion, _, _ = get_champion_results()
    now = datetime.now().strftime('%d %B %Y %H:%M')

    champion_label = '—'
    champion_mape = '—'
    champion_mae = '—'
    champion_rmse = '—'
    if champion:
        scenario = str(champion.get('scenario') or 'S4 (Tren SMA)')
        champion_label = 'PSO-ELM ' + scenario.split('(')[0].strip()
        champion_mape = f"{float(champion['mape']):.4f}%"
        champion_mae = f"${float(champion['mae']):,.2f}"
        champion_rmse = f"${float(champion['rmse']):,.2f}"

    if poc:
        poc_text = (
            f"ELM Baseline MAPE: <b>{float(poc['elm_mape']):.4f}%</b><br>"
            f"PSO-ELM MAPE: <b>{float(poc['pso_mape']):.4f}%</b><br>"
            f"Peningkatan MAPE: <b>{float(poc['delta_mape']):.2f}%</b>"
        )
    else:
        poc_text = '<em>Hasil perbandingan model belum tersedia di database.</em>'

    if tournament is not None and not tournament.empty:
        ranking = ' → '.join([
            str(x).split('(')[0].strip()
            for x in tournament.sort_values('PSO MAPE')['Skenario']
        ])
    else:
        ranking = '—'

    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>{APP_NAME} Report</title>
<style>
body{{font-family:Arial,sans-serif;color:#181818;margin:40px;line-height:1.55}}h1{{font-size:28px;margin-bottom:4px}}h2{{margin-top:30px;border-bottom:2px solid #F2A900;padding-bottom:7px}}.gold{{color:#A77700}}.kpi{{display:inline-block;border:1px solid #ded7ca;border-top:3px solid #F2A900;border-radius:10px;padding:12px 15px;margin:4px 7px 4px 0;min-width:150px}}.kpi small{{display:block;color:#777}}.kpi b{{font-size:19px}}.note{{background:#fff8e7;border-left:4px solid #F2A900;padding:12px 14px}}.tbl{{border-collapse:collapse;width:100%;font-size:11px}}.tbl th,.tbl td{{border-bottom:1px solid #e7e1d6;padding:7px;text-align:left}}.tbl th{{background:#f7f4ed}}
</style></head><body>
<h1>{APP_NAME}</h1><div class="gold">Daily Bitcoin Price Forecasting Report</div><p>Dibuat: {now}</p>
<div class="kpi"><small>Aset</small><b>{RESEARCH_SYMBOL}</b></div>
<div class="kpi"><small>Champion</small><b>{champion_label}</b></div>
<div class="kpi"><small>MAPE</small><b>{champion_mape}</b></div>
<div class="kpi"><small>Validasi</small><b>{WFV_FOLDS}-Fold WFV</b></div>
<h2>Ringkasan Metode</h2><p class="note">Data BTC-USD diproses menggunakan fitur teknikal dan dievaluasi melalui Walk-Forward Validation. PSO mengoptimasi hyperparameter ELM. Model champion yang tersimpan digunakan untuk prediksi harga penutupan H+1.</p>
<h2>Hasil Pengujian</h2>
<p>{poc_text}</p>
<p>Ranking skenario: <b>{ranking}</b></p>
<p>Champion: MAE <b>{champion_mae}</b>, RMSE <b>{champion_rmse}</b>, MAPE <b>{champion_mape}</b>.</p>
<h2>Riwayat Prediksi</h2>{_html_table(predictions[['prediction_mode','source_type','input_date','prediction_date','last_close','predicted_close','delta_pct','created_at']] if 'source_type' in predictions.columns else predictions[['prediction_mode','input_date','prediction_date','last_close','predicted_close','delta_pct','created_at']] if not predictions.empty else predictions)}
<h2>Riwayat Proses Sistem</h2>{_html_table(experiments[['id','name','experiment_type','status','created_at','completed_at']] if not experiments.empty else experiments)}
<p style="margin-top:35px;color:#777;font-size:10px">Periode penelitian: {RESEARCH_START} sampai {RESEARCH_END} (end exclusive) • PSO {PSO_PARTICLES}×{PSO_ITERATIONS} • Target Close(t+1). Semua angka hasil pengujian pada laporan ini dibaca dari database aplikasi.</p>
</body></html>'''
    return html


def render():
    hero(
        'Riwayat & Laporan',
        'Tinjau prediksi yang telah tersimpan, kelola catatan proses, dan unduh ringkasan hasil untuk dokumentasi.',
        'HISTORY • REPORTING'
    )
    preds = list_predictions()
    exps = list_experiments()
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card('Total Prediksi', len(preds), 'research + live')
    with c2: metric_card('Prediksi Live', sum(1 for p in preds if p.get('prediction_mode') == 'live' and p.get('source_type','operational') != 'historical_replay'), 'hasil terbaru')
    with c3: metric_card('Proses Tersimpan', len(exps), 'riwayat sistem')
    with c4: metric_card('Format Laporan', 'HTML / CSV', 'siap diunduh')

    tab_pred, tab_report, tab_manage = st.tabs(['Riwayat Prediksi', 'Unduh Laporan', 'Kelola Riwayat Proses'])

    with tab_pred:
        if not preds:
            st.info('Belum ada prediksi tersimpan.')
        else:
            df = pd.DataFrame(preds)
            if 'source_type' not in df.columns:
                df['source_type'] = 'operational'
            df['source_label'] = df['source_type'].map({
                'historical_replay': 'Replay Historis',
                'operational': 'Prediksi Aktual',
            }).fillna(df['source_type'])
            show = df[['id','prediction_mode','source_label','input_date','prediction_date','last_close','predicted_close','delta_pct','created_at']].copy()
            show.columns = ['ID','Mode','Sumber','Tanggal Data','Tanggal Prediksi','Harga Terakhir','Prediksi H+1','Perubahan (%)','Dibuat']
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button('Unduh Riwayat Prediksi CSV', df.to_csv(index=False).encode('utf-8'), 'bitcoin_prediction_history.csv', 'text/csv', use_container_width=True, key='simple_pred_csv')
            with st.expander('Hapus riwayat prediksi tertentu'):
                labels = {f"{p['id']} — {p['prediction_date']}": p for p in preds}
                target = labels[st.selectbox('Pilih prediksi', list(labels), key='simple_delete_prediction_select')]
                confirm = st.checkbox('Saya yakin ingin menghapus riwayat ini', key='simple_delete_prediction_confirm')
                if st.button('Hapus Riwayat Prediksi', disabled=not confirm, use_container_width=True, key='simple_delete_prediction'):
                    delete_prediction(target['id']); st.success('Riwayat prediksi dihapus.'); st.rerun()

    with tab_report:
        html = _report_html()
        st.markdown('<div class="callout source"><b>Laporan ringkas</b><br>Laporan memuat metodologi, hasil pengujian utama, champion model, dan riwayat prediksi yang tersimpan. Nilai hasil pengujian dibaca dari database aplikasi.</div>', unsafe_allow_html=True)
        st.download_button('Unduh Laporan Sistem (HTML)', html.encode('utf-8'), 'bitcoin_forecast_intelligence_report.html', 'text/html', use_container_width=True, key='simple_report_html')
        if preds:
            st.download_button('Unduh Semua Prediksi (CSV)', pd.DataFrame(preds).to_csv(index=False).encode('utf-8'), 'all_predictions.csv', 'text/csv', use_container_width=True, key='simple_report_pred')
        if exps:
            st.download_button('Unduh Riwayat Proses (CSV)', pd.DataFrame(exps).to_csv(index=False).encode('utf-8'), 'process_history.csv', 'text/csv', use_container_width=True, key='simple_report_exp')
        with st.expander('Ekspor arsip teknis'):
            snapshot = {
                'generated_at': datetime.now().isoformat(),
                'datasets': list_datasets(),
                'experiments': exps,
                'models': [{k:v for k,v in m.items() if k != 'model_path'} for m in list_models()],
                'predictions': preds,
            }
            st.download_button('Unduh Arsip JSON', json.dumps(snapshot, indent=2, default=str).encode('utf-8'), 'bitcoin_forecast_archive.json', 'application/json', use_container_width=True, key='simple_report_json')

    with tab_manage:
        if not exps:
            st.info('Belum ada riwayat proses.')
        else:
            st.caption('Nama dan catatan dapat diperbarui. Nilai hasil perhitungan tetap tersimpan sebagai keluaran sistem dan tidak diedit dari halaman ini.')
            df = pd.DataFrame(exps)
            st.dataframe(df[['id','name','experiment_type','status','created_at','completed_at']], use_container_width=True, hide_index=True)
            labels = {f"{e['id']} — {e['name']}": e for e in exps}
            exp = labels[st.selectbox('Pilih riwayat proses', list(labels), key='simple_manage_exp')]
            with st.form(f'simple_edit_exp_{exp["id"]}'):
                name = st.text_input('Nama', exp['name'])
                notes = st.text_area('Catatan', exp.get('notes') or '')
                if st.form_submit_button('Simpan Perubahan', use_container_width=True):
                    update_experiment(exp['id'], name, notes); st.success('Catatan diperbarui.'); st.rerun()
            with st.expander('Detail hasil tersimpan'):
                runs = pd.DataFrame(experiment_runs(exp['id']))
                folds = pd.DataFrame(experiment_folds(exp['id']))
                if not runs.empty:
                    st.dataframe(runs, use_container_width=True, hide_index=True)
                if not folds.empty:
                    st.dataframe(folds, use_container_width=True, hide_index=True)
            confirm = st.checkbox('Saya yakin ingin menghapus riwayat proses ini', key=f'simple_exp_delete_confirm_{exp["id"]}')
            if st.button('Hapus Riwayat Proses', disabled=not confirm, use_container_width=True, key=f'simple_exp_delete_{exp["id"]}'):
                delete_experiment(exp['id']); st.success('Riwayat proses dihapus.'); st.rerun()

    with st.expander('Aktivitas sistem terbaru'):
        logs = pd.DataFrame(list_activity_logs(50))
        if logs.empty:
            st.caption('Belum ada aktivitas.')
        else:
            st.dataframe(logs[['action','object_type','object_id','description','created_at']], use_container_width=True, hide_index=True)
