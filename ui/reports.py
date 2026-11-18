import json
from datetime import datetime
import pandas as pd
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    list_datasets, list_experiments, list_models, list_predictions,
    list_activity_logs, latest_parity_report, latest_prediction,
)
from research_engine.config import *
from research_engine.parity import flatten_checks


def _html_table(df):
    if df is None or df.empty:
        return '<p><em>No data.</em></p>'
    return df.to_html(index=False, border=0, classes='tbl')


def _build_html():
    datasets = pd.DataFrame(list_datasets())
    experiments = pd.DataFrame(list_experiments())
    models = pd.DataFrame(list_models())
    predictions = pd.DataFrame(list_predictions())
    parity = latest_parity_report()
    parity_text = 'Not available'
    parity_detail = ''
    if parity:
        try:
            rep = json.loads(parity['report_json'])
            checks = flatten_checks(rep)
            pass_n = sum(1 for x in checks if x.get('status') == 'PASS')
            parity_text = f"{'PASS' if parity['passed'] else 'STRICT FAIL'} — {pass_n}/{len(checks)} checks"
            fail = pd.DataFrame([x for x in checks if x.get('status') != 'PASS'])
            parity_detail = _html_table(fail) if not fail.empty else '<p>All checks passed.</p>'
        except Exception:
            parity_text = 'Stored report could not be parsed.'

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>BagSistem Report</title>
<style>
body{{font-family:Arial,sans-serif;color:#172033;margin:36px;line-height:1.5;background:#fff}}
h1{{font-size:28px}}h2{{margin-top:30px;border-bottom:1px solid #e5e7eb;padding-bottom:8px}}
.kpi{{display:inline-block;border:1px solid #e5e7eb;border-radius:12px;padding:12px 16px;margin:4px 8px 4px 0;min-width:150px}}
.kpi small{{display:block;color:#667085}}.kpi b{{font-size:20px}}.tbl{{border-collapse:collapse;width:100%;font-size:12px}}
.tbl th,.tbl td{{border-bottom:1px solid #e5e7eb;padding:8px;text-align:left}}.tbl th{{background:#f8fafc}}
.note{{background:#f8fafc;border-left:4px solid #6366f1;padding:12px 14px}}
</style></head><body>
<h1>BagSistem • PSO-ELM Research & Live Report</h1><p>Generated: {now}</p>
<div class="kpi"><small>Research asset</small><b>{RESEARCH_SYMBOL}</b></div>
<div class="kpi"><small>Champion</small><b>S4 Tren SMA</b></div>
<div class="kpi"><small>WFV</small><b>{WFV_FOLDS}-Fold</b></div>
<div class="kpi"><small>PSO</small><b>{PSO_PARTICLES}×{PSO_ITERATIONS}</b></div>
<h2>Methodology Contract</h2><p class="note">Research Mode is frozen to {RESEARCH_START} → {RESEARCH_END} (end exclusive), target Close H+1, baseline NH={ELM_BASELINE_NH}/C={ELM_BASELINE_C:.0e}, and source champion S4. Live Mode is an operational extension and does not alter research parity.</p>
<h2>Parity</h2><p><b>{parity_text}</b></p>{parity_detail}
<h2>Datasets</h2>{_html_table(datasets)}
<h2>Experiments</h2>{_html_table(experiments)}
<h2>Models</h2>{_html_table(models)}
<h2>Predictions</h2>{_html_table(predictions)}
</body></html>'''
    return html


def render():
    hero('Laporan & Ekspor','Ekspor data audit, model, prediksi, dan ringkasan sistem tanpa mengubah machine output yang tersimpan.', 'PELAPORAN • AUDIT • EKSPOR')
    datasets = pd.DataFrame(list_datasets())
    experiments = pd.DataFrame(list_experiments())
    models = pd.DataFrame(list_models())
    predictions = pd.DataFrame(list_predictions())
    logs = pd.DataFrame(list_activity_logs(500))

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card('Datasets',len(datasets),'metadata registry')
    with c2: metric_card('Experiments',len(experiments),'audit history')
    with c3: metric_card('Models',len(models),'research + live')
    with c4: metric_card('Predictions',len(predictions),'immutable outputs')

    st.markdown('### System report')
    html = _build_html()
    st.download_button('Download System Report (HTML)', html.encode('utf-8'), 'bagsistem_report.html', 'text/html', use_container_width=True, key='report_html')
    st.caption('HTML dapat dibuka langsung di browser dan dicetak ke PDF melalui Print → Save as PDF bila diperlukan.')

    st.markdown('### Data exports')
    cols = st.columns(2)
    with cols[0]:
        if not datasets.empty:
            st.download_button('Dataset Registry CSV',datasets.to_csv(index=False).encode('utf-8'),'datasets.csv','text/csv',use_container_width=True,key='report_ds')
        if not experiments.empty:
            st.download_button('Experiment Registry CSV',experiments.to_csv(index=False).encode('utf-8'),'experiments.csv','text/csv',use_container_width=True,key='report_exp')
        if not logs.empty:
            st.download_button('Activity Log CSV',logs.to_csv(index=False).encode('utf-8'),'activity_logs.csv','text/csv',use_container_width=True,key='report_logs')
    with cols[1]:
        if not models.empty:
            st.download_button('Model Registry CSV',models.to_csv(index=False).encode('utf-8'),'models.csv','text/csv',use_container_width=True,key='report_models')
        if not predictions.empty:
            st.download_button('Riwayat Prediksi CSV',predictions.to_csv(index=False).encode('utf-8'),'predictions.csv','text/csv',use_container_width=True,key='report_predictions')
        snapshot = {
            'generated_at': datetime.now().isoformat(),
            'research_config': {
                'symbol': RESEARCH_SYMBOL,'start': RESEARCH_START,'end': RESEARCH_END,'interval': RESEARCH_INTERVAL,
                'baseline_nh': ELM_BASELINE_NH,'baseline_C': ELM_BASELINE_C,'wfv_folds': WFV_FOLDS,
                'pso_particles': PSO_PARTICLES,'pso_iterations': PSO_ITERATIONS,'pso_val_ratio': PSO_VAL_RATIO,
                'champion': RESEARCH_CHAMPION_SCENARIO,
            },
            'datasets': list_datasets(), 'experiments': list_experiments(), 'models': list_models(), 'predictions': list_predictions(),
        }
        st.download_button('Full Audit Snapshot JSON',json.dumps(snapshot,indent=2,default=str).encode('utf-8'),'bagsistem_audit.json','application/json',use_container_width=True,key='report_json')

    st.markdown('### Preview')
    tab1,tab2,tab3 = st.tabs(['Models','Predictions','Activity Log'])
    with tab1:
        st.dataframe(models,use_container_width=True,hide_index=True) if not models.empty else st.info('No models.')
    with tab2:
        st.dataframe(predictions,use_container_width=True,hide_index=True) if not predictions.empty else st.info('No predictions.')
    with tab3:
        st.dataframe(logs,use_container_width=True,hide_index=True) if not logs.empty else st.info('No activity logs.')
