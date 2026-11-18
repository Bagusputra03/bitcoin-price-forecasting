import pandas as pd
import plotly.express as px
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import (
    list_datasets, load_market_data, create_experiment, save_runs, finish_experiment,
    list_experiments, experiment_runs, experiment_folds, update_experiment, delete_experiment,
)
from research_engine.preprocessing import prepare_market_data
from research_engine.experiments import run_sensitivity, run_poc, run_tournament
from research_engine.config import *


def _dataset_select(key):
    ds = [d for d in list_datasets() if (d.get('mode') or 'research') != 'live']
    if not ds:
        st.info('Tambahkan dataset Research/Custom terlebih dahulu.'); return None
    labels = {f"{d['id']} — {d['name']} [{(d.get('mode') or 'research').upper()}]": d for d in ds}
    return labels[st.selectbox('Dataset', list(labels), key=key)]


def _chart_style(fig, height=390):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=8,r=8,t=45,b=8))
    return fig


def sensitivity():
    hero('Analisis Sensitivitas ELM','Lima konfigurasi ekstrem, 10 run per konfigurasi, WFV 5-fold pada skenario S3.', 'EXPERIMENT • SENSITIVITY')
    d = _dataset_select('sens_ds')
    if not d: return
    if st.button('Jalankan sensitivitas', type='primary', use_container_width=True):
        raw = load_market_data(d['id']); df = prepare_market_data(raw)
        eid = create_experiment('Analisis Sensitivitas ELM','sensitivity',d['id'],config={'grid':SENSITIVITY_GRID,'runs':SENSITIVITY_RUNS})
        try:
            with st.status('Menjalankan ELM sensitivity...', expanded=True) as status:
                table,runs = run_sensitivity(df); save_runs(eid,runs); finish_experiment(eid)
                status.update(label=f'Sensitivity selesai • {eid}', state='complete', expanded=False)
            st.session_state['sens_table'] = table
        except Exception as e:
            finish_experiment(eid,'failed'); st.error(str(e))
    table = st.session_state.get('sens_table')
    if table is not None:
        best = table.loc[table['MAPE Mean'].idxmin()]
        c1,c2,c3 = st.columns(3)
        with c1: metric_card('Grid Best MAPE', f"{best['MAPE Mean']:.4f}%", f"NH {int(best['n_hidden'])}")
        with c2: metric_card('Baseline Locked', f'NH {ELM_BASELINE_NH}', f'C {ELM_BASELINE_C:.0e}')
        with c3: metric_card('Runs / Config', SENSITIVITY_RUNS, 'WFV 5-fold each')
        st.dataframe(table, use_container_width=True, hide_index=True)
        fig = px.bar(table,x=table['n_hidden'].astype(str),y='MAPE Mean',error_y='MAPE Std',labels={'x':'n_hidden'},title='MAPE Mean ± Std')
        st.plotly_chart(_chart_style(fig),use_container_width=True)
        st.info(f'Baseline penelitian tetap ditetapkan secara rule-of-thumb: NH={ELM_BASELINE_NH}, C={ELM_BASELINE_C:.0e}; bukan dipilih otomatis dari grid.')


def poc():
    hero('Proof of Concept — S3 Momentum','ELM baseline dibandingkan dengan PSO–ELM hyperparameter optimizer memakai distribusi run dan uji Mann–Whitney.', 'EXPERIMENT • POC')
    d = _dataset_select('poc_ds')
    if not d: return
    if st.button('Jalankan POC', type='primary', use_container_width=True):
        raw=load_market_data(d['id']); df=prepare_market_data(raw)
        eid=create_experiment('Proof of Concept S3','poc',d['id'],config={'elm_runs':POC_ELM_RUNS,'pso_runs':POC_PSO_RUNS})
        bar=st.progress(0,text='Menyiapkan...')
        def cb(v,t): bar.progress(min(v,1.0),text=t)
        try:
            summary,runs,params=run_poc(df,cb); save_runs(eid,runs); finish_experiment(eid)
            st.session_state['poc_summary']=summary; st.session_state['poc_params']=params; bar.empty()
            st.success(f'POC selesai • {eid}')
        except Exception as e:
            finish_experiment(eid,'failed'); bar.empty(); st.error(str(e))
    s=st.session_state.get('poc_summary')
    if s:
        c1,c2,c3,c4=st.columns(4)
        with c1: metric_card('ELM MAPE',f"{s['elm_mape']:.4f}%",f"± {s['elm_std']:.4f}")
        with c2: metric_card('PSO–ELM MAPE',f"{s['pso_mape']:.4f}%",f"± {s['pso_std']:.4f}")
        with c3: metric_card('Δ MAPE',f"{s['delta_mape']:+.2f}%",'improvement')
        with c4: metric_card('p-value',f"{s['p_value']:.4f}",'Signifikan' if s['significant'] else 'Tidak signifikan')
        dfp=st.session_state.get('poc_params')
        if dfp is not None and not dfp.empty:
            fig=px.scatter(dfp,x='n_hidden',y='log10C',color='Fold',title='Hyperparameter PSO terpilih per fold')
            st.plotly_chart(_chart_style(fig),use_container_width=True)


def tournament():
    hero('Tournament S1–S5','Setiap skenario diuji ELM baseline dan PSO–ELM sebanyak 10 run menggunakan WFV 5-fold.', 'EXPERIMENT • TOURNAMENT')
    d = _dataset_select('tour_ds')
    if not d: return
    if st.button('Jalankan tournament lengkap', type='primary', use_container_width=True):
        raw=load_market_data(d['id']); df=prepare_market_data(raw)
        eid=create_experiment('Tournament S1–S5','tournament',d['id'],config={'runs':TOURNAMENT_RUNS,'val_ratio':PSO_VAL_RATIO})
        bar=st.progress(0,text='Menyiapkan tournament...')
        def cb(v,t): bar.progress(min(v,1.0),text=t)
        try:
            table,runs=run_tournament(df,cb); save_runs(eid,runs); finish_experiment(eid)
            st.session_state['tour_table']=table; bar.empty(); st.success(f'Tournament selesai • {eid}')
        except Exception as e:
            finish_experiment(eid,'failed'); bar.empty(); st.error(str(e))
    table=st.session_state.get('tour_table')
    if table is not None:
        rank = table.sort_values('PSO MAPE').reset_index(drop=True)
        c1,c2,c3 = st.columns(3)
        with c1: metric_card('Champion', rank.iloc[0]['Skenario'].split('(')[0].strip(), 'lowest PSO MAPE')
        with c2: metric_card('Best PSO MAPE', f"{rank.iloc[0]['PSO MAPE']:.4f}%", rank.iloc[0]['Skenario'])
        with c3: metric_card('Scenarios', len(table), 'S1–S5')
        st.dataframe(table,use_container_width=True,hide_index=True)
        fig=px.bar(table,x='Skenario',y=['ELM MAPE','PSO MAPE'],barmode='group',title='ELM vs PSO–ELM per skenario')
        st.plotly_chart(_chart_style(fig,420),use_container_width=True)


def history():
    hero('Riwayat & Audit Eksperimen','CRUD metadata + audit read-only sampai level run dan fold. Metrik mesin tidak dapat diedit manual.', 'EXPERIMENT REGISTRY • AUDIT')
    rows=list_experiments()
    if not rows:
        st.info('Belum ada eksperimen.'); return

    df_all=pd.DataFrame(rows)
    f1,f2,f3 = st.columns([1,1,2])
    type_options=['Semua']+sorted(df_all['experiment_type'].dropna().unique().tolist())
    status_options=['Semua']+sorted(df_all['status'].dropna().unique().tolist())
    typ=f1.selectbox('Type',type_options); stat=f2.selectbox('Status',status_options)
    query=f3.text_input('Cari eksperimen',placeholder='ID atau nama...')
    filtered=df_all.copy()
    if typ!='Semua': filtered=filtered[filtered['experiment_type']==typ]
    if stat!='Semua': filtered=filtered[filtered['status']==stat]
    if query.strip():
        q=query.lower().strip(); filtered=filtered[filtered.apply(lambda r:q in f"{r['id']} {r['name']}".lower(),axis=1)]
    if filtered.empty:
        st.info('Tidak ada eksperimen sesuai filter.'); return

    st.dataframe(filtered[['id','name','experiment_type','dataset_name','dataset_mode','status','started_at','completed_at']],use_container_width=True,hide_index=True)
    st.download_button('Export Experiment Registry CSV',filtered.to_csv(index=False).encode('utf-8'),'experiment_registry.csv','text/csv',use_container_width=True,key='exp_registry_export')

    ids=set(filtered['id'])
    filtered_rows=[r for r in rows if r['id'] in ids]
    labels={f"{r['id']} — {r['name']}":r for r in filtered_rows}
    r=labels[st.selectbox('Pilih eksperimen',list(labels),key='history_exp')]

    tab_detail,tab_runs,tab_folds,tab_manage=st.tabs(['Detail','Runs','Fold Results','Manage'])
    with tab_detail:
        c1,c2,c3,c4=st.columns(4)
        with c1: metric_card('Type',r['experiment_type'],'experiment')
        with c2: metric_card('Status',r['status'].upper(),'machine state')
        with c3: metric_card('Dataset',r['dataset_name'],(r.get('dataset_mode') or 'research').upper())
        with c4: metric_card('ID',r['id'],'audit key')
        st.caption(r.get('notes') or 'Tidak ada catatan.')
        with st.expander('Konfigurasi eksperimen'):
            st.code(r.get('config_json') or '{}',language='json')

    with tab_runs:
        runs=experiment_runs(r['id'])
        if not runs:
            st.info('Eksperimen ini belum memiliki run tersimpan.')
        else:
            dfr=pd.DataFrame(runs)
            st.dataframe(dfr[['scenario','model_type','run_number','seed','mae','rmse','mape','execution_time']],use_container_width=True,hide_index=True)
            st.download_button('Export Runs CSV',dfr.to_csv(index=False).encode('utf-8'),f"{r['id']}_runs.csv",'text/csv',use_container_width=True,key=f"runs_export_{r['id']}")

    with tab_folds:
        folds=experiment_folds(r['id'])
        if not folds:
            st.info('Tidak ada fold detail untuk eksperimen ini.')
        else:
            dff=pd.DataFrame(folds)
            st.dataframe(dff[['scenario','model_type','run_number','fold_number','train_size','test_size','n_hidden','c_value','mae','rmse','mape']],use_container_width=True,hide_index=True)
            st.download_button('Export Fold Results CSV',dff.to_csv(index=False).encode('utf-8'),f"{r['id']}_folds.csv",'text/csv',use_container_width=True,key=f"fold_export_{r['id']}")

    with tab_manage:
        st.markdown('#### Update metadata')
        with st.form(f"edit_exp_{r['id']}"):
            name=st.text_input('Nama',r['name']); notes=st.text_area('Catatan',r['notes'] or '')
            if st.form_submit_button('Save Metadata',use_container_width=True):
                update_experiment(r['id'],name,notes); st.success('Metadata diperbarui.'); st.rerun()
        st.markdown('#### Delete experiment')
        st.caption('Run/fold terkait ikut terhapus. Model deployment yang mereferensikan eksperimen tetap disimpan dengan experiment_id menjadi NULL.')
        confirm=st.checkbox(f"Saya yakin menghapus {r['id']}",key=f"delete_exp_confirm_{r['id']}")
        if st.button('Delete Experiment',use_container_width=True,disabled=not confirm,key=f"delete_exp_{r['id']}"):
            delete_experiment(r['id']); st.success('Eksperimen dihapus.'); st.rerun()
