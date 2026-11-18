import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import hero, metric_card
from services.repository import latest_parity_report
from services.research_results_service import (
    get_sensitivity_results,
    get_poc_results,
    get_tournament_results,
    get_champion_results,
)
from research_engine.config import (
    ELM_BASELINE_NH, ELM_BASELINE_C, PSO_PARTICLES, PSO_ITERATIONS,
    WFV_FOLDS, SCENARIOS,
)


def _style(fig, height=390):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=45, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='rgba(120,105,80,.12)')
    return fig


def _source_caption(label, source):
    if not source:
        return
    when = source.get('completed_at') or source.get('created_at') or '—'
    st.caption(f'{label} dibaca dari hasil perhitungan yang tersimpan di database • {when}')


def render():
    hero(
        'Hasil Pengujian Model',
        'Ringkasan pengujian ELM dan PSO–ELM, perbandingan lima skenario fitur, serta model terbaik yang digunakan untuk prediksi.',
        'MODEL EVALUATION • RESEARCH RESULTS'
    )

    if latest_parity_report():
        st.markdown(
            '<div class="callout live"><b>Jejak validasi penelitian tersedia</b><br>'
            'Hasil komputasi penelitian telah tersimpan di basis data dan digunakan sebagai sumber angka pada halaman ini.</div>',
            unsafe_allow_html=True,
        )

    sensitivity, sens_src = get_sensitivity_results()
    poc, poc_src = get_poc_results()
    tour, tour_src = get_tournament_results()
    champ, champ_folds, champ_src = get_champion_results()

    missing = []
    if not poc: missing.append('perbandingan ELM–PSO-ELM')
    if tour.empty: missing.append('turnamen S1–S5')
    if not champ: missing.append('evaluasi champion')
    if missing:
        st.warning(
            'Sebagian hasil belum tersedia di database: ' + ', '.join(missing) + '. '
            'Angka tidak akan digantikan oleh nilai statis pada antarmuka.'
        )

    st.markdown('### 1. Perbandingan ELM dan PSO–ELM')
    if not poc:
        st.info('Belum ada hasil POC tersimpan di database.')
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card('ELM Baseline', f"{poc['elm_mape']:.4f}%", f"MAPE ± {poc['elm_std']:.4f}")
        with c2: metric_card('PSO–ELM', f"{poc['pso_mape']:.4f}%", f"MAPE ± {poc['pso_std']:.4f}")
        with c3: metric_card('Peningkatan', f"{poc['delta_mape']:+.2f}%", 'penurunan MAPE')
        with c4: metric_card('Kesimpulan', 'PSO–ELM', 'lebih akurat pada S3')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='ELM Baseline', x=['ELM Baseline'], y=[poc['elm_mape']],
            marker_color='#222222', error_y=dict(type='data', array=[poc['elm_std']])
        ))
        fig.add_trace(go.Bar(
            name='PSO–ELM', x=['PSO–ELM'], y=[poc['pso_mape']],
            marker_color='#F2A900', error_y=dict(type='data', array=[poc['pso_std']])
        ))
        fig.update_layout(title='MAPE Mean ± Standar Deviasi', yaxis_title='MAPE (%)', showlegend=False)
        st.plotly_chart(_style(fig, 330), use_container_width=True)
        _source_caption('Perbandingan model', poc_src)

    st.markdown('### 2. Perbandingan Lima Skenario Fitur')
    if tour.empty:
        st.info('Belum ada hasil turnamen yang tersimpan di database.')
    else:
        display = tour.copy()
        display.insert(0, 'Peringkat', range(1, len(display) + 1))
        display['Skenario'] = display['Skenario'].str.replace(r'\s*\(.*\)', '', regex=True)
        display['ELM MAPE'] = display['ELM MAPE'].map(lambda x: f'{float(x):.4f}%')
        display['PSO MAPE'] = display['PSO MAPE'].map(lambda x: f'{float(x):.4f}%')
        display['ΔMAPE'] = display['ΔMAPE'].map(lambda x: f'{float(x):+.2f}%')
        st.dataframe(
            display[['Peringkat', 'Skenario', 'ELM MAPE', 'PSO MAPE', 'ΔMAPE']],
            use_container_width=True, hide_index=True
        )

        fig2 = go.Figure()
        short = [x.split('(')[0].strip() for x in tour['Skenario']]
        fig2.add_trace(go.Bar(name='ELM Baseline', x=short, y=tour['ELM MAPE'], marker_color='#2A2A2A'))
        fig2.add_trace(go.Bar(name='PSO–ELM', x=short, y=tour['PSO MAPE'], marker_color='#F2A900'))
        fig2.update_layout(barmode='group', title='MAPE ELM vs PSO–ELM per Skenario', yaxis_title='MAPE (%)')
        st.plotly_chart(_style(fig2, 390), use_container_width=True)
        _source_caption('Turnamen skenario', tour_src)

    st.markdown('### 3. Model Terbaik')
    if not champ:
        st.info('Belum ada hasil evaluasi champion yang tersimpan di database.')
    else:
        scenario = str(champ.get('scenario') or 'S4 (Tren SMA)')
        short_scenario = scenario.split('(')[0].strip()
        scenario_name = scenario[scenario.find('(')+1:scenario.rfind(')')] if '(' in scenario else 'Tren SMA'
        a, b, c, d = st.columns(4)
        with a: metric_card('Skenario Terbaik', short_scenario, scenario_name)
        with b: metric_card('MAPE', f"{float(champ['mape']):.4f}%", 'WFV 5-fold')
        with c: metric_card('RMSE', f"${float(champ['rmse']):,.2f}", 'rata-rata')
        with d: metric_card('MAE', f"${float(champ['mae']):,.2f}", 'rata-rata')

        features = SCENARIOS.get(scenario, SCENARIOS.get('S4 (Tren SMA)', []))
        st.markdown(
            f'<div class="callout source"><b>{scenario} dipilih sebagai model champion.</b><br>'
            f'Fitur: {", ".join(features)}. Skenario ini digunakan pada proses prediksi Bitcoin H+1.</div>',
            unsafe_allow_html=True,
        )
        _source_caption('Evaluasi champion', champ_src)

    with st.expander('Lihat detail evaluasi dan konfigurasi penelitian'):
        st.markdown(f'''**Konfigurasi utama**  
- ELM baseline: **NH={ELM_BASELINE_NH}, C={ELM_BASELINE_C:.0e}**  
- Optimasi PSO: **{PSO_PARTICLES} partikel × {PSO_ITERATIONS} iterasi**  
- Evaluasi: **Walk-Forward Validation {WFV_FOLDS}-fold**  
- Target: **Close(t+1)**
''')

        st.markdown('**Analisis sensitivitas ELM**')
        if sensitivity.empty:
            st.caption('Belum ada hasil sensitivitas tersimpan di database.')
        else:
            cols = [c for c in ['n_hidden', 'C', 'MAE Mean', 'RMSE Mean', 'MAPE Mean', 'MAPE Std'] if c in sensitivity.columns]
            sshow = sensitivity[cols].copy()
            if 'C' in sshow: sshow['C'] = sshow['C'].map(lambda x: f'{float(x):.0e}')
            if 'MAPE Mean' in sshow: sshow['MAPE Mean'] = sshow['MAPE Mean'].map(lambda x: f'{float(x):.4f}%')
            if 'MAPE Std' in sshow: sshow['MAPE Std'] = sshow['MAPE Std'].map(lambda x: f'{float(x):.4f}%')
            st.dataframe(sshow, use_container_width=True, hide_index=True)
            _source_caption('Analisis sensitivitas', sens_src)

        if champ_folds is not None and not champ_folds.empty:
            st.markdown('**Evaluasi champion per fold**')
            f = champ_folds.copy()
            wanted = [c for c in ['fold_number', 'train_size', 'test_size', 'n_hidden', 'c_value', 'mae', 'rmse', 'mape'] if c in f.columns]
            st.dataframe(f[wanted], use_container_width=True, hide_index=True)
