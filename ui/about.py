import pandas as pd
import streamlit as st
from ui.theme import hero, metric_card
from research_engine.config import *


def render():
    hero('Metodologi & Informasi Sistem','Dokumentasi ringkas mengenai arsitektur, batas antara Research Mode dan Live Mode, serta aturan audit BagSistem.', 'SYSTEM DOCUMENTATION')

    c1,c2,c3,c4=st.columns(4)
    with c1: metric_card('Asset',RESEARCH_SYMBOL,'Yahoo Finance')
    with c2: metric_card('Horizon','H+1','Close next day')
    with c3: metric_card('Champion','S4','Tren SMA')
    with c4: metric_card('Validation','5-Fold WFV','expanding window')

    st.markdown('### Arsitektur')
    st.markdown('''
<div class="architecture">
  <div><b>Streamlit UI</b><small>Login • CRUD • Dashboard • Reports</small></div><span>→</span>
  <div><b>Service Layer</b><small>Repository • Audit • Registry</small></div><span>→</span>
  <div><b>Research Engine</b><small>ELM • PSO • WFV • Metrics</small></div><span>→</span>
  <div><b>SQLite</b><small>Snapshots • Runs • Models • Predictions</small></div>
</div>''',unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(['Research Mode','Live Mode','Skenario','Data Governance'])
    with tab1:
        st.markdown('''### Frozen research pipeline
Research Mode mereproduksi alur notebook final dan **tidak berubah ketika Live Mode digunakan**.

1. BTC-USD daily OHLCV.
2. Interpolasi linear dan feature engineering.
3. Target `Close(t+1)`.
4. Min-Max sesuai source.
5. WFV 5-fold expanding window.
6. ELM baseline dibandingkan dengan PSO-ELM.
7. Tournament S1–S5.
8. Champion source: **S4 (Tren SMA)**.
9. Parity validation terhadap output notebook final.
''')
        st.code(f'''Research period: {RESEARCH_START} → {RESEARCH_END} (end exclusive)
ELM baseline: NH={ELM_BASELINE_NH}, C={ELM_BASELINE_C:.0e}
PSO: N={PSO_PARTICLES}, I={PSO_ITERATIONS}, c1={PSO_C1}, c2={PSO_C2}, w={PSO_WMAX}→{PSO_WMIN}
PSO bounds: NH {PSO_NH_BOUNDS}, log10(C) {PSO_LOG10C_BOUNDS}
Internal validation: {PSO_VAL_RATIO:.0%}
Champion seed: {CHAMPION_PSO_SEED}''')
    with tab2:
        st.markdown('''### Operational live extension
Live Mode memakai **arsitektur S4 dan proses deployment yang sama**, tetapi memperbarui BTC-USD sampai daily candle terbaru dari Yahoo Finance.

- Data live disimpan sebagai snapshot terpisah (`mode=live`).
- PSO-ELM S4 dilatih ulang pada seluruh data live yang sudah memiliki target.
- Candle terbaru dipertahankan sebagai input inferensi.
- Hasil H+1 disimpan dengan `prediction_mode=live`.
- MAPE penelitian **tidak diklaim sebagai MAPE live baru**.

Karena interval penelitian adalah `1d`, Live Mode bukan tick-by-tick trading feed. Prediksi mengikuti daily candle terbaru yang tersedia.
''')
    with tab3:
        rows=[{'Scenario':k,'Features':' + '.join(v)} for k,v in SCENARIOS.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    with tab4:
        st.markdown('''### CRUD & immutability rules
- **Dataset:** Create, Read, Update metadata, Delete untuk dataset yang tidak dikunci.
- **Experiment:** metadata dapat diubah; run/fold metrics read-only.
- **Model:** dapat diaktifkan/dihapus; parameter hasil training tidak diedit manual.
- **Prediction:** machine output tersimpan immutable; hanya histori record yang dapat dihapus.
- **Research snapshot:** dapat dikunci untuk mencegah accidental deletion.
- **Live snapshot:** diperbarui hanya melalui Mode Prediksi Live.

### Authentication
Aplikasi menggunakan satu akun penuh (single-user) dengan password hash PBKDF2 dan Streamlit session state. Logout membersihkan session authentication.
''')

    st.markdown('<div class="callout source"><b>Interpretasi penggunaan</b><br>BagSistem adalah sistem penelitian/prediksi, bukan rekomendasi investasi. Output harga H+1 adalah keluaran model berdasarkan data dan metodologi yang diterapkan.</div>',unsafe_allow_html=True)
