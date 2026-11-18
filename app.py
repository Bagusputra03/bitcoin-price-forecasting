import streamlit as st
from database.database import init_db
from auth.authentication import is_authenticated, logout
from ui.theme import apply_theme, APP_NAME, APP_TAGLINE
from ui.login import render_login

st.set_page_config(
    page_title=APP_NAME,
    page_icon='₿',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items=None,
)
init_db()
apply_theme()

if not is_authenticated():
    render_login()
    st.stop()

from ui import simple_dashboard, simple_data_process, simple_results, simple_prediction, simple_history

with st.sidebar:
    st.markdown(
        '<div class="brand-shell"><div class="brand-icon">₿</div>'
        '<div><div class="brand-title">Bitcoin Forecast<br>Intelligence</div>'
        '<div class="brand-sub">DAILY PRICE FORECASTING</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-caption">MENU UTAMA</div>', unsafe_allow_html=True)
    page = st.radio(
        'Navigasi',
        ['Dashboard', 'Data & Proses', 'Hasil Pengujian', 'Prediksi Bitcoin', 'Riwayat & Laporan'],
        label_visibility='collapsed',
        key='main_navigation',
    )
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="signed-card"><small>AKUN AKTIF</small><b>{st.session_state.get("username", "admin")}</b>'
        f'<span>{APP_TAGLINE}</span></div>',
        unsafe_allow_html=True,
    )
    if st.button('Keluar', use_container_width=True, key='sidebar_logout'):
        logout()
        st.rerun()

routes = {
    'Dashboard': simple_dashboard.render,
    'Data & Proses': simple_data_process.render,
    'Hasil Pengujian': simple_results.render,
    'Prediksi Bitcoin': simple_prediction.render,
    'Riwayat & Laporan': simple_history.render,
}
routes[page]()
