import streamlit as st
from auth.authentication import login
from ui.theme import APP_NAME


def render_login():
    st.markdown(
        f'<div class="login-shell"><div class="login-brand">₿</div>'
        f'<div class="eyebrow" style="color:#A77700">{APP_NAME.upper()}</div>'
        '<h2 style="margin:.45rem 0 .3rem;font-size:27px">Selamat Datang</h2>'
        '<p style="color:#746E63;margin-top:0;line-height:1.6">Masuk untuk mengakses dashboard prediksi harga Bitcoin dan hasil pengujian model.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1.15, 1, 1.15])
    with c2:
        with st.form('login_form', clear_on_submit=False):
            username = st.text_input('Username', placeholder='Masukkan username')
            password = st.text_input('Password', type='password', placeholder='Masukkan password')
            submit = st.form_submit_button('Masuk', type='primary', use_container_width=True)
            if submit:
                if login(username, password):
                    st.rerun()
                else:
                    st.error('Username atau password tidak sesuai.')
