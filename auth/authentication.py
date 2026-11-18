import hashlib, hmac


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iterations, salt, expected = encoded.split('$', 3)
        if scheme != 'pbkdf2_sha256':
            return False
        actual = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), int(iterations)).hex()
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _streamlit():
    import streamlit as st
    return st


def is_authenticated():
    st = _streamlit()
    return bool(st.session_state.get('authenticated', False))


def login(username, password):
    st = _streamlit()
    try:
        valid_user = username == st.secrets['auth']['username']
        valid_pass = verify_password(password, st.secrets['auth']['password_hash'])
    except Exception:
        valid_user = valid_pass = False
    if valid_user and valid_pass:
        st.session_state['authenticated'] = True
        st.session_state['username'] = username
        return True
    return False


def logout():
    """Clear the whole app session so no prediction/experiment state leaks after logout."""
    st = _streamlit()
    try:
        st.session_state.clear()
    except AttributeError:
        for key in list(st.session_state.keys()):
            st.session_state.pop(key, None)
