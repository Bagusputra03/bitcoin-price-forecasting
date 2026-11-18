import hashlib
from auth import authentication
from auth.authentication import verify_password


def _encoded(password='secret', salt='salt123', iterations=1000):
    digest = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), iterations).hex()
    return f'pbkdf2_sha256${iterations}${salt}${digest}'


def test_password_verification():
    encoded = _encoded()
    assert verify_password('secret', encoded)
    assert not verify_password('wrong', encoded)
    assert not verify_password('secret', 'invalid')


def test_logout_clears_session(monkeypatch):
    class Session(dict):
        pass
    class FakeSt:
        session_state = Session(authenticated=True, username='admin', latest_live_prediction={'id':'PRD-X'})
    monkeypatch.setattr(authentication, '_streamlit', lambda: FakeSt)
    authentication.logout()
    assert FakeSt.session_state == {}
