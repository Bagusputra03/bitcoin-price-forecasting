import pandas as pd

def download_yfinance(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance belum terpasang. Jalankan: pip install -r requirements.txt") from exc
    raw = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
    if raw.empty:
        raise RuntimeError("Yahoo Finance tidak mengembalikan data untuk parameter tersebut.")
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    try:
        raw.index = raw.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    raw.index.name = "Date"
    return raw
