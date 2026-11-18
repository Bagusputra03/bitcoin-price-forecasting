import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]

def prepare_market_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Menyalin preprocessing/feature engineering dari blok final penelitian."""
    if raw is None or raw.empty:
        raise ValueError("Dataset kosong.")
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom OHLCV tidak lengkap: {missing}")

    df = raw[REQUIRED].copy()
    df = df.interpolate(method="linear").dropna()

    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss))
    df["MACD"] = (
        df["Close"].ewm(span=12, adjust=False).mean()
        - df["Close"].ewm(span=26, adjust=False).mean()
    )
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Up"] = df["BB_Mid"] + (df["BB_Std"] * 2)
    df["BB_Low"] = df["BB_Mid"] - (df["BB_Std"] * 2)
    df["Target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df

def prepare_inference_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering untuk inferensi H+1 tanpa membuat Target.

    Rumus fitur identik dengan pipeline penelitian, tetapi baris terbaru dipertahankan
    karena pada saat inferensi nilai Target(t+1) memang belum diketahui.
    """
    if raw is None or raw.empty:
        raise ValueError("Dataset kosong.")
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom OHLCV tidak lengkap: {missing}")

    df = raw[REQUIRED].copy()
    df = df.interpolate(method="linear").dropna()
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss))
    df["MACD"] = (
        df["Close"].ewm(span=12, adjust=False).mean()
        - df["Close"].ewm(span=26, adjust=False).mean()
    )
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_30"] = df["Close"].rolling(30).mean()
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Up"] = df["BB_Mid"] + (df["BB_Std"] * 2)
    df["BB_Low"] = df["BB_Mid"] - (df["BB_Std"] * 2)
    df.dropna(inplace=True)
    return df
