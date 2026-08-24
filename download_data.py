import os
import pandas as pd
import yfinance as yf

SYMBOLS = {
    "GOLD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def fetch(symbol: str) -> pd.DataFrame:
    frames = []
    # Yahoo hourly limit ~730d; request in two chunks for overlap safety
    for period in ("730d",):
        df = yf.download(symbol, period=period, interval="60m", progress=False,
                         auto_adjust=False, prepost=False)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit(f"No data for {symbol}")
    df = pd.concat(frames)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.dropna()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, sym in SYMBOLS.items():
        df = fetch(sym)
        path = os.path.join(DATA_DIR, f"{name}_60m.csv")
        df.to_csv(path)
        print(f"{name:7s} {sym:10s} bars={len(df):5d} {df.index[0]} .. {df.index[-1]}")


if __name__ == "__main__":
    main()
