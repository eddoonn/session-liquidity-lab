"""Audit 2/4: data quality -- duplicates, holes, dead bars, bad ticks, roll risk."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search2 import load
from download_data import SYMBOLS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def daily_atr(df, n=20):
    d = df["Close"].resample("1D").last().dropna()
    h = df["High"].resample("1D").max().reindex(d.index)
    l = df["Low"].resample("1D").min().reindex(d.index)
    tr = pd.concat([h - l, (h - d.shift()).abs(), (l - d.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=5).mean()


def main():
    lines = []
    for inst in SYMBOLS:
        df = load(inst)
        idx = df.index
        dup = int(idx.duplicated().sum())
        zr = int(((df["High"] == df["Low"])).sum())
        # gaps between consecutive bars on Mon-Fri
        dt = idx.to_series().diff()
        wd = idx.weekday < 5
        holes = dt[(dt > pd.Timedelta(hours=3)) & wd]
        # bad ticks / roll jumps: |close-to-close| > 5x daily ATR
        datr = daily_atr(df)
        dcc = df["Close"].resample("1D").last().dropna()
        jump = (dcc.diff().abs() / datr).dropna()
        big = jump[jump > 5]
        y26 = df[df.index >= "2026-01-01"]
        dt26 = y26.index.to_series().diff()
        holes26 = dt26[(dt26 > pd.Timedelta(hours=3)) & (y26.index.weekday < 5)]
        msg = [
            f"{inst}: bars={len(df)} dup={dup} zerorange={zr}",
            f"  holes>3h weekdays(full): {len(holes)}, longest {holes.max() if len(holes) else '-'}",
            f"  holes>3h weekdays(2026): {len(holes26)}",
            f"  daily moves >5xATR20: {len(big)} "
            f"{dict(zip(big.index.strftime('%Y-%m-%d'), big.round(1)))}" if len(big) else
            "  daily moves >5xATR20: 0",
        ]
        print("\n".join(msg))
        lines += msg + [""]
    with open(os.path.join(OUT, "data_quality.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
