"""Consistency check: do hourly bars equal the aggregate of their 15m children?
If not, neither dataset is trustworthy as 'truth' for the other."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from search2 import load

for inst in ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]:
    df_h = load(inst)
    df_f = pd.read_csv(f"data/{inst}_15m.csv", index_col=0, parse_dates=True)
    df_f.index = pd.to_datetime(df_f.index, utc=True)
    ph = df_f.index.floor("h")
    g = df_f.groupby(ph)
    f_hi, f_lo = g["High"].max(), g["Low"].min()
    common = df_h.index[df_h.index >= df_f.index[0]]
    hh = df_h.loc[common, "High"]
    ll = df_h.loc[common, "Low"]
    hi_mismatch = (f_hi.reindex(common) < hh - 1e-9) | (f_hi.reindex(common).isna() & (hh > 0))
    lo_mismatch = (f_lo.reindex(common) > ll + 1e-9) | f_lo.reindex(common).isna()
    n = len(common)
    print(f"{inst}: hours compared={n} | 15m-missing-hour={(f_hi.reindex(common).isna()).sum()} "
          f"| high mismatch={int((f_hi.reindex(common) < hh - 1e-9).sum())} "
          f"low mismatch={int((f_lo.reindex(common) > ll + 1e-9).sum())}")
