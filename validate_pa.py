"""Validate path-aware hourly simulation against 15m ground truth (last ~60d)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from search2 import load
from engine import Data, run_pattern
from fine15 import sim15, BOOKS

COSTS = {"GOLD": 0.35, "EURUSD": 0.00020, "GBPUSD": 0.00028, "USDJPY": 0.024}

hdr = f"{'book':16s} {'nH=nF':>6s} {'R_naive':>8s} {'R_path':>8s} {'R_15m':>8s} {'dPA-15m':>8s} {'exact%':>7s}"
print(hdr)
rows = []
for inst in ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]:
    df_h = load(inst)
    df_f = pd.read_csv(f"data/{inst}_15m.csv", index_col=0, parse_dates=True)
    df_f.index = pd.to_datetime(df_f.index, utc=True)
    d = Data(df_h, atr_lens=(10,))
    for bname, spec0 in BOOKS[inst]:
        spec_h = dict(spec0); spec_h["path_aware"] = True
        tr_hn = [t for t in run_pattern(d, dict(spec0), cost=COSTS[inst])
                 if t["entry_time"] >= df_f.index[0]]
        tr_hp = [t for t in run_pattern(d, spec_h, cost=COSTS[inst])
                 if t["entry_time"] >= df_f.index[0]]
        tr_f = sim15(df_h, df_f, spec0, cost=COSTS[inst])
        mh = {(t["date"], t["side"]): t for t in tr_hp}
        mf = {(t["date"], t["side"]): t for t in tr_f}
        common = set(mh) & set(mf)
        rh_ = sum(mh[k]["r"] for k in common)
        rf_ = sum(mf[k]["r"] for k in common)
        rn = sum(t["r"] for t in tr_hn)
        agree = np.mean([abs(mh[k]["r"] - mf[k]["r"]) < 1e-9 for k in common]) * 100 if common else 0.0
        name = f"{inst} {bname}"
        print(f"{name:16s} {len(tr_hp):3d}/{len(tr_f):<3d} {rn:+8.2f} {rh_:+8.2f} {rf_:+8.2f} {rh_-rf_:+8.2f} {agree:6.0f}%")
        rows.append(dict(inst=inst, book=bname, n=len(common), r_naive=rn,
                         r_pathaware=rh_, r_15m=rf_, delta_pa_15m=rh_ - rf_, exact_pct=agree))
pd.DataFrame(rows).to_csv("results/fine15_pathaware_validation.csv", index=False)
print("\nsaved results/fine15_pathaware_validation.csv")
