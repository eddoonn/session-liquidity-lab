"""5-minute ground truth: consistency vs hourly, then recalibrate execution tax
for all 7 books at 5m resolution (and compare against the 15m estimates)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from search2 import load
from engine import Data, run_pattern
from fine15 import sim_fine, BOOKS

COSTS = {"GOLD": 0.35, "EURUSD": 0.00020, "GBPUSD": 0.00028, "USDJPY": 0.024}


def check_consistency(inst, df_h):
    f = pd.read_csv(f"data/{inst}_5m.csv", index_col=0, parse_dates=True)
    f.index = pd.to_datetime(f.index, utc=True)
    ph = f.index.floor("h")
    g = f.groupby(ph)
    hi_m = int((g["High"].max().reindex(df_h.index[df_h.index >= f.index[0]]) <
                df_h["High"][df_h.index >= f.index[0]] - 1e-9).sum())
    lo_m = int((g["Low"].min().reindex(df_h.index[df_h.index >= f.index[0]]) >
                df_h["Low"][df_h.index >= f.index[0]] + 1e-9).sum())
    return hi_m, lo_m


def main():
    print(f"{'book':16s} {'n':>4s} {'R_naive':>8s} {'R_path':>8s} {'R_15m':>8s} {'R_5m':>8s} "
          f"{'tax5m/naive':>11s} {'tax5m/path':>10s}")
    rows = []
    for inst in ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]:
        df_h = load(inst)
        hm, lm = check_consistency(inst, df_h)
        df5 = pd.read_csv(f"data/{inst}_5m.csv", index_col=0, parse_dates=True)
        df5.index = pd.to_datetime(df5.index, utc=True)
        d = Data(df_h, atr_lens=(10,))
        for bname, spec0 in BOOKS[inst]:
            tr_n = [t for t in run_pattern(d, dict(spec0), cost=COSTS[inst])
                    if t["entry_time"] >= df5.index[0]]
            sp_p = dict(spec0); sp_p["path_aware"] = True
            tr_p = [t for t in run_pattern(d, sp_p, cost=COSTS[inst])
                    if t["entry_time"] >= df5.index[0]]
            tr_15 = sim_fine(df_h, pd.read_csv(f"data/{inst}_15m.csv", index_col=0,
                              parse_dates=True).assign(), spec0, COSTS[inst], bar_min=15)
            tr_5 = sim_fine(df_h, df5, spec0, COSTS[inst], bar_min=5)
            key = lambda ts: (ts.date(), ts.side if hasattr(ts, 'side') else None)
            mh = {(t["date"], t["side"]): t for t in tr_n}
            mp = {(t["date"], t["side"]): t for t in tr_p}
            m5 = {(t["date"], t["side"]): t for t in tr_5}
            common = set(mh) & set(m5)
            rn = sum(mh[k]["r"] for k in common)
            rp = sum(mp.get(k, {}).get("r", 0) for k in common)
            r5 = sum(m5[k]["r"] for k in common)
            m15 = {(t["date"], t["side"]): t for t in tr_15}
            r15 = sum(m15[k]["r"] for k in common & set(m15))
            taxn = (r5 - rn) / len(common)
            taxp = (r5 - rp) / len(common)
            name = f"{inst} {bname}"
            print(f"{name:16s} {len(common):4d} {rn:+8.2f} {rp:+8.2f} {r15:+8.2f} {r5:+8.2f} "
                  f"{taxn:+11.3f} {taxp:+10.3f}")
            rows.append(dict(inst=inst, book=bname, n=len(common), R_naive=rn, R_path=rp,
                             R_15m=r15, R_5m=r5, tax_per_trade_naive=taxn,
                             tax_per_trade_pathaware=taxp))
    out = pd.DataFrame(rows)
    out.to_csv("results/fine5m_calibration.csv", index=False)

    # 5m vs hourly aggregate consistency report
    print("\n5m-vs-hourly high/low mismatches:")
    for inst in ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]:
        df_h = load(inst)
        hm, lm = check_consistency(inst, df_h)
        print(f"  {inst}: high mismatches={hm}, low mismatches={lm}")


if __name__ == "__main__":
    main()
