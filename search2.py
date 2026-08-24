"""Honest walk-forward search v2:
- corrected fill model (contiguous-gap stops fill at open; holes fill at level)
- headline picks chosen ONLY on pre-2026 train data
- 2026 YTD reported as untouched OOS, plus conservative stress variants
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Data, run_pattern, summarize, segment_r
from families import expand, spec_id, FAMILIES
from download_data import SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COSTS = {"GOLD": 0.35, "EURUSD": 0.00020, "GBPUSD": 0.00028, "USDJPY": 0.024}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MIN_TRADES = 60


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_60m.csv"), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_specs = list(expand())
    print(f"{len(all_specs)} configs/instrument | split at {SPLIT.date()} | costs {COSTS}\n")

    picks = []
    for inst in SYMBOLS:
        df = load(inst)
        cost = COSTS[inst]
        d = Data(df, atr_lens=(10,))
        rows, store = [], {}
        for fam, spec in all_specs:
            sid = spec_id(fam, spec)
            trades = run_pattern(d, spec, cost=cost)
            tr = [t for t in trades if t["entry_time"] < SPLIT]
            if len(tr) < MIN_TRADES:
                continue
            st = summarize(tr)
            segs = segment_r(tr, 3)
            te = [t for t in trades if t["entry_time"] >= SPLIT]
            ts = summarize(te)
            rows.append(dict(inst=inst, family=fam.replace("_asia_breakout", "").replace("_rev", ""),
                             spec=sid, n_tr=st["trades"], tr_r=st["total_r"], tr_wr=st["win_rate_pct"],
                             tr_pf=st["profit_factor"], tr_dd=st["max_dd_r"],
                             segs=" / ".join(f"{x:+.1f}" for x in segs),
                             seg_ok=all(x > 0 for x in segs),
                             te_n=ts.get("trades", 0), te_r=ts.get("total_r", 0.0),
                             te_pf=ts.get("profit_factor", 0)))
        g = pd.DataFrame(rows)
        gated = g[(g["seg_ok"]) & (g["tr_pf"] >= 1.3) & (g["tr_dd"] <= 8)].sort_values("tr_r", ascending=False)
        print(f"=== {inst}: {len(g)} evaluated -> {len(gated)} gated ===")
        base_row = gated[gated["spec"].str.startswith("A_")]
        if not base_row.empty:
            b = base_row.iloc[0]
            print(f"baseline-A best-by-train: tr {b['tr_r']:+.1f}R -> OOS {b['te_r']:+.1f}R ({b['te_n']} trades)")
        med = gated[~gated["spec"].str.startswith("A_")]["te_r"].median()
        print(f"gated non-baseline candidates: {len(gated)-len(base_row)} | median OOS {med:+.1f}R")
        top3 = gated.head(3)
        print(top3[["family", "spec", "n_tr", "tr_r", "tr_pf", "te_r", "te_n", "te_pf"]]
              .to_string(index=False, max_colwidth=64))

        # ---- headline pick: TRAIN-BEST only (no test peeking) ----
        win = gated.iloc[0]
        sid = win["spec"]
        for fam, spec in all_specs:
            if spec_id(fam, spec) == sid:
                W = spec
                break
        trades = run_pattern(d, W, cost=cost)
        y26 = [t for t in trades if t["entry_time"] >= SPLIT]
        s26 = summarize(y26)
        cons = dict(W); cons["same_bar_exits"] = False
        y_c = [t for t in run_pattern(d, cons, cost=cost) if t["entry_time"] >= SPLIT]
        s_c = summarize(y_c)
        y_dc = [t for t in run_pattern(d, dict(W), cost=cost * 2) if t["entry_time"] >= SPLIT]
        s_dc = summarize(y_dc)
        m = pd.DataFrame(y26); m["month"] = m["entry_time"].dt.strftime("%Y-%m")
        mo = m.groupby("month")["r"].agg(["count", "sum"]).round(2)
        qs = segment_r(y26, 4) if len(y26) >= 8 else []
        print(f"\nPICK {inst}: {sid}")
        print(mo.to_string())
        print(f"OOS 2026: {s26['total_r']:+.1f}R/{s26['trades']}t PF{s26['profit_factor']} "
              f"WR{s26['win_rate_pct']}% DD{s26['max_dd_r']}R qtrs: {' / '.join(f'{x:+.1f}' for x in qs)}")
        print(f"stress: noSameBar {s_c['total_r']:+.1f}R | doubleCost {s_dc['total_r']:+.1f}R\n")
        picks.append(dict(inst=inst, spec=sid, oos_r=s26["total_r"], oos_n=s26["trades"],
                          oos_pf=s26["profit_factor"], oos_wr=s26["win_rate_pct"], oos_dd=s26["max_dd_r"],
                          cons_r=s_c["total_r"], cost2_r=s_dc["total_r"]))
        g.to_csv(os.path.join(OUT_DIR, f"all_configs_{inst}.csv"), index=False)

    fin = pd.DataFrame(picks)
    fin.to_csv(os.path.join(OUT_DIR, "picks_v2.csv"), index=False)
    print("==== HEADLINE PICKS (train-selected) ====")
    print(fin[["inst", "oos_r", "oos_n", "oos_pf", "oos_wr", "oos_dd", "cons_r", "cost2_r"]].to_string(index=False))
    print(f"\nPORTFOLIO OOS 2026 YTD: {fin['oos_r'].sum():+.1f}R over {fin['oos_n'].sum()} trades")


if __name__ == "__main__":
    main()
