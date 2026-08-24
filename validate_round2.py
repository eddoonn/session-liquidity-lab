"""Validate round-2 finalists against 5m ground truth; produce calibrated OOS table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from engine import Data, run_pattern, summarize
from search4 import load, build_specs, SPLIT, COSTS
from fine15 import sim_fine

FINALISTS = {
    # per instrument: (ref_win, mult, tp) triples picked from train-ranked survivors
    "GOLD":   [((8, 9), 2.0, 1.5), ((8, 9), 1.5, 2.0), ((7, 9), 2.0, 1.5)],
    "EURUSD": [((7, 8), 1.5, 2.0), ((8, 9), 1.5, 2.0), ((8, 9), 1.5, 1.5)],
    "GBPUSD": [((7, 9), 1.5, 2.0), ((7, 9), 1.5, 1.5), ((7, 8), 1.5, 2.0)],
    "USDJPY": [((8, 9), 2.0, 1.5), ((7, 9), 2.0, 1.5), ((8, 9), 1.5, 2.0)],
}

rows = []
for inst, cands in FINALISTS.items():
    df = load(inst)
    df5 = pd.read_csv(f"data/{inst}_5m.csv", index_col=0, parse_dates=True)
    df5.index = pd.to_datetime(df5.index, utc=True)
    d = Data(df, atr_lens=(10,))
    cost = COSTS[inst]
    for ref, mult, tp in cands:
        spec = dict(ref_win=ref, ref_off=0, act_win=(13, 17), mode="cont",
                    trigger="close_beyond", confirm=False, buf=0.0, atr_len=10,
                    mult=mult, tp=tp, exit_hour=20, max_hold=36)
        tr_n = run_pattern(d, dict(spec), cost=cost)
        tn = [t for t in tr_n if t["entry_time"] < SPLIT]
        y26 = pd.DataFrame([t for t in tr_n if t["entry_time"] >= SPLIT])
        st26 = summarize(y26.to_dict("records"))
        tr_f = sim_fine(df, df5, spec, cost, bar_min=5)
        m26 = {(t["date"], t["side"]): t for t in tr_f}
        common = [(t["date"], t["side"]) for t in
                  [x for x in tr_n if x["entry_time"] >= df5.index[0]]
                  if (t[0], t[1]) in m26] if False else None
        hn = {(t["date"], t["side"]): t for t in tr_n if t["entry_time"] >= df5.index[0]}
        common = set(hn) & set(m26)
        r_h = sum(hn[k]["r"] for k in common)
        r_f = sum(m26[k]["r"] for k in common)
        tax = (r_f - r_h) / len(common)
        calib = st26["total_r"] + tax * st26["trades"]
        rows.append(dict(inst=inst, ref=f"{ref[0]:02d}-{ref[1]:02d}", mult=mult, tp=tp,
                         ytd_n=st26["trades"], R_naive=st26["total_r"],
                         wr=st26["win_rate_pct"], pf=st26["profit_factor"],
                         dd=st26["max_dd_r"],
                         ov_n=len(common), R_ov_h=r_h, R_ov_5m=r_f,
                         tax_per_trade=round(tax, 3), R_calibrated=round(calib, 1)))
res = pd.DataFrame(rows)
res.to_csv("results/round2_5m_validation.csv", index=False)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
