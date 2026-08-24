import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from search2 import load
from engine import Data, run_pattern, summarize
from opt2 import base_spec, COSTS

PICKS = {
    "GOLD_NYJUDAS_v3": ("GOLD", base_spec(ref_win=(6, 13), ref_off=0, act_win=(13, 17),
                        mode="rev", buf=0.25, atr_len=10, mult=0.75, tp=1.0,
                        exit_hour=18, be_trigger=0.5, weekdays=[0, 1, 2, 3], max_hold=36)),
    "USDJPY_BASE_v3": ("USDJPY", base_spec(ref_win=(19, 22), ref_off=1, act_win=(22, 9),
                       mode="rev", buf=1.25, atr_len=10, mult=1.0, tp=0.75,
                       exit_hour=6, weekdays=[0, 1, 2, 3])),
}
series = {}
rows = []
for name, (inst, spec) in PICKS.items():
    df = load(inst)
    d = Data(df, atr_lens=(10,))
    tr = [t for t in run_pattern(d, dict(spec), cost=COSTS[inst])
          if t["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")]
    y = pd.DataFrame(tr)
    y.to_csv(f"results/detailed/trades2026_{name}.csv", index=False)
    s = summarize(tr)
    series[name] = y.assign(day=pd.to_datetime(y["exit_time"]).dt.date).groupby("day")["r"].sum()
    rows.append(dict(book=name, trades=s["trades"], R=s["total_r"], pf=s["profit_factor"],
                     wr=s["win_rate_pct"], dd=s["max_dd_r"],
                     months_pos=int((y.assign(m=y["entry_time"].dt.strftime("%Y-%m"))
                                     .groupby("m")["r"].sum() > 0).sum())))
tot = pd.concat(series, axis=1).fillna(0)
eq = tot.sum(axis=1).cumsum()
dd = (eq.cummax() - eq).max()
rows.append(dict(book="COMBINED", trades=int(tot.abs().gt(0).any(axis=1).sum()),
                 R=round(tot.sum().sum(), 1), pf=np.nan, wr=np.nan, dd=round(dd, 1),
                 months_pos=np.nan))
out = pd.DataFrame(rows)
out.to_csv("results/final_portfolio_v3.csv", index=False)
print(out.to_string(index=False))
print(f"\ncombined daily-R positive days: {(tot.sum(axis=1) > 0).sum()} / "
      f"{(tot.sum(axis=1) != 0).sum()}")
