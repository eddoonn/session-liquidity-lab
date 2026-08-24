import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from search2 import load
from engine import Data, run_pattern, summarize, segment_r
from opt import base_spec

df_g = load("GOLD"); df_j = load("USDJPY")
d_g = Data(df_g, atr_lens=(10,)); d_j = Data(df_j, atr_lens=(10,))
MT = [0, 1, 2, 3]
picks = {
 "GOLD_NYJUDAS_v2": (d_g, 0.35, base_spec(ref_win=(6, 13), ref_off=0, act_win=(13, 17),
                      buf=0.25, mult=0.75, tp=1.0, exit_hour=18, be_trigger=0.6,
                      weekdays=MT, max_hold=36)),
 "USDJPY_BASE_v2": (d_j, 0.024, base_spec(ref_win=(19, 22), act_win=(22, 10), buf=1.25,
                    mult=1.0, tp=0.75, exit_hour=7, weekdays=MT)),
 "GOLD_BASE_v2": (d_g, 0.35, base_spec(ref_win=(19, 22), act_win=(22, 10), buf=1.25,
                  mult=1.0, tp=1.0, exit_hour=7, be_trigger=0.6, weekdays=MT)),
}
series = {}
for name, (d, cost, spec) in picks.items():
    tr = [t for t in run_pattern(d, dict(spec), cost=cost)
          if t["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")]
    s = summarize(tr); qs = segment_r(tr, 4)
    m = pd.DataFrame(tr); m["mo"] = m["entry_time"].dt.strftime("%Y-%m")
    monthly = [round(x, 1) for x in m.groupby("mo")["r"].sum()]
    print(f"{name}: n={s['trades']} R={s['total_r']:+.1f} PF={s['profit_factor']} "
          f"WR={s['win_rate_pct']}% DD={s['max_dd_r']}R")
    print(f"   quarters: {' / '.join(f'{q:+.1f}' for q in qs)}")
    print(f"   months:   {monthly}")
    series[name] = pd.DataFrame(tr).assign(
        d=lambda x: pd.to_datetime(x["exit_time"]).dt.date).groupby("d")["r"].sum()
c = pd.concat([series["GOLD_NYJUDAS_v2"], series["GOLD_BASE_v2"]], axis=1).fillna(0)
print(f"\ndaily-R correlation between the two GOLD books: {c.corr().iloc[0, 1]:+.2f}")
tot = pd.concat(series, axis=1).fillna(0).sum(axis=1)
eq = tot.cumsum()
print(f"combined 3 books: {tot.sum():+.1f}R | maxDD {(eq.cummax()-eq).max():.1f}R | "
      f"positive days {(tot>0).sum()}/{(tot!=0).sum()}")
