"""Robustness battery for the two v3 picks."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from engine import Data, run_pattern, summarize
from search2 import load
from fine15 import sim_fine
from opt2 import base_spec, COSTS, BROKER, SPLIT

PICKS = {
    "GOLD_NYJUDAS_v3": ("GOLD", base_spec(ref_win=(6, 13), ref_off=0, act_win=(13, 17),
                        mode="rev", trigger="sweep", buf=0.25, atr_len=10, mult=0.75,
                        tp=1.0, exit_hour=18, be_trigger=0.5, weekdays=[0, 1, 2, 3],
                        max_hold=36)),
    "USDJPY_BASE_v3": ("USDJPY", base_spec(ref_win=(19, 22), ref_off=1, act_win=(22, 9),
                       mode="rev", trigger="sweep", buf=1.25, atr_len=10, mult=1.0,
                       tp=0.75, exit_hour=6, weekdays=[0, 1, 2, 3])),
}

for name, (inst, spec) in PICKS.items():
    df = load(inst)
    d = Data(df, atr_lens=(10,))
    cost = COSTS[inst]
    tr = [t for t in run_pattern(d, dict(spec), cost=cost) if t["entry_time"] >= SPLIT]
    y = pd.DataFrame(tr)
    s = summarize(tr)
    print(f"\n{'='*64}\n{name}  |  {s['trades']} trades | {s['total_r']:+.1f}R | "
          f"PF {s['profit_factor']} | WR {s['win_rate_pct']}% | DD {s['max_dd_r']}R")
    # side split
    for side in ["long", "short"]:
        sd = summarize(y[y["side"] == side].to_dict("records"))
        print(f"  {side:5s}: n={sd.get('trades',0):3d} R={sd.get('total_r',0):+.1f} "
              f"WR={sd.get('win_rate_pct',0):.0f}% PF={sd.get('profit_factor',0)}")
    # weekday split
    y["wd"] = pd.to_datetime(y["entry_time"]).dt.day_name()
    wd = y.groupby("wd")["r"].agg(["count", "sum"]).round(1)
    print("  weekdays:"); print("   " + wd.to_string().replace("\n", "\n   "))
    # concentration: top-5 trades share of gross wins
    rs = np.sort(y["r"].values)[::-1]
    gw = rs[rs > 0].sum()
    print(f"  top-5 wins share of gross profit: {rs[:5].sum()/gw*100:.0f}%")
    # reason mix
    rm = y["reason"].value_counts()
    print(f"  exits: {dict(rm)}")
    # tax stability Jul vs Aug
    df5 = pd.read_csv(f"data/{inst}_5m.csv", index_col=0, parse_dates=True)
    df5.index = pd.to_datetime(df5.index, utc=True)
    taxes = {}
    for mo in [7, 8]:
        lo = pd.Timestamp(f"2026-{mo:02d}-01", tz="UTC")
        hi = pd.Timestamp(f"2026-{mo+1:02d}-01", tz="UTC")
        tr_h = [t for t in run_pattern(d, dict(spec), cost=cost)
                if lo <= t["entry_time"] < hi]
        tr_f = sim_fine(df, df5, dict(spec), cost, bar_min=5)
        mh = {(t["date"], t["side"]): t for t in tr_h}
        mf = {(t["date"], t["side"]): t for t in tr_f}
        common = set(mh) & set(mf)
        if common:
            taxes[mo] = ((sum(mf[k]["r"] for k in common) - sum(mh[k]["r"] for k in common))
                         / len(common), len(common))
    print(f"  monthly tax split: " + " | ".join(
        f"{m}: {v[0]:+.3f}/trade (n={v[1]})" for m, v in taxes.items()))
    # broker cost scenario
    trb = [t for t in run_pattern(d, dict(spec), cost=BROKER[inst])
           if t["entry_time"] >= SPLIT]
    sb = summarize(trb)
    print(f"  broker-cost ({BROKER[inst]}): {sb['total_r']:+.1f}R PF{sb['profit_factor']}")
    # bootstrap CI (week blocks of daily R)
    dd = y.assign(day=pd.to_datetime(y["exit_time"]).dt.date).groupby("day")["r"].sum()
    wk = pd.Series(dd.values, index=pd.to_datetime(dd.index)).groupby(
        lambda x: f"{x.isocalendar().year}-W{x.isocalendar().week:02d}").sum()
    rng = np.random.default_rng(7)
    boots = [wk.sample(len(wk), replace=True, random_state=int(rng.integers(1e9))).sum()
             for _ in range(2000)]
    print(f"  week-bootstrap total R: point {s['total_r']:+.1f}  95% CI "
          f"[{np.percentile(boots, 2.5):+.1f}, {np.percentile(boots, 97.5):+.1f}]")
