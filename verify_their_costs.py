import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from search2 import load
from engine import Data, run_pattern, summarize, segment_r

# Asia-NY master cost table (analyze_2026_final.py)
THEIR_COSTS = {"GOLD": 0.5, "USDJPY": 0.005}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")

PICKS = {
    "GOLD_NYJUDAS_v3": ("GOLD", base_spec := dict(ref_win=(6, 13), ref_off=0,
        act_win=(13, 17), mode="rev", trigger="sweep", confirm=False, buf=0.25,
        atr_len=10, mult=0.75, tp=1.0, exit_hour=18, be_trigger=0.5,
        weekdays=[0, 1, 2, 3], max_hold=36)),
    "USDJPY_BASE_v3": ("USDJPY", dict(ref_win=(19, 22), ref_off=1, act_win=(22, 9),
        mode="rev", trigger="sweep", confirm=False, buf=1.25, atr_len=10, mult=1.0,
        tp=0.75, exit_hour=6, be_trigger=None, weekdays=[0, 1, 2, 3], max_hold=48)),
}
for name, (inst, spec) in PICKS.items():
    df = load(inst)
    d = Data(df, atr_lens=(10,))
    tr = [t for t in run_pattern(d, dict(spec), cost=THEIR_COSTS[inst])
          if t["entry_time"] >= SPLIT]
    s = summarize(tr)
    qs = segment_r(tr, 4)
    m = pd.DataFrame(tr); m["mo"] = m["entry_time"].dt.strftime("%Y-%m")
    neg = int((m.groupby("mo")["r"].sum() < 0).sum())
    print(f"{name} @their cost {THEIR_COSTS[inst]}: n={s['trades']} R={s['total_r']:+.1f} "
          f"PF={s['profit_factor']} WR={s['win_rate_pct']}% DD={s['max_dd_r']}R "
          f"neg_months={neg} quarters={' / '.join(f'{q:+.1f}' for q in qs)}")
