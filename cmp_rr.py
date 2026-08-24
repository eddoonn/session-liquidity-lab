import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from search2 import load
from engine import Data, run_pattern

df = load("EURUSD")
d = Data(df, atr_lens=(10,))
for rr in (0.75, 1.0):
    for cost in (0.0002, 0.00005, 0.0):
        spec = dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                    trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                    tp=rr, exit_hour=8, max_hold=48)
        tr = [t for t in run_pattern(d, spec, cost=cost)
              if t["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")]
        m = pd.DataFrame(tr)
        m["mo"] = m["entry_time"].dt.strftime("%Y-%m")
        tot = round(m["r"].sum(), 2)
        monthly = [round(x, 1) for x in m.groupby("mo")["r"].sum()]
        print(f"rr={rr} cost={cost}: n={len(tr)} YTD={tot:+.2f} monthly={monthly}")
print("THEIRS:            n=125 YTD=+37.28 monthly=[3.2, 8.1, 6.3, 7.0, 3.0, 2.7, 3.0, 4.2]")
