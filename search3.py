"""Search v3 — STRICT execution mode baked into every config:
entry = next bar open after signal bar CLOSES (no resting-order fill assumptions),
exits evaluated only from the bar AFTER entry (no same-bar SL/TP),
costs doubled vs v2 for extra margin.
Selection strictly on train (<2026-01-01); 2026 YTD untouched.
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
MIN_TRADES = 50

EXTRA_FAMILIES = {
    # H: Monday-range breakout Tue/Wed (weekly structure momentum)
    "H_mon_range_cont": dict(
        refs=[((0, 23), 1)],
        acts=[(6, 10), (13, 16)],
        modes=["cont", "rev"], triggers=["sweep"], confirms=[False],
        bufs=[0.25, 0.5], atr_lens=[10], mults=[1.0, 1.5],
        tps=[1.0, 1.5, 2.0], exits=[10, 16, 19],
    ),
    # I: NY session sweeps prior-day NY-late extremes (24h-old liquidity still hunted)
    "I_ny_prevday_rev": dict(
        refs=[((19, 21), 1), ((19, 22), 1)],
        acts=[(13, 16), (13, 17)],
        modes=["rev"], triggers=["sweep"], confirms=[False],
        bufs=[0.25, 0.5, 0.75, 1.0], atr_lens=[10], mults=[1.0, 1.5],
        tps=[0.75, 1.0, 1.5, "opp"], exits=[17, 19, 21],
    ),
}


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_60m.csv"), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fam_all = dict(FAMILIES); fam_all.update(EXTRA_FAMILIES)
    specs = []
    for fam, sp in expand(fam_all):
        sp["entry"] = "next_open"          # <- strict: market-in on next open
        sp["same_bar_exits"] = False       # <- strict: exits start next bar
        specs.append((fam, sp))
    print(f"{len(specs)} strict-mode configs/instrument | costs x2\n")

    picks = []
    for inst in SYMBOLS:
        df = load(inst)
        cost = COSTS[inst] * 2
        d = Data(df, atr_lens=(10,))
        rows, store = [], {}
        for fam, spec in specs:
            sid = spec_id(fam, spec)
            trades = run_pattern(d, spec, cost=cost)
            tr = [t for t in trades if t["entry_time"] < SPLIT]
            if len(tr) < MIN_TRADES:
                continue
            st = summarize(tr)
            segs = segment_r(tr, 3)
            te = [t for t in trades if t["entry_time"] >= SPLIT]
            ts = summarize(te)
            rows.append(dict(inst=inst, family=fam, spec=sid,
                             n_tr=st["trades"], tr_r=st["total_r"], tr_pf=st["profit_factor"],
                             tr_dd=st["max_dd_r"], seg_ok=all(x > 0 for x in segs),
                             te_n=ts.get("trades", 0), te_r=ts.get("total_r", 0.0),
                             te_pf=ts.get("profit_factor", 0)))
        g = pd.DataFrame(rows)
        gated = g[(g["seg_ok"]) & (g["tr_pf"] >= 1.4) & (g["tr_dd"] <= 7)].sort_values("tr_r", ascending=False)
        base_row = gated[gated["spec"].str.startswith("A_asia")]
        btxt = ""
        if not base_row.empty:
            b = base_row.iloc[0]
            btxt = f" | baseline-A strict: tr {b['tr_r']:+.1f} -> oos {b['te_r']:+.1f}"
        print(f"=== {inst}: {len(g)} -> {len(gated)} gated{btxt}")
        if gated.empty:
            continue
        print(gated.head(4)[["family", "spec", "n_tr", "tr_r", "tr_pf", "te_r", "te_n", "te_pf"]]
              .to_string(index=False, max_colwidth=62))

        win = gated.iloc[0]
        sid = win["spec"]
        W = next(sp for f2, sp in specs if spec_id(f2, sp) == sid)
        y26 = [t for t in run_pattern(d, W, cost=cost) if t["entry_time"] >= SPLIT]
        s26 = summarize(y26)
        m = pd.DataFrame(y26); m["month"] = m["entry_time"].dt.strftime("%Y-%m")
        mo = m.groupby("month")["r"].agg(["count", "sum"]).round(2)
        qs = segment_r(y26, 4) if len(y26) >= 8 else []
        print(f"PICK {inst}: {sid}")
        print(mo.to_string())
        print(f"OOS: {s26['total_r']:+.1f}R/{s26['trades']}t PF{s26['profit_factor']} "
              f"WR{s26['win_rate_pct']}% DD{s26['max_dd_r']} qtrs {' / '.join(f'{x:+.1f}' for x in qs)}\n")
        picks.append(dict(inst=inst, spec=sid, fam=win["family"], oos_r=s26["total_r"],
                          oos_n=s26["trades"], oos_pf=s26["profit_factor"], oos_wr=s26["win_rate_pct"]))
        g.to_csv(os.path.join(OUT_DIR, f"strict_configs_{inst}.csv"), index=False)

    fin = pd.DataFrame(picks)
    fin.to_csv(os.path.join(OUT_DIR, "picks_strict.csv"), index=False)
    print("==== STRICT-MODE PICKS ====")
    print(fin.to_string(index=False))
    print(f"\nSTRICT PORTFOLIO 2026 YTD: {fin['oos_r'].sum():+.1f}R over {fin['oos_n'].sum()} trades")


if __name__ == "__main__":
    main()
