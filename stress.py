"""Stress-test the breakout family under progressively more conservative fill assumptions,
plus honest train-only selection (no test peeking) for the headline numbers."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Data, run_pattern, summarize, segment_r
from families import expand, spec_id, FAMILIES
from search import load, COSTS, SPLIT, MIN_TRADES, evaluate

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# the D-family config that won everywhere (train-selected, not test-picked):
# ref = same-session 00-04 range (off 0), action 03-08 UTC, continuation,
# buf 0.25xATR10, SL 1.0-1.5xATR, TP 2R, exit 08 or 10


def variants(inst, base_spec):
    df = load(inst)
    cost = COSTS[inst]
    d = Data(df, atr_lens=(10,))
    out = {}
    for name, mut in [
        ("A_as_traded", {}),
        ("B_no_samebar_exits", {"same_bar_exits": False}),
        ("C_next_open_entry", {"entry": "next_open"}),
        ("D_both_conservative", {"same_bar_exits": False, "entry": "next_open"}),
        ("E_double_cost", {"cost2": True}),
    ]:
        spec = dict(base_spec)
        spec.update(mut)
        c = cost * 2 if mut.get("cost2") else cost
        trades = run_pattern(d, spec, cost=c)
        y26 = [t for t in trades if t["entry_time"] >= SPLIT]
        tr = [t for t in trades if t["entry_time"] < SPLIT]
        st_t, st_y = summarize(tr), summarize(y26)
        out[name] = dict(
            train_r=st_t["total_r"], train_wr=st_t["win_rate_pct"], train_n=st_t["trades"],
            ytd_r=st_y["total_r"], ytd_wr=st_y["win_rate_pct"], ytd_n=st_y["trades"],
            ytd_pf=st_y["profit_factor"], ytd_dd=st_y["max_dd_r"],
            avg_r=st_y["avg_r"],
            tp_share=round(np.mean([t["reason"] == "tp" for t in y26]), 2),
            sl_share=round(np.mean([t["reason"] == "sl" for t in y26]), 2),
            same_bar_tp=round(np.mean([(t["exit_time"] == t["entry_time"]) and t["reason"] == "tp"
                                       for t in y26]), 2),
        )
    return pd.DataFrame(out).T


def train_only_pick(inst):
    """Select purely on TRAIN gates: top D-family config by train total_r; report its 2026."""
    df = load(inst)
    cost = COSTS[inst]
    d = Data(df, atr_lens=(10,))
    rows = []
    for fam, spec in expand({"D": FAMILIES["D_asia_breakout"]}):
        trades = run_pattern(d, spec, cost=cost)
        tr = [t for t in trades if t["entry_time"] < SPLIT]
        if len(tr) < MIN_TRADES:
            continue
        st = summarize(tr)
        segs = segment_r(tr, 3)
        rows.append(dict(spec_id=spec_id(fam, spec), trades=st["trades"], r=st["total_r"],
                         wr=st["win_rate_pct"], pf=st["profit_factor"], dd=st["max_dd_r"],
                         ok=all(x > 0 for x in segs), _spec=spec))
    g = pd.DataFrame([{k: v for k, v in r.items() if k != "_spec"} for r in rows])
    g["_s"] = [r["_spec"] for r in rows]
    gated = g[(g["ok"]) & (g["pf"] >= 1.3)].sort_values("r", ascending=False)
    top = gated.iloc[0]
    spec = gated.iloc[0]["_s"]
    # conservative variant of the train-best config
    cons = dict(spec); cons.update({"same_bar_exits": False})
    tr_c = run_pattern(d, cons, cost=cost)
    y_c = [t for t in tr_c if t["entry_time"] >= SPLIT]
    sc = summarize(y_c)
    return top, sc, spec


if __name__ == "__main__":
    base = dict(ref_win=(0, 4), ref_off=0, act_win=(3, 8), mode="cont", trigger="sweep",
                confirm=False, buf=0.25, atr_len=10, mult=1.0, tp=2.0, exit_hour=10, max_hold=36)
    print("=== Stress matrix — D_breakout base config per instrument ===")
    frames = {}
    for inst in ("GOLD", "EURUSD", "GBPUSD", "USDJPY"):
        v = variants(inst, base)
        frames[inst] = v
        mult = 1.5 if inst == "GBPUSD" else 1.0
        print(f"\n--- {inst} ---")
        print(v.round(2).to_string())
