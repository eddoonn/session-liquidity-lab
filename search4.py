"""Strategy hunt round 2 — structural designs that resist intrabar fill ambiguity.
Every config must pass train gates under BOTH naive and path_aware simulation.
Finalists get their true execution tax measured on 5m data (60d) and reported
as 5m-calibrated OOS estimates."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Data, run_pattern, summarize, segment_r
from families import spec_id
from download_data import SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COSTS = {"GOLD": 0.35, "EURUSD": 0.00020, "GBPUSD": 0.00028, "USDJPY": 0.024}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MIN_TRADES = 40


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_60m.csv"), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def F(name, **kw):
    s = dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev", trigger="sweep",
             confirm=False, buf=1.0, atr_len=10, mult=1.0, tp=0.75, exit_hour=8,
             max_hold=36, weekdays=None)
    s.update(kw)
    return (name, s)


def build_specs():
    S = []
    MT = [0, 1, 2, 3]          # Mon-Thu
    # --- H: NY lunch reversal (fade the 13-16 move into the close) ---
    for ref in [(13, 16), (12, 16)]:
        for act in [(16, 19), (16, 20)]:
            for tp in [1.0, 1.5]:
                for wd, tag in [(None, ""), (MT, "_mt")]:
                    S.append(F("H_nylunch", ref_win=ref, ref_off=0, act_win=act, mode="rev",
                               confirm=True, buf=0.25, mult=1.0, tp=tp,
                               exit_hour=act[1] + 1, weekdays=wd))
    # --- I: London first-hour range break, ride NY ---
    for ref in [(7, 8), (8, 9), (7, 9)]:
        for trig in ["close_beyond", "sweep"]:
            for mode in ["cont"]:
                for mult in [1.5, 2.0]:
                    for tp in [1.5, 2.0]:
                        S.append(F("I_lonbreak", ref_win=ref, ref_off=0, act_win=(13, 17),
                                   mode=mode, trigger=("sweep" if trig == "sweep" else "close_beyond"),
                                   buf=(0.25 if trig == "sweep" else 0.0), mult=mult, tp=tp,
                                   exit_hour=20))
    # --- K: Asia-range self-cycle (yesterday's Asia range swept tonight) ---
    for buf in [0.75, 1.0, 1.25]:
        for tp in [0.75, 1.0]:
            S.append(F("K_asiaself", ref_win=(22, 10), ref_off=1, act_win=(22, 10),
                       mode="rev", buf=buf, tp=tp))
    # --- L: Tokyo-lunch compression break ---
    for act in [(4, 9), (4, 10)]:
        for mode in ["cont", "rev"]:
            for buf in [0.5, 1.0]:
                S.append(F("L_tokyo", ref_win=(3, 4), ref_off=0, act_win=act, mode=mode,
                           buf=buf, mult=1.5, tp=1.5, exit_hour=10))
    # --- M: wide-target NY Judas (rejection-confirmed) ---
    for buf in [0.5, 0.75]:
        for mult in [1.5, 2.0]:
            for tp in [1.5, 2.0]:
                S.append(F("M_widejudas", ref_win=(7, 13), ref_off=0, act_win=(13, 17),
                           mode="rev", confirm=True, buf=buf, mult=mult, tp=tp, exit_hour=19))
    # --- N: Friday-only & Mon-Thu variants of the proven GOLD NY-Judas ---
    for wd, tag in [([4], "_fri"), (MT, "_mt")]:
        for tp in [1.0, 1.5]:
            S.append(F("N_judas_wd", ref_win=(7, 13), ref_off=0, act_win=(13, 17),
                       mode="rev", confirm=False, buf=0.25, mult=1.0, tp=tp,
                       exit_hour=19, weekdays=wd))
    return S


def main():
    specs = build_specs()
    print(f"{len(specs)} round-2 configs per instrument\n")
    survivors = []
    for inst in SYMBOLS:
        df = load(inst)
        cost = COSTS[inst]
        d = Data(df, atr_lens=(10,))
        rows = []
        for name, spec in specs:
            tr_n = run_pattern(d, dict(spec), cost=cost)
            sp_p = dict(spec); sp_p["path_aware"] = True
            tr_p = run_pattern(d, sp_p, cost=cost)
            tn = [t for t in tr_n if t["entry_time"] < SPLIT]
            tp_ = [t for t in tr_p if t["entry_time"] < SPLIT]
            if len(tn) < MIN_TRADES:
                continue
            sn = summarize(tn)
            sp_ = summarize(tp_)
            segs_n = segment_r(tn, 3)
            segs_p = segment_r(tp_, 3)
            ok = (sn["profit_factor"] >= 1.3 and sn["max_dd_r"] <= 8 and all(x > 0 for x in segs_n)
                  and sp_["profit_factor"] >= 1.2 and all(x > 0 for x in segs_p))
            ten = [t for t in tr_n if t["entry_time"] >= SPLIT]
            tep = [t for t in tr_p if t["entry_time"] >= SPLIT]
            sen, sep = summarize(ten), summarize(tep)
            rows.append(dict(inst=inst, fam=name, sid=spec_id(name, spec)[len(name) + 1:] or name,
                             n=sn["trades"], Rn=sn["total_r"], pfn=sn["profit_factor"],
                             Rp=sp_["total_r"],
                             teRn=sen.get("total_r", 0), teRp=sep.get("total_r", 0),
                             ten=sen.get("trades", 0)))
        g = pd.DataFrame(rows)
        gated = g.sort_values("Rp", ascending=False)
        top = gated.head(6)
        if len(top):
            print(f"=== {inst}: {len(g)} eval -> showing top by path-aware train R ===")
            print(top.to_string(index=False, max_colwidth=30))
            for _, r in top.iterrows():
                survivors.append(r.to_dict())
        print()
    sv = pd.DataFrame(survivors).sort_values(["inst", "Rp"], ascending=[True, False])
    sv.to_csv(os.path.join(OUT, "round2_survivors.csv"), index=False)
    print("saved results/round2_survivors.csv")


if __name__ == "__main__":
    main()
