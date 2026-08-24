"""DETAILED 2026 BACKTEST - final deliverable.
Layers:
  1. Trade-by-trade export for all 7 books (4 baseline + 3 new)
  2. Monthly detail per book (WR/PF/DD/reasons/hold/side split)
  3. Weekly stacked portfolio
  4. Scenarios: cost 1x/1.5x/2x x {naive, 15m-haircut-adjusted}
  5. Parameter neighborhoods for the new books (knife-edge check)
  6. Week-block bootstrap CIs + book correlation matrix
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search2 import load, COSTS
from engine import Data, run_pattern, summarize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "detailed")
os.makedirs(OUT, exist_ok=True)

SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MONTHS = [f"2026-{m:02d}" for m in range(1, 9)]

# measured 60d fine-resolution execution haircut (R units per trade, from validate_pa/fine15)
HAIRCUT_PER_TRADE = {
    ("GOLD", "BASE_A"): -0.292,
    ("GOLD", "NYJUDAS"): -0.269,
    ("EURUSD", "BASE_A"): -0.154,
    ("EURUSD", "NYJUDAS"): -0.192,
    ("GBPUSD", "BASE_A"): -0.218,
    ("GBPUSD", "NYTREND"): -0.233,
    ("USDJPY", "BASE_A"): -0.079,
}

BOOKS = {
    ("GOLD", "BASE_A"): dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                             trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                             tp=0.75, exit_hour=8, max_hold=48),
    ("GOLD", "NYJUDAS"): dict(ref_win=(7, 13), ref_off=0, act_win=(13, 17), mode="rev",
                              trigger="sweep", confirm=False, buf=0.25, atr_len=10, mult=1.0,
                              tp=1.0, exit_hour=19, max_hold=36),
    ("EURUSD", "BASE_A"): dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                               trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                               tp=0.75, exit_hour=8, max_hold=48),
    ("EURUSD", "NYJUDAS"): dict(ref_win=(7, 13), ref_off=0, act_win=(13, 17), mode="rev",
                                trigger="sweep", confirm=False, buf=0.5, atr_len=10, mult=1.0,
                                tp=1.0, exit_hour=17, max_hold=36),
    ("GBPUSD", "BASE_A"): dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                               trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                               tp=0.75, exit_hour=8, max_hold=48),
    ("GBPUSD", "NYTREND"): dict(ref_win=(7, 13), ref_off=0, act_win=(14, 17), mode="cont",
                                trigger="close_beyond", confirm=False, buf=0.0, atr_len=10,
                                mult=1.5, tp=1.5, exit_hour=20, max_hold=36),
    ("USDJPY", "BASE_A"): dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                               trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                               tp=0.75, exit_hour=8, max_hold=48),
}

NEIGHBORS = {
    ("GOLD", "NYJUDAS"): [("buf", 0.0), ("buf", 0.5), ("mult", 0.75), ("mult", 1.25),
                          ("tp", 0.75), ("tp", 1.5), ("exit_hour", 18), ("exit_hour", 20)],
    ("EURUSD", "NYJUDAS"): [("buf", 0.25), ("buf", 0.75), ("mult", 0.75), ("mult", 1.25),
                            ("tp", 0.75), ("tp", 1.5), ("exit_hour", 16), ("exit_hour", 19)],
    ("GBPUSD", "NYTREND"): [("mult", 1.0), ("mult", 2.0), ("tp", 1.0), ("tp", 2.0),
                            ("exit_hour", 19), ("exit_hour", 21),
                            ("ref_win", (6, 12)), ("ref_win", (7, 14))],
}

DATA_CACHE = {}


def get_data(inst):
    if inst not in DATA_CACHE:
        DATA_CACHE[inst] = (load(inst), Data(load(inst), atr_lens=(10,)))
    return DATA_CACHE[inst]


def run_book(inst, book, spec, cost_mult=1.0):
    df, d = get_data(inst)
    tr = run_pattern(d, spec, cost=COSTS[inst] * cost_mult)
    y = pd.DataFrame([t for t in tr if t["entry_time"] >= SPLIT])
    return y


def main():
    summary = []
    daily_books = {}
    for (inst, book), spec in BOOKS.items():
        y = run_book(inst, book, spec)
        y["month"] = y["entry_time"].dt.strftime("%Y-%m")
        y.to_csv(os.path.join(OUT, f"trades2026_{inst}_{book}.csv"), index=False)

        st = summarize(y.to_dict("records"))
        hc = HAIRCUT_PER_TRADE[(inst, book)]
        adj = st["total_r"] + hc * st["trades"]

        # monthly rows
        mrows = []
        for m in MONTHS:
            mm = y[y["month"] == m]
            s = summarize(mm.to_dict("records"))
            if s["trades"] == 0:
                continue
            hold_h = (mm["exit_time"] - mm["entry_time"]).dt.total_seconds().mean() / 3600
            mrows.append(dict(month=m, n=s["trades"], wr=s["win_rate_pct"], R=s["total_r"],
                              pf=s["profit_factor"], dd=s["max_dd_r"],
                              tp=int((mm["reason"] == "tp").sum()), sl=int((mm["reason"] == "sl").sum()),
                              time_=int((mm["reason"] == "time").sum()),
                              longs=int((mm["side"] == "long").sum()),
                              shorts=int((mm["side"] == "short").sum()),
                              hold_h=round(hold_h, 1)))
        mdf = pd.DataFrame(mrows)
        mdf.to_csv(os.path.join(OUT, f"monthly2026_{inst}_{book}.csv"), index=False)

        # daily series for correlation/bootstrap (by EXIT date)
        dd = y.groupby(y["exit_time"].dt.date)["r"].sum()
        daily_books[f"{inst}_{book}"] = dd

        summary.append(dict(inst=inst, book=book, n=st["trades"], R_naive=st["total_r"],
                            wr=st["win_rate_pct"], pf=st["profit_factor"],
                            dd_naive=st["max_dd_r"],
                            R_15m_adj=round(adj, 1)))

    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(OUT, "summary_all_books.csv"), index=False)
    print("=== ALL BOOKS 2026 (naive convention + 15m-calibrated adjustment) ===")
    print(sm.to_string(index=False))

    # ---- scenarios ----
    print("\n=== SCENARIOS (portfolio level) ===")
    base_books = ["GOLD_BASE_A", "EURUSD_BASE_A", "GBPUSD_BASE_A", "USDJPY_BASE_A"]
    scen_rows = []
    for cm, cmlabel in [(1.0, "cost1x"), (1.5, "cost1.5x"), (2.0, "cost2x")]:
        tot_n, tot_adj_n = 0.0, 0.0
        stk_n, stk_adj_n = 0.0, 0.0
        for (inst, book), spec in BOOKS.items():
            y = run_book(inst, book, spec, cost_mult=cm)
            s = summarize(y.to_dict("records"))
            hc = HAIRCUT_PER_TRADE[(inst, book)] * (1.0 if cm == 1.0 else min(cm, 1.5))
            adj = s["total_r"] + hc * s["trades"]
            if book == "BASE_A":
                tot_n += s["total_r"]; tot_adj_n += adj
            else:
                stk_n += s["total_r"]; stk_adj_n += adj
        scen_rows.append(dict(scenario=cmlabel, baseline_R=tot_n, baseline_15m_adj=tot_adj_n,
                              stacked_extra_R=stk_n, stacked_extra_adj=stk_adj_n))
    sc = pd.DataFrame(scen_rows)
    sc["stacked_total_R"] = sc["baseline_R"] + sc["stacked_extra_R"]
    sc["stacked_total_adj"] = sc["baseline_15m_adj"] + sc["stacked_extra_adj"]
    sc.to_csv(os.path.join(OUT, "scenarios.csv"), index=False)
    print(sc.round(1).to_string(index=False))

    # ---- neighborhoods ----
    print("\n=== NEIGHBORHOODS (OOS 2026 naive R; pick in bold context) ===")
    nb_rows = []
    for (inst, book), deltas in NEIGHBORS.items():
        spec0 = BOOKS[(inst, book)]
        y0 = run_book(inst, book, spec0)
        r0 = summarize(y0.to_dict("records"))["total_r"]
        for key, val in deltas:
            sp = dict(spec0); sp[key] = val
            yy = run_book(inst, book, sp)
            rr = summarize(yy.to_dict("records"))
            nb_rows.append(dict(inst=inst, book=book, param=f"{key}={val}",
                                R=rr.get("total_r", 0), n=rr.get("trades", 0)))
        nb_rows.append(dict(inst=inst, book=book, param="PICK", R=r0,
                            n=summarize(y0.to_dict("records"))["trades"]))
    nb = pd.DataFrame(nb_rows)
    nb.to_csv(os.path.join(OUT, "neighborhoods.csv"), index=False)
    for (inst, book) in NEIGHBORS:
        sub = nb[(nb["book"] == book) & (nb["inst"] == inst)]
        nbrs = sub[sub["param"] != "PICK"]
        pos = int((nbrs["R"] > 0).sum())
        print(f"{inst} {book}: {pos}/{len(nbrs)} neighbors profitable | "
              f"neighbor R range [{nbrs['R'].min():+.1f}, {nbrs['R'].max():+.1f}] | "
              f"pick {sub[sub['param']=='PICK']['R'].iloc[0]:+.1f}")

    # ---- correlations & bootstrap ----
    daily_df = pd.DataFrame(daily_books).fillna(0.0).sort_index()
    daily_df.to_csv(os.path.join(OUT, "daily_book_returns.csv"))
    corr = daily_df.corr()
    corr.to_csv(os.path.join(OUT, "book_correlations.csv"))
    print("\n=== BOOK CORRELATIONS (daily R) ===")
    print(corr.round(2).to_string())

    rng = np.random.default_rng(42)
    wk_key = [f"{pd.Timestamp(k).isocalendar().year}-W{pd.Timestamp(k).isocalendar().week:02d}"
              for k in daily_df.index]
    weekly = daily_df.groupby(pd.Series(wk_key, index=daily_df.index)).sum()
    arr = weekly.values
    n_w = len(arr)
    boot_tot, boot_dd = [], []
    base_cols = [c for c in daily_df.columns if c.endswith("BASE_A")]
    base_idx = [list(daily_df.columns).index(c) for c in base_cols]
    for _ in range(2000):
        idx = rng.integers(0, n_w, n_w)
        samp = arr[idx]
        tot = samp.sum()
        stk_eq = np.cumsum(samp.sum(axis=1))
        stk_dd = (np.maximum.accumulate(stk_eq) - stk_eq).max()
        b_eq = np.cumsum(samp[:, base_idx].sum(axis=1))
        b_dd = (np.maximum.accumulate(b_eq) - b_eq).max()
        boot_tot.append(tot); boot_dd.append(stk_dd)
    boot_tot = np.array(boot_tot); boot_dd = np.array(boot_dd)
    tot_point = arr.sum()
    print("\n=== BOOTSTRAP (week blocks, 2000 draws) ===")
    b_tot_pt = daily_df[base_cols].values.sum()
    print(f"stacked 7 books : total {tot_point:+.1f}R  95% CI "
          f"[{np.percentile(boot_tot,2.5):+.1f}, {np.percentile(boot_tot,97.5):+.1f}]")
    print(f"stacked maxDD  : median {np.percentile(boot_dd,50):.1f}R  95th pct {np.percentile(boot_dd,97.5):.1f}R")
    pd.DataFrame(dict(total_R=boot_tot)).to_csv(os.path.join(OUT, "bootstrap_totals.csv"), index=False)


if __name__ == "__main__":
    main()
