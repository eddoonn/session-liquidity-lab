"""Final report: stacked portfolio (baseline books + validated new books), monthly table,
equity curve, and research conclusions."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search2 import load, COSTS
from engine import Data, run_pattern

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]

BASELINE_2026 = {  # from asia-gold-reversal/results/months_2026*.csv (verified reproducible)
    "GOLD":   [1.56, 2.63, 9.33, 5.48, 7.48, 3.26, 1.50, 4.55],
    "EURUSD": [3.18, 8.11, 6.26, 6.97, 2.98, 2.67, 2.95, 4.16],
    "GBPUSD": [5.65, 6.80, 5.92, 1.39, 2.41, 5.93, -0.47, 2.93],
    "USDJPY": [4.28, 7.08, 5.59, 3.73, 6.75, 1.03, 2.68, 7.36],
}

NEW_BOOKS = {
    "GOLD": dict(name="NY-Judas fade", spec=dict(ref_win=(7, 13), ref_off=0, act_win=(13, 17),
                 mode="rev", trigger="sweep", confirm=False, buf=0.25, atr_len=10, mult=1.0,
                 tp=1.0, exit_hour=19, max_hold=36)),
    "EURUSD": dict(name="NY-Judas fade", spec=dict(ref_win=(7, 13), ref_off=0, act_win=(13, 17),
                   mode="rev", trigger="sweep", confirm=False, buf=0.5, atr_len=10, mult=1.0,
                   tp=1.0, exit_hour=17, max_hold=36)),
    "GBPUSD": dict(name="NY-trend continuation", spec=dict(ref_win=(7, 13), ref_off=0,
                   act_win=(14, 17), mode="cont", trigger="close_beyond", confirm=False,
                   buf=0.0, atr_len=10, mult=1.5, tp=1.5, exit_hour=20, max_hold=36)),
}


def main():
    rows_monthly = {}
    trades_all = {}
    for inst in SYMBOLS_ALL:
        df = load(inst)
        d = Data(df, atr_lens=(10,))
        base_m = BASELINE_2026[inst]
        rows_monthly[(inst, "baseline")] = base_m
        if inst in NEW_BOOKS:
            spec = NEW_BOOKS[inst]["spec"]
            tr = [t for t in run_pattern(d, spec, cost=COSTS[inst])
                  if t["entry_time"] >= SPLIT]
            mdf = pd.DataFrame(tr)
            mdf["month"] = mdf["entry_time"].dt.strftime("%Y-%m")
            mv = [round(mdf[mdf["month"] == m]["r"].sum(), 2) for m in MONTHS]
            rows_monthly[(inst, NEW_BOOKS[inst]["name"])] = mv
            trades_all[inst] = mdf

        print(f"{inst}: baseline {sum(base_m):+.2f}R"
              + (f" | new book {sum(rows_monthly[(inst, NEW_BOOKS[inst]['name'])]):+.2f}R"
                 if inst in NEW_BOOKS else ""))

    # ---- stacked portfolio table ----
    print("\n=== STACKED PORTFOLIO 2026 (baseline + new books, $100 risk/book) ===")
    header = ["Month"] + [f"{i}_{NEW_BOOKS[i]['name'].split()[0]}" if i in NEW_BOOKS else i
                          for i in SYMBOLS_ALL] + ["TOTAL", "$@100"]
    lines = ["\t".join(header)]
    totals = np.zeros(len(MONTHS))
    for k, m in enumerate(MONTHS):
        vals, row = [], []
        for inst in SYMBOLS_ALL:
            v = BASELINE_2026[inst][k]
            if inst in NEW_BOOKS:
                v += rows_monthly[(inst, NEW_BOOKS[inst]["name"])][k]
            vals.append(v)
            row.append(f"{v:+.2f}")
        totals[k] = sum(vals)
        n_trades = ""
        lines.append("\t".join([m] + row + [f"{totals[k]:+.2f}", f"${totals[k]*400:,.0f}"]))
    lines.append("YTD\t" + "\t".join("") + f"\tTOTAL YTD: {totals.sum():+.2f}R -> ${totals.sum()*400:,.0f}")
    print("\n".join(lines))

    # ---- equity curve: baseline portfolio vs stacked ----
    base_tot = np.array([sum(BASELINE_2026[i][k] for i in SYMBOLS_ALL) for k in range(8)])
    eq_b = np.concatenate([[0], np.cumsum(base_tot)])
    eq_s = np.concatenate([[0], np.cumsum(totals)])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(range(9), eq_b, marker="o", label=f"Baseline Asia-NY x4 ({eq_b[-1]:+.1f}R)")
    ax.plot(range(9), eq_s, marker="o", label=f"Stacked + new books ({eq_s[-1]:+.1f}R)")
    ax.fill_between(range(9), eq_b, eq_s, alpha=0.15, color="green")
    ax.set_xticks(range(9)); ax.set_xticklabels(["start"] + [m[5:] for m in MONTHS])
    ax.axhline(0, color="gray", lw=1); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylabel("Cumulative R"); ax.set_title("2026 YTD - baseline vs stacked portfolio (4 instruments)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "portfolio_stacked.png"), dpi=150)
    print(f"\nsaved {os.path.join(OUT, 'portfolio_stacked.png')}")

    pd.DataFrame([{**{"inst": i, "book": "baseline"},
                    **{m: v for m, v in zip(MONTHS, BASELINE_2026[i])}} for i in SYMBOLS_ALL]
                 + [{**{"inst": i, "book": NEW_BOOKS[i]["name"]},
                     **{m: v for m, v in zip(MONTHS, rows_monthly[(i, NEW_BOOKS[i]["name"])])}}
                    for i in NEW_BOOKS]).to_csv(os.path.join(OUT, "stacked_monthly.csv"), index=False)


SYMBOLS_ALL = ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]

if __name__ == "__main__":
    main()
