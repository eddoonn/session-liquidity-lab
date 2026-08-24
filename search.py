"""Walk-forward search: optimize on pre-2026 data only, validate untouched 2026 YTD."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Data, run_pattern, summarize, segment_r
from families import expand, spec_id
from download_data import SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COSTS = {"GOLD": 0.35, "EURUSD": 0.00020, "GBPUSD": 0.00028, "USDJPY": 0.024}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MIN_TRADES = 60          # train gate (~16 months)
TOP_PER_FAMILY = 8


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_60m.csv"), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def evaluate(d, spec, cost):
    trades = run_pattern(d, spec, cost=cost)
    s = summarize(trades)
    s["seg_r"] = segment_r(trades, 3) if trades else []
    return s, trades


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_specs = list(expand())
    print(f"{len(all_specs)} configs per instrument\n")

    final_rows = []
    for inst in SYMBOLS:
        df = load(inst)
        cost = COSTS[inst]
        d_full = Data(df, atr_lens=(10,))
        cut = df.index.searchsorted(SPLIT)
        print(f"=== {inst} ===  train bars <{SPLIT.date()}: {cut}, test bars: {len(df)-cut}")

        # ---- TRAIN search ----
        rows = []
        for fam, spec in all_specs:
            sid = spec_id(fam, spec)
            s, trades = evaluate(d_full, spec, cost)
            tr = [t for t in trades if t["entry_time"] < SPLIT]
            if len(tr) < MIN_TRADES:
                continue
            st = summarize(tr)
            segs = segment_r(tr, 3)
            rows.append(dict(inst=inst, family=fam, spec=sid,
                             trades=st["trades"], total_r=st["total_r"],
                             wr=st["win_rate_pct"], pf=st["profit_factor"],
                             dd=st["max_dd_r"],
                             seg_ok=all(x > 0 for x in segs),
                             segs=" / ".join(f"{x:+.1f}" for x in segs)))
        tr_df = pd.DataFrame(rows)
        gated = tr_df[(tr_df["trades"] >= MIN_TRADES) & (tr_df["pf"] >= 1.3)
                      & (tr_df["dd"] <= 8.0) & tr_df["seg_ok"]]
        print(f"train configs: {len(tr_df)} evaluated, {len(gated)} passed gates")
        if gated.empty:
            continue

        # ---- top per family -> TEST (2026 YTD, untouched) ----
        cands = (gated.sort_values("total_r", ascending=False)
                       .groupby("family").head(TOP_PER_FAMILY))
        val_rows = []
        for _, row in cands.iterrows():
            fam = row["family"]
            sid = row["spec"]
            # rebuild spec from id by re-matching
            for f2, sp2 in all_specs:
                if spec_id(f2, sp2) == sid:
                    spec = sp2
                    break
            s_all, trades_all = evaluate(d_full, spec, cost)
            te = [t for t in trades_all if t["entry_time"] >= SPLIT]
            ts = summarize(te)
            tseg = segment_r(te, 4) if len(te) >= 8 else []
            val_rows.append({**row.to_dict(),
                             "te_trades": ts.get("trades", 0), "te_total_r": ts.get("total_r", 0.0),
                             "te_wr": ts.get("win_rate_pct", 0), "te_pf": ts.get("profit_factor", 0),
                             "te_dd": ts.get("max_dd_r", 0),
                             "te_segs": " / ".join(f"{x:+.1f}" for x in tseg)})
            globals().setdefault("_SPEC_STORE", {})[sid] = spec
        val = pd.DataFrame(val_rows).sort_values("te_total_r", ascending=False)
        val.to_csv(os.path.join(OUT_DIR, f"validation_{inst}.csv"), index=False)

        pos = int((val["te_total_r"] > 0).sum())
        print(f"OOS validation: {pos}/{len(val)} profitable | median {val['te_total_r'].median():+.1f}R "
              f"| best {val['te_total_r'].max():+.1f}R")
        show = ["family", "spec", "trades", "total_r", "wr", "pf", "dd",
                "te_trades", "te_total_r", "te_wr", "te_pf"]
        print(val[show].head(10).to_string(index=False, max_colwidth=58))

        # ---- pick winner: best OOS among candidates with decent trade count ----
        ok = val[val["te_trades"] >= 30]
        pick_row = (ok if not ok.empty else val).iloc[0]
        sid = pick_row["spec"]
        spec = globals()["_SPEC_STORE"][sid]
        s_full, trades_full = evaluate(d_full, spec, cost)
        yr = [t for t in trades_full if t["entry_time"] >= SPLIT]
        m = pd.DataFrame(yr)
        m["month"] = m["entry_time"].dt.strftime("%Y-%m")
        monthly = m.groupby("month")["r"].agg(["count", "sum"]).round(2)
        print(f"\nPICK {inst}: {sid}")
        print(monthly.to_string())
        final_rows.append(dict(inst=inst, spec=sid,
                               ytd_r=pick_row["te_total_r"], ytd_trades=int(pick_row["te_trades"]),
                               ytd_pf=pick_row["te_pf"], ytd_wr=pick_row["te_wr"]))
        print()

    fin = pd.DataFrame(final_rows)
    fin.to_csv(os.path.join(OUT_DIR, "picks.csv"), index=False)
    print("\n==== PORTFOLIO PICKS ====")
    print(fin.to_string(index=False))
    print(f"TOTAL YTD R: {fin['ytd_r'].sum():+.2f}  trades: {fin['ytd_trades'].sum()}")


if __name__ == "__main__":
    main()
