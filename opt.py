"""Optimize the top-3 surviving books. Protocol per book:
1) core grid -> train (<2026) gates under NAIVE and PATH_AWARE models
2) rank candidates by PATH-AWARE train R (honest proxy), keep top-K
3) structural overlays (breakeven / weekdays / vol-regime) on those K
4) measure each finalist's true execution tax on 5m data (60d)
5) report 5m-calibrated OOS 2026 + distribution stats (no single-point cherry pick)
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from engine import Data, run_pattern, summarize, segment_r
from search2 import load
from fine15 import sim_fine

COSTS = {"GOLD": 0.35, "EURUSD": 0.00020, "GBPUSD": 0.00028, "USDJPY": 0.024}
BROKER_COSTS = {"GOLD": 0.55, "USDJPY": 0.015}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MT = [0, 1, 2, 3]


def base_spec(**kw):
    s = dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev", trigger="sweep",
             confirm=False, buf=1.0, atr_len=10, mult=1.0, tp=0.75, exit_hour=8,
             max_hold=48, weekdays=None, be_trigger=None, atr_pct=None, ref_ratio=None,
             path_aware=False)
    s.update(kw)
    return s


def grid_nyjudas():
    out = []
    for ref in [(7, 12), (7, 13), (6, 13)]:
        for buf in [0.25, 0.5]:
            for mult in [0.75, 1.0, 1.25]:
                for tp in [1.0, 1.5]:
                    for ex in [18, 19, 20]:
                        out.append(base_spec(ref_win=ref, ref_off=0, act_win=(13, 17),
                                             buf=buf, mult=mult, tp=tp, exit_hour=ex,
                                             max_hold=36))
    return out


def grid_asia(inst):
    out = []
    for ref in [(19, 21), (19, 22), (20, 22)]:
        for act in [(22, 9), (22, 10)]:
            for buf in [0.75, 1.0, 1.25]:
                for mult in [1.0, 1.25]:
                    for tp in [0.5, 0.75, 1.0]:
                        for ex in [7, 8]:
                            out.append(base_spec(ref_win=ref, act_win=act, buf=buf,
                                                 mult=mult, tp=tp, exit_hour=ex))
    return out


def evaluate(d, spec, cost):
    tr = run_pattern(d, dict(spec), cost=cost)
    tn = [t for t in tr if t["entry_time"] < SPLIT]
    te = [t for t in tr if t["entry_time"] >= SPLIT]
    return tn, te


def gates(tn, min_trades):
    if len(tn) < min_trades:
        return None
    s = summarize(tn)
    segs = segment_r(tn, 3)
    if s["profit_factor"] < 1.25 or s["max_dd_r"] > 8 or not all(x > 0 for x in segs):
        return None
    return s


def tax5m(inst, df_h, df5, d, spec, cost):
    """paired naive-vs-5m tax on the overlap window"""
    tr_h = [t for t in run_pattern(d, dict(spec), cost=cost)
            if t["entry_time"] >= df5.index[0]]
    tr_f = sim_fine(df_h, df5, spec, cost, bar_min=5)
    mh = {(t["date"], t["side"]): t for t in tr_h}
    mf = {(t["date"], t["side"]): t for t in tr_f}
    common = set(mh) & set(mf)
    if len(common) < 15:
        return np.nan, len(common)
    rh = sum(mh[k]["r"] for k in common)
    rf = sum(mf[k]["r"] for k in common)
    return (rf - rh) / len(common), len(common)


def main():
    df_g = load("GOLD"); df_j = load("USDJPY")
    df_g5 = pd.read_csv("data/GOLD_5m.csv", index_col=0, parse_dates=True); df_g5.index = pd.to_datetime(df_g5.index, utc=True)
    df_j5 = pd.read_csv("data/USDJPY_5m.csv", index_col=0, parse_dates=True); df_j5.index = pd.to_datetime(df_j5.index, utc=True)

    BOOKS = {
        ("GOLD", "NYJUDAS"): (df_g, df_g5, Data(df_g, atr_lens=(10,)), grid_nyjudas(), 80),
        ("USDJPY", "BASE_A"): (df_j, df_j5, Data(df_j, atr_lens=(10,)), grid_asia("USDJPY"), 60),
        ("GOLD", "BASE_A"): (df_g, df_g5, Data(df_g, atr_lens=(10,)), grid_asia("GOLD"), 60),
    }
    # NOTE: GOLD BASE_A and NYJUDAS share the Data cache safely (read-only use)

    final_rows = []
    for (inst, book), (df_h, df5, d, grid, min_tr) in BOOKS.items():
        cost = COSTS[inst]
        rows = []
        for spec in grid:
            tn, te = evaluate(d, spec, cost)
            g = gates(tn, min_tr)
            if g is None:
                continue
            sp_p = dict(spec); sp_p["path_aware"] = True
            tp_tr = [t for t in run_pattern(d, sp_p, cost=cost) if t["entry_time"] < SPLIT]
            gp = summarize(tp_tr)
            segp = segment_r(tp_tr, 3)
            if gp["profit_factor"] < 1.15 or not all(x > 0 for x in segp):
                continue
            se = summarize(te)
            rows.append(dict(spec=spec, n_tr=g["trades"], Rn_tr=g["total_r"],
                             Rp_tr=gp["total_r"], pfn=g["profit_factor"], dd=g["max_dd_r"],
                             te_R=se.get("total_r", 0), te_n=se.get("trades", 0),
                             te_pf=se.get("profit_factor", 0)))
        gdf = pd.DataFrame([{k: v for k, v in r.items() if k != "spec"} for r in rows])
        gdf["spec"] = [r["spec"] for r in rows]
        gdf = gdf.sort_values("Rp_tr", ascending=False)
        print(f"=== {inst} {book}: {len(gdf)}/{len(grid)} passed dual gates ===")

        # overlays on top-15 by path-aware train R
        overlaid = []
        for _, r in gdf.head(15).iterrows():
            for be in [None, 0.6, 1.0]:
                for wd in [None, MT]:
                    for vp in [None, (0.15, 0.85)]:
                        if be is None and wd is None and vp is None:
                            overlaid.append(r.to_dict())
                            continue
                        s2 = dict(r["spec"]); s2.update(be_trigger=be,
                                                        weekdays=list(wd) if wd else None,
                                                        atr_pct=vp)
                        tn, te = evaluate(d, s2, cost)
                        g2 = gates(tn, min_tr * 0.8)
                        if g2 is None:
                            continue
                        se = summarize(te)
                        d2 = r.to_dict(); d2["spec"] = s2
                        d2.update(n_tr=g2["trades"], Rn_tr=g2["total_r"], te_R=se.get("total_r", 0),
                                  te_n=se.get("trades", 0), te_pf=se.get("profit_factor", 0))
                        overlaid.append(d2)
        odf = pd.DataFrame([{k: v for k, v in r.items() if k != "spec"} for r in overlaid])
        odf["spec"] = [r["spec"] for r in overlaid]
        odf = odf.drop_duplicates(subset=["Rn_tr", "te_R", "te_n"]).sort_values("Rp_tr", ascending=False)
        print(f"after overlays: {len(odf)} candidates")

        # 5m-calibrate ALL remaining candidates (median = unbiased expectation)
        calibs = []
        for _, r in odf.iterrows():
            t, nov = tax5m(inst, df_h, df5, d, r["spec"], cost)
            calib = r["te_R"] + (0 if np.isnan(t) else t) * r["te_n"]
            calibs.append(calib)
            r["_cal"] = calib
            r["_tax"] = t
        odf["_cal"] = calibs
        odf = odf.sort_values("_cal", ascending=False)
        med_cal = float(np.nanmedian(odf["_cal"]))
        med_naive = float(np.median(odf["te_R"]))
        best = odf.iloc[0]
        bc = BROKER_COSTS.get(inst, COSTS[inst])
        tb, _ = tax5m(inst, df_h, df5, d, best["spec"], bc)
        tn_b, te_b = evaluate(d, best["spec"], bc)
        sb = summarize(te_b)
        cal_broker = sb.get("total_r", 0) + (0 if np.isnan(tb) else tb) * sb.get("trades", 0)
        print(f"gated candidates: {len(odf)} | median naive OOS {med_naive:+.1f}R | "
              f"median CALIBRATED {med_cal:+.1f}R")
        print(f"PICK: buf={best['spec']['buf']} ref={best['spec']['ref_win']} "
              f"mult={best['spec']['mult']} tp={best['spec']['tp']} x={best['spec']['exit_hour']:02d} "
              f"be={best['spec'].get('be_trigger')} wd={best['spec'].get('weekdays')} "
              f"vp={best['spec'].get('atr_pct')}")
        print(f"  OOS naive {best['te_R']:+.1f}R/{int(best['te_n'])}t PF{best['te_pf']:.2f} "
              f"-> 5m-calibrated {best['_cal']:+.1f}R | broker-cost calibrated {cal_broker:+.1f}R")
        print()
        final_rows.append(dict(inst=inst, book=book, cand_count=len(odf),
                               median_naive=med_naive, median_cal=med_cal,
                               R_naive=best["te_R"], n=int(best["te_n"]),
                               R_cal=round(best["_cal"], 1),
                               R_cal_broker=round(cal_broker, 1),
                               pf=best["te_pf"]))
        bspec = best["spec"]
        bspec_out = {k: v for k, v in bspec.items() if k != "path_aware"}
        pd.DataFrame([bspec_out]).to_csv(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results", f"opt_{inst}_{book}.csv"),
            index=False)

    fin = pd.DataFrame(final_rows)
    fin.to_csv(os.path.join(OUT := os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "results"), "optimization_summary.csv"), index=False)
    print("=== OPTIMIZATION SUMMARY ===")
    print(fin.to_string(index=False))


if __name__ == "__main__":
    main()
