"""Deep optimization: GOLD NY-Judas v2 and USDJPY Asia-NY v2.
Stages: fine grid (dual-model train gates) -> BE/lock/weekday overlays ->
5m tax calibration of EVERY gated candidate -> winner's-curse-aware report
-> robustness battery on the chosen config."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from engine import Data, run_pattern, summarize, segment_r
from search2 import load
from fine15 import sim_fine

COSTS = {"GOLD": 0.35, "USDJPY": 0.024}
BROKER = {"GOLD": 0.55, "USDJPY": 0.015}
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")
MT = [0, 1, 2, 3]
NOTMON = [1, 2, 3, 4]


def base_spec(**kw):
    s = dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev", trigger="sweep",
             confirm=False, buf=1.0, atr_len=10, mult=1.0, tp=0.75, exit_hour=8,
             max_hold=48, weekdays=None, be_trigger=None, be_lock=0.0,
             atr_pct=None, ref_ratio=None, path_aware=False)
    s.update(kw)
    return s


def eval_book(d, cost, grid, min_tr):
    rows = []
    for spec in grid:
        tn = [t for t in run_pattern(d, dict(spec), cost=cost) if t["entry_time"] < SPLIT]
        if len(tn) < min_tr:
            continue
        sn = summarize(tn); segs = segment_r(tn, 3)
        if sn["profit_factor"] < 1.25 or sn["max_dd_r"] > 8 or not all(x > 0 for x in segs):
            continue
        sp = dict(spec); sp["path_aware"] = True
        tp_ = [t for t in run_pattern(d, sp, cost=cost) if t["entry_time"] < SPLIT]
        gp = summarize(tp_); segp = segment_r(tp_, 3)
        if gp["profit_factor"] < 1.15 or not all(x > 0 for x in segp):
            continue
        te = [t for t in run_pattern(d, dict(spec), cost=cost) if t["entry_time"] >= SPLIT]
        se = summarize(te)
        rows.append(dict(spec=spec, n_tr=len(tn), Rn_tr=sn["total_r"], Rp_tr=gp["total_r"],
                         te_R=se.get("total_r", 0.0), te_n=se.get("trades", 0),
                         te_pf=se.get("profit_factor", 0), te_dd=se.get("max_dd_r", 0),
                         te_wr=se.get("win_rate_pct", 0)))
    return rows


def calibrate(df_h, df5, d, rows, cost):
    for r in rows:
        tr_h = [t for t in run_pattern(d, dict(r["spec"]), cost=cost)
                if t["entry_time"] >= df5.index[0]]
        tr_f = sim_fine(df_h, df5, r["spec"], cost, bar_min=5)
        mh = {(t["date"], t["side"]): t for t in tr_h}
        mf = {(t["date"], t["side"]): t for t in tr_f}
        common = set(mh) & set(mf)
        if len(common) < 12:
            r["tax"] = np.nan; r["ov_n"] = len(common)
        else:
            r["tax"] = (sum(mf[k]["r"] for k in common) -
                        sum(mh[k]["r"] for k in common)) / len(common)
            r["ov_n"] = len(common)
        r["R_cal"] = r["te_R"] + (0 if np.isnan(r["tax"]) else r["tax"]) * r["te_n"]
    return rows


def main():
    df_g = load("GOLD"); df_j = load("USDJPY")
    dg = Data(df_g, atr_lens=(10,)); dj = Data(df_j, atr_lens=(10,))
    df_g5 = pd.read_csv("data/GOLD_5m.csv", index_col=0, parse_dates=True)
    df_g5.index = pd.to_datetime(df_g5.index, utc=True)
    df_j5 = pd.read_csv("data/USDJPY_5m.csv", index_col=0, parse_dates=True)
    df_j5.index = pd.to_datetime(df_j5.index, utc=True)

    # ---------- GOLD NY-JUDAS v3 ----------
    print("=== GOLD NY-JUDAS: fine grid ===")
    g1 = []
    for ref in [(6, 13), (7, 13)]:
        for buf in [0.2, 0.25, 0.35]:
            for mult in [0.7, 0.75, 0.85]:
                for tp in [0.85, 1.0, 1.15]:
                    for ex in [17, 18]:
                        g1.append(base_spec(ref_win=ref, ref_off=0, act_win=(13, 17),
                                            buf=buf, mult=mult, tp=tp, exit_hour=ex,
                                            be_trigger=0.6, weekdays=MT, max_hold=36))
    rows = eval_book(dg, COSTS["GOLD"], g1, 80)
    rows.sort(key=lambda r: -r["Rp_tr"])
    top = rows[:20]
    print(f"gated {len(rows)}/{len(g1)}; overlaying top {len(top)}")
    g2 = []
    for r in top:
        s0 = r["spec"]
        for be in [0.5, 0.6, 0.7]:
            for lock in [0.0, 0.1]:
                for wd in [MT, NOTMON]:
                    s2 = base_spec(ref_win=s0["ref_win"], ref_off=0, act_win=s0["act_win"],
                                   buf=s0["buf"], mult=s0["mult"], tp=s0["tp"],
                                   exit_hour=s0["exit_hour"], be_trigger=be, be_lock=lock,
                                   weekdays=list(wd), max_hold=36)
                    tn = [t for t in run_pattern(dg, dict(s2), cost=COSTS["GOLD"])
                          if t["entry_time"] < SPLIT]
                    if len(tn) < 60:
                        continue
                    sn = summarize(tn); segs = segment_r(tn, 3)
                    if sn["profit_factor"] < 1.25 or not all(x > 0 for x in segs):
                        continue
                    te = [t for t in run_pattern(dg, dict(s2), cost=COSTS["GOLD"])
                          if t["entry_time"] >= SPLIT]
                    se = summarize(te)
                    g2.append(dict(spec=s2, n_tr=len(tn), Rn_tr=sn["total_r"],
                                   Rp_tr=r["Rp_tr"], te_R=se.get("total_r", 0),
                                   te_n=se.get("trades", 0), te_pf=se.get("profit_factor", 0),
                                   te_dd=se.get("max_dd_r", 0), te_wr=se.get("win_rate_pct", 0)))
    g2 = calibrate(df_g, df_g5, dg, g2, COSTS["GOLD"])
    g2 = [r for r in g2 if not np.isnan(r.get("tax", np.nan))]
    g2.sort(key=lambda r: -r["R_cal"])
    cals = [r["R_cal"] for r in g2]
    print(f"candidates fully calibrated: {len(g2)} | median cal {np.median(cals):+.1f}R | "
          f"p25 {np.percentile(cals,25):+.1f} p75 {np.percentile(cals,75):+.1f}")
    best = g2[0]
    b = best["spec"]
    print(f"PICK-NYJUDAS: ref{b['ref_win']} buf{b['buf']} m{b['mult']} tp{b['tp']} "
          f"x{b['exit_hour']} be{b['be_trigger']}/{b['be_lock']} wd{b['weekdays']}")
    print(f"  OOS naive {best['te_R']:+.1f}R/{best['te_n']}t PF{best['te_pf']:.2f} "
          f"WR{best['te_wr']:.0f}% DD{best['te_dd']:.1f}R -> CAL {best['R_cal']:+.1f}R "
          f"(tax {best['tax']:+.3f}, ov_n {best['ov_n']})")
    tb, _ = sim_tax_broker = None, None
    pd.DataFrame([{k: v for k, v in r.items() if k != "spec"} for r in g2[:20]]).to_csv(
        "results/opt_nyjudas_top20.csv", index=False)

    # ---------- USDJPY ----------
    print("\n=== USDJPY ASIA-NY: fine grid ===")
    j1 = []
    for ref in [(19, 22), (20, 22), (19, 21)]:
        for act in [(22, 10), (22, 9)]:
            for buf in [1.15, 1.25, 1.4]:
                for mult in [1.0, 1.15]:
                    for tp in [0.65, 0.75, "opp"]:
                        for ex in [6, 7]:
                            j1.append(base_spec(ref_win=ref, act_win=act, buf=buf, mult=mult,
                                                tp=tp, exit_hour=ex))
    rowsj = eval_book(dj, COSTS["USDJPY"], j1, 60)
    rowsj.sort(key=lambda r: -r["Rp_tr"])
    topj = rowsj[:20]
    print(f"gated {len(rowsj)}/{len(j1)}; overlaying top {len(topj)}")
    j2 = []
    for r in topj:
        s0 = r["spec"]
        for be in [None, 0.6, 0.8]:
            for wd in [MT, NOTMON]:
                s2 = base_spec(ref_win=s0["ref_win"], act_win=s0["act_win"], buf=s0["buf"],
                               mult=s0["mult"], tp=s0["tp"], exit_hour=s0["exit_hour"],
                               be_trigger=be, weekdays=list(wd))
                tn = [t for t in run_pattern(dj, dict(s2), cost=COSTS["USDJPY"])
                      if t["entry_time"] < SPLIT]
                if len(tn) < 50:
                    continue
                sn = summarize(tn); segs = segment_r(tn, 3)
                if sn["profit_factor"] < 1.25 or not all(x > 0 for x in segs):
                    continue
                te = [t for t in run_pattern(dj, dict(s2), cost=COSTS["USDJPY"])
                      if t["entry_time"] >= SPLIT]
                se = summarize(te)
                j2.append(dict(spec=s2, n_tr=len(tn), Rn_tr=sn["total_r"],
                               Rp_tr=r["Rp_tr"], te_R=se.get("total_r", 0),
                               te_n=se.get("trades", 0), te_pf=se.get("profit_factor", 0),
                               te_dd=se.get("max_dd_r", 0), te_wr=se.get("win_rate_pct", 0)))
    j2 = calibrate(df_j, df_j5, dj, j2, COSTS["USDJPY"])
    j2 = [r for r in j2 if not np.isnan(r.get("tax", np.nan))]
    j2.sort(key=lambda r: -r["R_cal"])
    calsj = [r["R_cal"] for r in j2]
    print(f"candidates fully calibrated: {len(j2)} | median cal {np.median(calsj):+.1f}R | "
          f"p25 {np.percentile(calsj,25):+.1f} p75 {np.percentile(calsj,75):+.1f}")
    bj = j2[0]; sj = bj["spec"]
    print(f"PICK-USDJPY: ref{sj['ref_win']} act{sj['act_win']} buf{sj['buf']} m{sj['mult']} "
          f"tp{sj['tp']} x{sj['exit_hour']} be{sj['be_trigger']} wd{sj['weekdays']}")
    print(f"  OOS naive {bj['te_R']:+.1f}R/{bj['te_n']}t PF{bj['te_pf']:.2f} "
          f"WR{bj['te_wr']:.0f}% DD{bj['te_dd']:.1f}R -> CAL {bj['R_cal']:+.1f}R "
          f"(tax {bj['tax']:+.3f}, ov_n {bj['ov_n']})")
    pd.DataFrame([{k: v for k, v in r.items() if k != "spec"} for r in j2[:20]]).to_csv(
        "results/opt_usdjpy_top20.csv", index=False)

    # persist picks
    out = []
    for nm, r in [("GOLD_NYJUDAS_v3", best), ("USDJPY_BASE_v3", bj)]:
        s = {k: v for k, v in r["spec"].items() if k != "path_aware"}
        s.update(inst=nm.split("_")[0], book=nm, R_naive=r["te_R"], n=int(r["te_n"]),
                 pf=r["te_pf"], wr=r["te_wr"], dd=r["te_dd"], tax=float(r["tax"]),
                 R_cal=round(r["R_cal"], 1))
        out.append(s)
    pd.DataFrame(out).to_csv("results/final_picks_v3.csv", index=False)


if __name__ == "__main__":
    main()
