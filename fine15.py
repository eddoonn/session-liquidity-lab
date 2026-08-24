"""Audit 3/4: fine-resolution execution study.
Same decision variables (session levels, hourly ATR buffers), but trigger fills,
SL/TP path and time exits resolved on 15-minute bars over the last ~60 days.
Quantifies how much the hourly OHLC conventions flatter (or hurt) each book --
including the user's baseline Asia-NY book."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search2 import load, COSTS
from engine import Data, run_pattern

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

BOOKS = {
    "GOLD": [
        ("BASE_A", dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                        trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                        tp=0.75, exit_hour=8, max_hold=48)),
        ("NYJUDAS", dict(ref_win=(7, 13), ref_off=0, act_win=(13, 17), mode="rev",
                         trigger="sweep", confirm=False, buf=0.25, atr_len=10, mult=1.0,
                         tp=1.0, exit_hour=19, max_hold=36)),
    ],
    "EURUSD": [
        ("BASE_A", dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                        trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                        tp=0.75, exit_hour=8, max_hold=48)),
        ("NYJUDAS", dict(ref_win=(7, 13), ref_off=0, act_win=(13, 17), mode="rev",
                         trigger="sweep", confirm=False, buf=0.5, atr_len=10, mult=1.0,
                         tp=1.0, exit_hour=17, max_hold=36)),
    ],
    "GBPUSD": [
        ("BASE_A", dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                        trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                        tp=0.75, exit_hour=8, max_hold=48)),
        ("NYTREND", dict(ref_win=(7, 13), ref_off=0, act_win=(14, 17), mode="cont",
                         trigger="close_beyond", confirm=False, buf=0.0, atr_len=10,
                         mult=1.5, tp=1.5, exit_hour=20, max_hold=36)),
    ],
    "USDJPY": [
        ("BASE_A", dict(ref_win=(19, 21), ref_off=1, act_win=(22, 10), mode="rev",
                        trigger="sweep", confirm=False, buf=1.0, atr_len=10, mult=1.0,
                        tp=0.75, exit_hour=8, max_hold=48)),
    ],
}


def sim_fine(df_h, df_f, spec, cost, bar_min=15):
    """Re-simulate `spec` using 15m bars for execution. Decision inputs identical."""
    d = Data(df_h, atr_lens=(spec["atr_len"],))
    atr = d.atrs[spec["atr_len"]]
    # --- date-keyed reference levels built from the HOURLY data ---
    # logical-day convention identical to engine.day codes:
    # wrapped window: bars at hour >= start belong to the NEXT calendar day
    s_r, e_r = spec["ref_win"]
    hrs_h = np.asarray(df_h.index.hour)
    mref = (hrs_h >= s_r) & (hrs_h < e_r) if s_r <= e_r else (hrs_h >= s_r) | (hrs_h < e_r)
    norm_h = df_h.index.normalize()
    log_date = pd.Series(norm_h + pd.Timedelta(days=1), index=df_h.index) if s_r > e_r \
        else pd.Series(norm_h, index=df_h.index)
    lv_by_date = {}
    for did in pd.unique(log_date[mref]):
        sel = mref & (log_date == did)
        pos = np.where(sel)[0]
        lv_by_date[pd.Timestamp(did).date()] = (
            float(df_h["High"].values[pos].max()),
            float(df_h["Low"].values[pos].min()),
            df_h.index[pos[-1]])
    # ordered list of dates present in hourly data (for trading-day stepping)
    all_dates = np.array(sorted(lv_by_date.keys() |
                                {ts.date() for ts in df_h.index}))

    f = df_f
    fo, fh, fl, fc = (f[k].values for k in ("Open", "High", "Low", "Close"))
    fidx = f.index
    ph = fidx.floor("h")
    atr_of_ph = {ts: atr[i] for i, ts in enumerate(df_h.index)}
    rev = spec["mode"] == "rev"
    cb = spec["trigger"] == "close_beyond"
    trades = []
    hrs_f = np.asarray(fidx.hour)
    s, e = spec["act_win"]
    m = (hrs_f >= s) & (hrs_f < e) if s <= e else (hrs_f >= s) | (hrs_f < e)
    # logical session date for action-window bars (same convention as ref side:
    # bars at hour >= start belong to the NEXT calendar day when window wraps)
    if s > e:
        ld_f = pd.Series(np.where(hrs_f >= s,
                                  fidx.normalize() + pd.Timedelta(days=1),
                                  fidx.normalize()), index=fidx)
    else:
        ld_f = pd.Series(fidx.normalize(), index=fidx)
    pos_all = np.where(m)[0]
    if len(pos_all) == 0:
        return trades
    sess_dates = pd.unique(ld_f.to_numpy()[pos_all])
    for sd in sess_dates:
        seg = pos_all[ld_f.to_numpy()[pos_all] == sd]
        if len(seg) < 2:
            continue
        if fidx[seg[0]].weekday() == 6:
            continue
        first_pos15 = seg[0]
        # walk back calendar days; skip days without ref window; stop after 7 tries
        import datetime as _dt
        sd_d = pd.Timestamp(sd).date()
        ref = None
        for k in range(spec["ref_off"], spec["ref_off"] + 7):
            cand_date = sd_d - _dt.timedelta(days=k)
            cand = lv_by_date.get(cand_date)
            if cand and cand[2] < fidx[first_pos15]:
                ref = cand
                break
            _ = all_dates  # (kept for clarity; calendar walk handles gaps)
        if ref is None:
            continue
        rh, rl, _ = ref

        entry = None
        for j in seg:
            key = ph[j]
            a = atr_of_ph.get(key)
            if a is None or not np.isfinite(a) or a <= 0:
                continue
            bufd = spec["buf"] * a
            if cb:
                # evaluate on the LAST 15m bar of each parent hour (hourly close)
                nxt = j + 1
                if nxt < len(f) and ph[nxt] == key:
                    continue
                up, dn = fc[j] > rh, fc[j] < rl
            else:
                up = fh[j] >= rh + bufd
                dn = fl[j] <= rl - bufd
            if up and dn:
                continue
            if not (up or dn):
                continue
            swept_high = bool(up)
            side_short = swept_high if rev else (not swept_high)
            if cb:
                if j + 1 >= len(f):
                    break
                # enter at first 15m bar of NEXT parent hour (matches hourly next-open)
                nj = j + 1
                epos, entry_px = nj, float(fo[nj])
            else:
                lvl = (rh + bufd) if swept_high else (rl - bufd)
                contig = j == 0 or (fidx[j] - fidx[j - 1]) <= pd.Timedelta(minutes=bar_min * 2)
                if rev:
                    fill = (max(lvl, fo[j]) if side_short else min(lvl, fo[j])) if contig else lvl
                else:
                    fill = (max(lvl, fo[j]) if not side_short else min(lvl, fo[j])) if contig else lvl
                epos, entry_px = j, float(fill)
            risk = spec["mult"] * a
            sl = entry_px + risk if side_short else entry_px - risk
            tp = (rl if spec["tp"] == "opp" else
                  (entry_px - spec["tp"] * risk if side_short else entry_px + spec["tp"] * risk))
            entry = (side_short, epos, entry_px, float(sl), float(tp), float(risk))
            break
        if entry is None:
            continue
        side_short, epos, entry_px, sl, tp, risk = entry
        res = xo = xpx = why = None
        for j in range(epos, len(f)):
            if side_short:
                hs, ht = fh[j] >= sl, fl[j] <= tp
            else:
                hs, ht = fl[j] <= sl, fh[j] >= tp
            if hs:  # conservative within 15m bar too
                res, xo, xpx, why = (-risk - cost) / risk, j, sl, "sl"
                break
            if ht:
                g = (entry_px - tp) if side_short else (tp - entry_px)
                res, xo, xpx, why = (g - cost) / risk, j, tp, "tp"
                break
            # time exit at close of the exit hour
            nxt = j + 1
            if ph[j].hour == spec["exit_hour"] and (nxt >= len(f) or ph[nxt].hour != spec["exit_hour"]) \
                    and ph[j] != ph[epos]:
                g = (entry_px - fc[j]) if side_short else (fc[j] - entry_px)
                res, xo, xpx, why = (g - cost) / risk, j, float(fc[j]), "time"
                break
        if res is None:
            j = len(f) - 1
            g = (entry_px - fc[j]) if side_short else (fc[j] - entry_px)
            res, xo, xpx, why = (g - cost) / risk, j, float(fc[j]), "eod"
        trades.append(dict(date=fidx[first_pos15].date(), sess=fidx[first_pos15],
                           t_in=fidx[epos],
                           side="short" if side_short else "long",
                           entry=round(entry_px, 5), r=round(res, 3), reason=why))
    return trades


def main():
    print(f"{'='*88}\nFINE-RESOLUTION EXECUTION STUDY (15m bars, ~60d overlap)\n{'='*88}")
    rows_out = []
    for inst, books in BOOKS.items():
        df_h = load(inst)
        df_f = pd.read_csv(os.path.join(DATA_DIR, f"{inst}_15m.csv"), index_col=0,
                           parse_dates=True)
        df_f.index = pd.to_datetime(df_f.index, utc=True)
        t0 = df_f.index[0]
        d = Data(df_h, atr_lens=(10,))
        cost = COSTS[inst]
        for bname, spec in books:
            tr_h = [t for t in run_pattern(d, spec, cost=cost) if t["entry_time"] >= t0]
            tr_f = sim15(df_h, df_f, spec, cost)
            # match by (date, side)
            mh = {(t["date"], t["side"]): t for t in tr_h}
            mf = {(t["date"], t["side"]): t for t in tr_f}
            common = set(mh) & set(mf)
            rh_ = sum(mh[k]["r"] for k in common)
            rf_ = sum(mf[k]["r"] for k in common)
            dh = [mf[k]["r"] - mh[k]["r"] for k in common]
            wrh = np.mean([mh[k]["r"] > 0 for k in common]) * 100
            wrf = np.mean([mf[k]["r"] > 0 for k in common]) * 100
            reasons_differ = np.mean([mh[k]["reason"] != mf[k]["reason"] for k in common])
            row = dict(inst=inst, book=bname, nH=len(tr_h), nF=len(tr_f),
                       matched=len(common),
                       R_hourly=round(rh_, 2), R_15m=round(rf_, 2),
                       delta=round(rf_ - rh_, 2),
                       med_abs_dR=round(float(np.median(np.abs(dh))), 3) if dh else 0,
                       wrH=round(wrh, 1), wrF=round(wrf, 1),
                       reason_chg=round(reasons_differ * 100, 1))
            rows_out.append(row)
            print(f"{inst:7s} {bname:8s} nH={row['nH']:3d} nF={row['nF']:3d} "
                  f"matched={row['matched']:3d} | R h={row['R_hourly']:+7.2f} "
                  f"15m={row['R_15m']:+7.2f} delta={row['delta']:+6.2f} "
                  f"| medAbsdR={row['med_abs_dR']:.2f}R wr {row['wrH']}%->{row['wrF']}% "
                  f"exitReasonChg {row['reason_chg']}%")
    pd.DataFrame(rows_out).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "results", "fine15_comparison.csv"), index=False)


if __name__ == "__main__":
    main()


sim15 = sim_fine  # backward-compatible alias
