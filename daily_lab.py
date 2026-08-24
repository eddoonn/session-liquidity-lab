"""Daily-timeframe pattern lab — close-based fills only (no intrabar ambiguity),
~20y history via Yahoo daily bars. Walk-forward: select on <2026, validate 2026 YTD,
report full-history yearly stability."""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_data import SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SPLIT = pd.Timestamp("2026-01-01", tz="UTC")

# daily round-trip cost in price units (spread+slippage, conservative)
COSTS = {"GOLD": 0.60, "EURUSD": 0.00040, "GBPUSD": 0.00055, "USDJPY": 0.048}


def load_daily(name):
    path = os.path.join(DATA_DIR, f"{name}_1d.csv")
    sym = {"GOLD": "GC=F", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X"}[name]
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(sym, period="max", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.to_csv(path)
    return df


def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n // 2).mean().values


def run_sweep_reversal(df, lookback, buf_atr, atr_len, mult, tp_r, max_hold, cost, min_age=1):
    """N-day extreme swept intraday by buf*ATR, but day CLOSES back inside ->
    enter next open (rejection of the sweep). Close-based, unambiguous."""
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    idx = df.index
    n = len(o)
    a = atr(df, atr_len)
    trades = []
    roll_hi = pd.Series(h).rolling(lookback).max().values
    roll_lo = pd.Series(l).rolling(lookback).min().values
    for p in range(lookback + 1, n - 1):
        aa = a[p]
        if not np.isfinite(aa) or aa <= 0:
            continue
        hi_lvl, lo_lvl = roll_hi[p - min_age], roll_lo[p - min_age]
        swept_up = h[p] >= hi_lvl + buf_atr * aa and c[p] < hi_lvl      # rejection of upside sweep
        swept_dn = l[p] <= lo_lvl - buf_atr * aa and c[p] > lo_lvl      # rejection of downside sweep
        if swept_up == swept_dn:
            continue
        side_short = bool(swept_up)
        entry_px = float(o[p + 1])
        risk = mult * aa
        if entry_px <= 0:
            continue
        sl = entry_px + risk if side_short else entry_px - risk
        tp = entry_px - tp_r * risk if side_short else entry_px + tp_r * risk
        res = None
        for j in range(p + 1, min(p + 1 + max_hold, n)):
            if side_short:
                hit_sl, hit_tp = h[j] >= sl, l[j] <= tp
                mfe = max(0.0, entry_px - l[j]); mae = max(0.0, h[j] - entry_px)
            else:
                hit_sl, hit_tp = l[j] <= sl, h[j] >= tp
                mfe = max(0.0, h[j] - entry_px); mae = max(0.0, entry_px - l[j])
            if hit_sl:
                res = (-risk - cost) / risk; xo, xpx, xr, why = j, sl, "sl", None
                break
            if hit_tp:
                gross = (entry_px - tp) if side_short else (tp - entry_px)
                res = (gross - cost) / risk; xo, xpx, xr = j, tp, "tp"
                break
        if res is None:
            j = min(p + max_hold, n - 1)
            gross = (entry_px - c[j]) if side_short else (c[j] - entry_px)
            res = (gross - cost) / risk; xo, xpx, xr = j, c[j], "time"
        trades.append(dict(date=idx[p + 1].date(), t_in=df.index[p + 1], t_out=df.index[xo],
                           side="short" if side_short else "long", entry=entry_px,
                           r=round(res, 3), reason=xr))
    return trades


def summarize(tr):
    if not tr:
        return dict(trades=0)
    r = np.array([t["r"] for t in tr])
    w, ls = r[r > 0], r[r <= 0]
    pf = w.sum() / abs(ls.sum()) if len(ls) and ls.sum() != 0 else 99.0
    eq = np.cumsum(r)
    return dict(trades=len(r), total_r=round(float(r.sum()), 1),
                wr=round(100 * len(w) / len(r), 1), pf=round(pf, 2),
                dd=round(float((np.maximum.accumulate(eq) - eq).max()), 1))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grids = []
    for lb in (20, 55, 120):
        for buf in (0.25, 0.5, 1.0):
            for mult in (1.5, 2.0, 2.5):
                for tp in (2.0, 3.0):
                    for mh in (10, 20):
                        grids.append(dict(lookback=lb, buf_atr=buf, atr_len=14,
                                          mult=mult, tp_r=tp, max_hold=mh))
    print(f"{len(grids)} daily configs/instrument\n")
    picks = {}
    for inst in SYMBOLS:
        df = load_daily(inst)
        cost = COSTS[inst]
        rows = []
        for gcfg in grids:
            tr = run_sweep_reversal(df, cost=cost, **gcfg)
            st = summarize(tr)
            if st.get("trades", 0) < 100:
                continue
            tr_df = pd.DataFrame(tr)
            tr_tr = tr_df[tr_df["t_in"] < SPLIT]
            st_tr = summarize(tr_tr.to_dict("records"))
            r3 = np.array_split(tr_tr["r"].values, 3)
            seg_ok = all(x.sum() > 0 for x in r3 if len(x))
            te = tr_df[tr_df["t_in"] >= SPLIT]
            st_te = summarize(te.to_dict("records"))
            rows.append(dict(inst=inst, **gcfg, **{f"tr_{k}": v for k, v in st_tr.items()},
                             seg_ok=seg_ok, **{f"te_{k}": v for k, v in st_te.items()}))
        g = pd.DataFrame(rows)
        gd = g[(g["seg_ok"]) & (g["tr_pf"] >= 1.25)].sort_values("tr_total_r", ascending=False)
        print(f"=== {inst}: {len(df)} bars {df.index[0].date()}..{df.index[-1].date()} "
              f"-> {len(gd)} gated")
        if not gd.empty:
            print(gd.head(3).to_string(index=False))
            picks[inst] = gd.iloc[0].to_dict()
            # yearly stability of the pick
            win = gd.iloc[0]
            tr = pd.DataFrame(run_sweep_reversal(df, cost=cost, **{k: win[k] for k in
                              ("lookback", "buf_atr", "atr_len", "mult", "tp_r", "max_hold")}))
            yr = tr.groupby(tr["t_in"].dt.year)["r"].agg(["count", "sum"]).round(1)
            pos_years = int((yr["sum"] > 0).sum()); tot = len(yr)
            print(f"yearly positive: {pos_years}/{tot} | last5:",
                  yr["sum"].tail(5).to_dict())
        print()

    pd.DataFrame(picks.values()).to_csv(os.path.join(OUT_DIR, "picks_daily.csv"), index=False)


if __name__ == "__main__":
    main()
