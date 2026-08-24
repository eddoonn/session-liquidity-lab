"""Exploratory session-pattern analysis: where do liquidity sweeps live and what follows them?"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SYMBOLS = ["GOLD", "EURUSD", "GBPUSD", "USDJPY"]


def load(name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{name}_60m.csv"), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def atr(df, n=10):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def day_id(idx):
    """Unit-safe integer id per UTC calendar day, consecutive over days PRESENT in data."""
    days = idx.tz_convert("UTC").floor("D")
    codes, _ = pd.factorize(days)
    return codes.astype(np.int64)


def window_levels(df, s, e):
    """high/low of UTC hour window [s,e) per calendar day"""
    hrs = df.index.hour
    m = (hrs >= s) & (hrs < e) if s <= e else (hrs >= s) | (hrs < e)
    ids = day_id(df.index)
    out = {}
    for did in np.unique(ids[m]):
        pos = np.where(m & (ids == did))[0]
        out[int(did)] = (float(df["High"].values[pos].max()), float(df["Low"].values[pos].min()), int(pos[-1]))
    return out


def fwd_ret(df, pos, horizon):
    c = df["Close"].values
    j = min(pos + horizon, len(c) - 1)
    return c[j] - c[pos]


def analyze(name):
    df = load(name)
    a = atr(df).values
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    idx = df.index
    hrs = idx.hour
    ids = day_id(idx)

    print(f"\n{'='*70}\n{name}   bars={len(df)}  {idx[0].date()}..{idx[-1].date()}")

    # ---- 1. hourly range profile (in ATR units) ----
    rng = ((h - l) / np.maximum(a, 1e-9))
    prof = pd.Series(rng).groupby(hrs).median()
    top = prof.sort_values(ascending=False).head(6)
    print("hottest hours (median range/ATR):", {int(k): round(v, 2) for k, v in top.items()})

    # ---- 2. sweep events: Asia window (22-09 UTC) vs prior-day NY-late (19-21) levels ----
    ny = window_levels(df, 19, 21)
    asia_m = (hrs >= 22) | (hrs < 9)
    asia_ids = np.where(hrs >= 22, ids + 1, ids)
    stats = {"sweep_hi": [], "sweep_lo": []}
    for did in np.unique(asia_ids[asia_m]):
        seg = np.where(asia_m & (asia_ids == did))[0]
        if len(seg) < 2 or int(did) - 1 not in ny:
            continue
        rh, rl, _ = ny[int(did) - 1]
        aa = a[seg]
        hi_hit = None
        lo_hit = None
        for k, p in enumerate(seg):
            if hi_hit is None and h[p] >= rh + aa[k]:
                hi_hit = p
            if lo_hit is None and l[p] <= rl - aa[k]:
                lo_hit = p
        if hi_hit is not None and (lo_hit is None or hi_hit < lo_hit):
            stats["sweep_hi"].append((hi_hit, rl))          # swept high -> short candidate; target opposite low
        elif lo_hit is not None:
            stats["sweep_lo"].append((lo_hit, rh))

    for key, evs in stats.items():
        if not evs:
            continue
        rets = {}
        n = len(evs)
        for hor in (2, 4, 8):
            rr = []
            for p, tgt in evs:
                r = fwd_ret(df, p, hor)
                sign = -1 if key == "sweep_hi" else 1
                rr.append(sign * r / max(a[p], 1e-9))
            rr = np.array(rr)
            rets[hor] = (np.median(rr), (rr > 0).mean(), n)
        line = "  ".join(f"h{h_}: med={v[0]:+.2f}ATR wr={v[1]*100:.0f}%" for h_, v in rets.items())
        print(f"Asia sweep {key:9s} n={n:4d}  {line}")

    # ---- 3. London (06-09) sweep of Asia range extremes -> reversal into NY? ----
    asia_lv = {}
    am = ((hrs >= 22) | (hrs < 5))
    aids = np.where(hrs >= 22, ids + 1, ids)
    for did in np.unique(aids[am]):
        pos = np.where(am & (aids == did))[0]
        if len(pos) >= 2:
            asia_lv[int(did)] = (h[pos].max(), l[pos].min(), pos[-1])
    ev_hi, ev_lo = [], []
    lon_m = (hrs >= 6) & (hrs < 10)
    for did in np.unique(ids[lon_m]):
        seg = np.where(lon_m & (ids == did))[0]
        if len(seg) < 2 or int(did) not in asia_lv:
            continue
        ah, al, lastp = asia_lv[int(did)]
        if lastp > seg[0]:
            continue
        aa = a[seg]
        hi_hit = lo_hit = None
        for k, p in enumerate(seg):
            if hi_hit is None and h[p] >= ah + 0.5 * aa[k]:
                hi_hit = p
            if lo_hit is None and l[p] <= al - 0.5 * aa[k]:
                lo_hit = p
        if hi_hit is not None and (lo_hit is None or hi_hit < lo_hit):
            ev_hi.append(hi_hit)
        elif lo_hit is not None:
            ev_lo.append(lo_hit)
    for key, evs in (("lon_sweep_hi(short)", ev_hi), ("lon_sweep_lo(long)", ev_lo)):
        if not evs:
            continue
        rr = []
        for hor in (2, 4, 6):
            vals = [-fwd_ret(df, p, hor) / max(a[p], 1e-9) if "hi" in key
                    else fwd_ret(df, p, hor) / max(a[p], 1e-9) for p in evs]
            vals = np.array(vals)
            rr.append(f"h{hor}: med={np.median(vals):+.2f}ATR wr={(vals>0).mean()*100:.0f}%")
        print(f"{key:22s} n={len(evs):4d}  " + "  ".join(rr))

    # ---- 4. NY open (13-15) sweep of London morning (07-12) extreme ----
    lon2 = {}
    lm = (hrs >= 7) & (hrs < 13)
    for did in np.unique(ids[lm]):
        pos = np.where(lm & (ids == did))[0]
        if len(pos) >= 2:
            lon2[int(did)] = (h[pos].max(), l[pos].min(), pos[-1])
    ev_hi, ev_lo = [], []
    ny_m = (hrs >= 13) & (hrs < 16)
    for did in np.unique(ids[ny_m]):
        seg = np.where(ny_m & (ids == did))[0]
        if len(seg) < 2 or int(did) not in lon2:
            continue
        lh, ll, lastp = lon2[int(did)]
        if lastp > seg[0]:
            continue
        aa = a[seg]
        hi_hit = lo_hit = None
        for k, p in enumerate(seg):
            if hi_hit is None and h[p] >= lh + 0.25 * aa[k]:
                hi_hit = p
            if lo_hit is None and l[p] <= ll - 0.25 * aa[k]:
                lo_hit = p
        if hi_hit is not None and (lo_hit is None or hi_hit < lo_hit):
            ev_hi.append(hi_hit)
        elif lo_hit is not None:
            ev_lo.append(lo_hit)
    for key, evs in (("ny_sweep_hi(short)", ev_hi), ("ny_sweep_lo(long)", ev_lo)):
        if not evs:
            continue
        rr = []
        for hor in (2, 4, 6):
            vals = [-fwd_ret(df, p, hor) / max(a[p], 1e-9) if "hi" in key
                    else fwd_ret(df, p, hor) / max(a[p], 1e-9) for p in evs]
            vals = np.array(vals)
            rr.append(f"h{hor}: med={np.median(vals):+.2f}ATR wr={(vals>0).mean()*100:.0f}%")
        print(f"{key:22s} n={len(evs):4d}  " + "  ".join(rr))


if __name__ == "__main__":
    for s in SYMBOLS:
        analyze(s)
