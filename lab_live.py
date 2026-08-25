"""Shared utilities for live eToro deployment of the two v3 books."""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE = os.path.join(HERE, "results", "live")
os.makedirs(LIVE, exist_ok=True)


def load_env():
    """local .env first, then fall back to asia-gold-reversal/.env for broker keys"""
    for path in (os.path.join(HERE, ".env"),
                 os.path.join(HERE, "..", "asia-gold-reversal", ".env")):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())


def load_hourly(symbol, period="12d"):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval="60m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df[~df.index.duplicated(keep="first")].sort_index()


def atr_last(df, n=10):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n, min_periods=1).mean().iloc[-1])


def window_levels(df, s, e, day=None, drop_forming=True):
    """high/low over UTC hours [s,e) of ONE calendar day.
    day defaults to the last fully-closed bar's date. Drops the still-forming bar
    so levels match the backtest exactly."""
    idx = df.index
    if drop_forming:
        cutoff = idx[-1].floor("h")
        if idx[-1] >= cutoff and (pd.Timestamp.now(tz="UTC") - cutoff) < pd.Timedelta(hours=1):
            df = df[idx < cutoff]
            idx = df.index
    target = (idx[-1].date() if day is None
              else pd.Timestamp(day).date())
    hrs = idx.hour
    m = (hrs >= s) & (hrs < e) if s <= e else (hrs >= s) | (hrs < e)
    sub = df[m & (idx.normalize() == pd.Timestamp(target, tz="UTC"))]
    if sub.empty:
        raise SystemExit(f"no bars in {s:02d}-{e:02d}h window on {target}")
    return float(sub["High"].max()), float(sub["Low"].min())


def session_path(book, day=None):
    day = day or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    os.makedirs(LIVE, exist_ok=True)
    return os.path.join(LIVE, f"session_{book}_{day}.json")


def load_session(book, day=None):
    p = session_path(book, day)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def save_session(state, book, day=None):
    with open(session_path(book, day), "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_fill_log(record):
    p = os.path.join(LIVE, "fills.csv")
    new = not os.path.exists(p)
    pd.DataFrame([record]).to_csv(p, mode="a", header=new, index=False)


def notify(title, description="", color=0x2F6FDE, fields=None):
    try:
        from discord_push import post_json
        payload = {"embeds": [{"title": title, "description": description,
                               "color": color, "fields": fields or []}]}
        post_json(payload)
    except Exception as e:  # never crash a trading job on notifications
        print(f"(discord notify failed: {e})")


def order_outcome(cli, o, risk_usd):
    """Resolve one session order into a human-readable outcome via eToro records."""
    oid = o.get("order_id")
    try:
        info = cli.lookup(oid, reference_id=o.get("reference_id"))
        if isinstance(info, list) and info:
            info = info[0]
    except Exception:
        info = {}
    status = str(info.get("status", "")).lower()
    pid = (info.get("positionID") or info.get("positionId")
           or (info.get("position") or {}).get("positionID"))
    side = o["transaction"]
    out = {"side": side, "order_id": oid}
    if status in ("pending", "waiting", "ordered"):
        out["state"] = "still resting"
        return out
    try:
        hist = cli.history(min_date="2026-08-01")
    except Exception:
        hist = []
    rec = None
    for t in (hist if isinstance(hist, list) else []):
        if str(t.get("orderId")) == str(oid) or (pid and str(t.get("positionId")) == str(pid)):
            rec = t
            break
    if rec is None:
        out["state"] = "not filled (cancelled/expired)"
        return out
    entry, exitp = float(rec.get("openRate", 0)), float(rec.get("closeRate", 0))
    pnl = float(rec.get("netProfit", 0))
    short = side == "sellShort"
    sl_d = abs(float(o["sl"]) - entry)
    r = ((entry - exitp) if short else (exitp - entry)) / max(sl_d, 1e-9)
    reason = "SL" if ((exitp >= float(o["sl"]) - 1e-9) if short else (exitp <= float(o["sl"]) + 1e-9)) \
        else ("TP" if ((exitp <= float(o["tp"]) + 1e-9) if short else (exitp >= float(o["tp"]) - 1e-9)) else "time/manual")
    return {"side": side, "state": f"{reason} · {entry:.5g} -> {exitp:.5g}",
            "r": round(r, 2), "pnl": round(pnl, 2)}
