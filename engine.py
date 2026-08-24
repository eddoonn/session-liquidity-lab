"""Generic session-liquidity pattern engine.

Concept: markets leave resting liquidity at prior session extremes. Sessions sweep
those extremes; price either snaps back (reversal) or accelerates (continuation).
This engine parameterizes that entire hypothesis space and simulates honestly
(stop entries at trigger level, SL-before-TP intrabar priority, time exits, costs).
"""
import numpy as np
import pandas as pd


def atr_values(df, n=10):
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean().values


class Data:
    """Precomputed arrays + cached level maps & action segments."""

    def __init__(self, df, atr_lens=(10,)):
        self.index = df.index
        self.o = df["Open"].values.astype(float)
        self.h = df["High"].values.astype(float)
        self.l = df["Low"].values.astype(float)
        self.c = df["Close"].values.astype(float)
        self.atrs = {n: atr_values(df, n) for n in atr_lens}
        idx = df.index
        self.hrs = np.asarray(idx.hour)
        days = idx.tz_convert("UTC").floor("D")
        codes, _ = pd.factorize(days)
        self.ids = codes.astype(np.int64)
        self.n = len(self.o)
        self._levels_cache = {}
        self._seg_cache = {}

    def _win_mask_ids(self, s, e):
        m = (self.hrs >= s) & (self.hrs < e) if s <= e else (self.hrs >= s) | (self.hrs < e)
        ids = np.where(self.hrs >= s, self.ids + 1, self.ids) if s > e else self.ids
        return m, ids

    def levels(self, s, e):
        """dict[day_code] -> (high, low, last_bar_pos) of hour window [s,e)."""
        key = (s, e)
        if key not in self._levels_cache:
            m, ids = self._win_mask_ids(s, e)
            out = {}
            for did in np.unique(ids[m]):
                pos = np.where(m & (ids == did))[0]
                out[int(did)] = (float(self.h[pos].max()), float(self.l[pos].min()), int(pos[-1]))
            self._levels_cache[key] = out
        return self._levels_cache[key]

    def segments(self, s, e, skip_sunday=True, weekdays=None):
        """chronological [(day_code, pos_array)] of action-window sessions."""
        key = (s, e, skip_sunday, tuple(weekdays) if weekdays else None)
        if key not in self._seg_cache:
            m, ids = self._win_mask_ids(s, e)
            pos_all = np.where(m)[0]
            seg_ids = ids[pos_all]
            out = []
            for did in np.unique(seg_ids):
                seg = pos_all[seg_ids == did]
                if len(seg) < 2:
                    continue
                if skip_sunday and self.index[seg[0]].weekday() == 6:
                    continue
                if weekdays and self.index[seg[0]].weekday() not in weekdays:
                    continue
                out.append((int(did), seg))
            self._seg_cache[key] = out
        return self._seg_cache[key]


def _bar_legs(o_, h_, l_, c_):
    """Chronological price legs of a bar inferred from open/close polarity.
    Bullish close -> O-L-H-C ; bearish -> O-H-L-C."""
    if c_ >= o_:
        return [(o_, l_), (l_, h_), (h_, c_)]
    return [(o_, h_), (h_, l_), (l_, c_)]


def _leg_hits(lo, hi, x):
    return min(lo, hi) - 1e-12 <= x <= max(lo, hi) + 1e-12


def run_pattern(d: Data, spec, cost=0.0):
    """spec keys:
    ref_win (s,e), ref_off int, act_win (s,e),
    mode 'rev'|'cont', trigger 'sweep'|'close_beyond',
    buf float (xATR), atr_len int, mult float,
    tp float|'opp', rr ignored if opp, exit_hour int, max_hold int
    """
    o, h, l, c = d.o, d.h, d.l, d.c
    atr = d.atrs[spec["atr_len"]]
    hrs = d.hrs
    lv_all = d.levels(*spec["ref_win"])
    segs = d.segments(*spec["act_win"], weekdays=spec.get("weekdays"))
    off = spec["ref_off"]
    rev = spec["mode"] == "rev"
    cb = spec.get("trigger", "sweep") == "close_beyond"
    need_confirm = spec.get("confirm", False)
    max_hold = spec.get("max_hold", 36)
    trades = []
    # causal volatility-regime percentile (trailing 500h rank, shifted 1 bar)
    vp = spec.get("atr_pct")
    atr_pct = None
    if vp is not None:
        key = ("pct", spec["atr_len"])
        if key not in d.atrs:
            d.atrs[key] = pd.Series(d.atrs[spec["atr_len"]]).rolling(500).rank(
                pct=True).shift(1).values
        atr_pct = d.atrs[key]

    for did, seg in segs:
        first_pos = seg[0]
        ref = None
        for back in range(off, off + 7):
            cand = lv_all.get(did - back)
            if cand and cand[2] < first_pos:
                ref = cand
                break
        if ref is None:
            continue
        rh, rl, _ = ref
        # optional level-quality gate: reference range must be sane vs ATR
        rq = spec.get("ref_ratio")
        if rq and first_pos > 0:
            ratio = (rh - rl) / atr[first_pos]
            if not np.isfinite(ratio) or not (rq[0] <= ratio <= rq[1]):
                continue
        entry = None
        for k, p in enumerate(seg):
            a = atr[p]
            if not np.isfinite(a) or a <= 0:
                continue
            if atr_pct is not None and not np.isfinite(atr_pct[p]):
                continue
            if vp is not None and not (vp[0] <= atr_pct[p] <= vp[1]):
                continue
            bufd = spec["buf"] * a
            up = h[p] >= rh + bufd          # swept the high
            dn = l[p] <= rl - bufd          # swept the low
            if cb:
                up = c[p] > rh
                dn = c[p] < rl
            if up and dn:
                continue
            if not (up or dn):
                continue
            swept_high = bool(up)
            if rev:
                side_short = swept_high      # fade the sweep
            else:
                side_short = not swept_high  # ride the break
            entry_style = spec.get("entry", "level")
            if need_confirm or entry_style == "next_open":
                # rejection: bar must close back inside (rev) / accept beyond (cont)
                if need_confirm:
                    if rev:
                        ok = c[p] <= rh if side_short else c[p] >= rl
                    else:
                        ok = c[p] >= rh if not side_short else c[p] <= rl
                    if not ok:
                        continue
                if p + 1 >= d.n:
                    continue
                epos, entry_px = p + 1, float(o[p + 1])
                start_leg = 0
            else:
                lvl = (rh + bufd) if swept_high else (rl - bufd)
                # Contiguous gap -> stop/limit fills at the open (better/worse per order type).
                # Data hole -> price crossed the level while bars were missing: fill at level.
                contig = p == 0 or (d.index[p] - d.index[p - 1]) <= pd.Timedelta(hours=2)
                if contig:
                    if rev:  # resting limits: gap through level fills at better open
                        fill = max(lvl, o[p]) if side_short else min(lvl, o[p])
                    else:    # stop orders: gap through level fills at worse open
                        fill = max(lvl, o[p]) if not side_short else min(lvl, o[p])
                    start_leg = 0 if fill == o[p] else None  # locate leg later
                else:
                    fill = lvl
                    start_leg = 0            # position existed before this bar
                epos, entry_px = p, float(fill)
            risk = spec["mult"] * a
            if side_short:
                sl = entry_px + risk
                tp = rl if spec["tp"] == "opp" else entry_px - spec["tp"] * risk
            else:
                sl = entry_px - risk
                tp = rh if spec["tp"] == "opp" else entry_px + spec["tp"] * risk
            entry = (side_short, epos, entry_px, float(sl), float(tp), float(risk), start_leg)
            break
        if entry is None:
            continue

        side_short, epos, entry_px, sl, tp, risk, start_leg = entry
        path = bool(spec.get("path_aware", False))
        result_r = exit_px = exit_pos = reason = None
        mae = mfe = 0.0
        last_pos = min(epos + max_hold, d.n - 1)
        first_exit_pos = epos if spec.get("same_bar_exits", True) else epos + 1
        # breakeven: once a bar CLOSES with >= be_trigger*R profit, SL moves to entry
        # (+ be_lock*R locked profit)
        be = spec.get("be_trigger")
        be_lock = spec.get("be_lock", 0.0)
        be_armed = not be
        cur_sl = sl

        def _close_out(p_, px_, why_):
            gross = (entry_px - px_) if side_short else (px_ - entry_px)
            return ((gross - cost) / risk, px_, p_, why_)

        if not path:
            for p in range(first_exit_pos, d.n):
                if side_short:
                    mae = max(mae, h[p] - entry_px)
                    mfe = max(mfe, entry_px - l[p])
                    hit_sl, hit_tp = h[p] >= cur_sl, l[p] <= tp
                else:
                    mae = max(mae, entry_px - l[p])
                    mfe = max(mfe, h[p] - entry_px)
                    hit_sl, hit_tp = l[p] <= cur_sl, h[p] >= tp
                if hit_sl:
                    result_r, exit_px, exit_pos, reason = _close_out(p, cur_sl, "sl")
                    break
                if hit_tp:
                    result_r, exit_px, exit_pos, reason = _close_out(p, tp, "tp")
                    break
                if hrs[p] == spec["exit_hour"] and p > epos:
                    result_r, exit_px, exit_pos, reason = _close_out(p, c[p], "time")
                    break
                if p >= last_pos:
                    result_r, exit_px, exit_pos, reason = _close_out(p, c[p], "eod")
                    break
                if not be_armed:
                    prog = (entry_px - c[p]) if side_short else (c[p] - entry_px)
                    if prog >= be * risk:
                        be_armed = True
                        lock = be_lock * risk
                        cur_sl = (entry_px - lock) if side_short else (entry_px + lock)
        else:
            # OHLC path-aware walk: legs after the fill only; SL wins inside a leg.
            # The leg containing the fill is split AT the fill price so pre-entry
            # extremes inside that leg cannot trigger exits.
            split_leg = None
            if start_leg is None:
                lg0 = _bar_legs(o[epos], h[epos], l[epos], c[epos])
                li = next((i for i, (a_, b_) in enumerate(lg0)
                           if _leg_hits(a_, b_, entry_px)), None)
                if li is None:
                    start_leg = 0
                else:
                    prev_end = lg0[li - 1][1] if li > 0 else o[epos]
                    a_, b_ = lg0[li]
                    split_leg = (entry_px, b_) if prev_end <= entry_px else (a_, entry_px)
                    start_leg = li
            for p in range(max(first_exit_pos, epos), d.n):
                lg = _bar_legs(o[p], h[p], l[p], c[p])
                if p == epos and split_leg is not None:
                    lg = lg[:start_leg] + [split_leg] + lg[start_leg + 1:]
                hit = None
                for li in range(0, len(lg)):
                    a_, b_ = lg[li]
                    lo_, hi_ = (a_, b_) if a_ <= b_ else (b_, a_)
                    mae = max(mae, hi_ - entry_px if side_short else entry_px - lo_)
                    mfe = max(mfe, entry_px - lo_ if side_short else hi_ - entry_px)
                    sl_hit = _leg_hits(lo_, hi_, cur_sl)
                    tp_hit = _leg_hits(lo_, hi_, tp)
                    if sl_hit:
                        hit = (cur_sl, "sl")
                        break
                    if tp_hit:
                        hit = (tp, "tp")
                        break
                if hit:
                    result_r, exit_px, exit_pos, reason = _close_out(p, hit[0], hit[1])
                    break
                if hrs[p] == spec["exit_hour"] and p > epos:
                    result_r, exit_px, exit_pos, reason = _close_out(p, c[p], "time")
                    break
                if p >= last_pos:
                    result_r, exit_px, exit_pos, reason = _close_out(p, c[p], "eod")
                    break
                if not be_armed:
                    prog = (entry_px - c[p]) if side_short else (c[p] - entry_px)
                    if prog >= be * risk:
                        be_armed = True
                        lock = be_lock * risk
                        cur_sl = (entry_px - lock) if side_short else (entry_px + lock)
        if result_r is None:
            p = d.n - 1
            result_r, exit_px, exit_pos, reason = _close_out(p, c[p], "eod")

        trades.append({
            "date": d.index[seg[0]].date(), "side": "short" if side_short else "long",
            "entry_time": d.index[epos], "entry": round(entry_px, 5),
            "sl": round(sl, 5), "tp": round(tp, 5),
            "exit_time": d.index[exit_pos], "exit": round(exit_px, 5),
            "r": round(result_r, 3), "reason": reason,
            "mae_r": round(mae / risk, 2), "mfe_r": round(mfe / risk, 2),
        })
    return trades


def summarize(trades):
    if not trades:
        return {"trades": 0}
    r = np.array([t["r"] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    gw = wins.sum() if len(wins) else 0.0
    gl = abs(losses.sum()) if len(losses) else 0.0
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    streak = 0
    mxw = mxl = 0
    for x in r:
        if x > 0:
            streak = streak + 1 if streak > 0 else 1
        elif x <= 0:
            streak = streak - 1 if streak < 0 else -1
        mxw, mxl = max(mxw, streak), min(mxl, streak)
    return {
        "trades": len(r),
        "win_rate_pct": round(100 * len(wins) / len(r), 2),
        "total_r": round(float(r.sum()), 2),
        "avg_r": round(float(r.mean()), 3),
        "profit_factor": round(gw / gl, 2) if gl > 0 else 99.0,
        "max_dd_r": round(dd, 2),
        "longs": sum(1 for t in trades if t["side"] == "long"),
        "shorts": sum(1 for t in trades if t["side"] == "short"),
    }


def segment_r(trades, n_seg=3):
    if not trades:
        return []
    times = np.array([t["entry_time"].value for t in trades])
    rs = np.array([t["r"] for t in trades])
    order = np.argsort(times)
    times, rs = times[order], rs[order]
    edges = np.quantile(times, np.linspace(0, 1, n_seg + 1))
    return [float(rs[(times >= edges[i]) & (times < edges[i + 1] + (1 if i == n_seg - 1 else 0))].sum())
            for i in range(n_seg)]
