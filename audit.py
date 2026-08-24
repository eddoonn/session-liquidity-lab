"""Audit 1/4 v2: engine ground-truth tests, corrected scenarios."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Data, run_pattern

PASS, FAIL = 0, []


def check(name, cond, info=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {info}")


def mkdf(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC")
    return pd.DataFrame(np.array(rows, dtype=float), index=idx,
                        columns=["Open", "High", "Low", "Close"])


def spec(**kw):
    s = dict(ref_win=(0, 2), ref_off=0, act_win=(2, 6), mode="rev", trigger="sweep",
             confirm=False, buf=0.0, atr_len=10, mult=1.0, tp=1.0, exit_hour=8, max_hold=36)
    s.update(kw)
    return s


def run(rows, sp, cost=0.0):
    return run_pattern(Data(mkdf(rows), atr_lens=(10,)), sp, cost=cost)


WARM = [(100.0, 100.06, 99.94, 100.0)] * 4          # idx0-1 = ref window


def test_level_fill_short_tp():
    # small penetration so self-inclusive ATR stays below penetration (no same-bar SL);
    # next bar breaks down -> TP
    rows = WARM + [(100, 100.04, 99.96, 100),
                   (100.02, 100.11, 100.01, 100.08),
                   (100.02, 100.03, 99.80, 99.85),
                   (99.85, 99.90, 99.80, 99.85)]
    tr = run(rows, spec())
    ok = len(tr) == 1 and tr[0]["side"] == "short" and tr[0]["reason"] == "tp"
    check("limit fill @level + short TP", ok, str(tr))
    if tr:
        check("entry == ref high", abs(tr[0]["entry"] - 100.06) < 1e-9, tr[0]["entry"])


def test_sl_priority_same_bar():
    # wide-bottom reference so only the HIGH sweep triggers, yet the same bar
    # spans both SL (above) and TP (below) -> conservative order must take SL
    warm = [(100.0, 100.06, 99.50, 100.0)] * 4
    rows = warm + [(100, 100.04, 99.52, 100),
                   (100.02, 100.80, 99.60, 100.05)]
    tr = run(rows, spec())
    ok = len(tr) == 1 and tr[0]["reason"] == "sl"
    check("sl-before-tp same bar", ok, str(tr))


def test_ambiguous_double_sweep_skip():
    rows = WARM + [(100, 100.04, 99.96, 100),
                   (100.02, 101.30, 99.30, 100.0)]
    tr = run(rows, spec())
    check("double-sweep same bar -> skip", len(tr) == 0, str(tr))


def test_time_exit_and_max_hold():
    rows = WARM + [(100, 100.04, 99.96, 100),
                   (100.02, 100.07, 100.00, 100.04)]
    rows += [(100, 100.06, 99.98, 100.0)] * 8           # through hour 13; hour-8 exists
    tr = run(rows, spec(tp=5.0, mult=0.2))
    check("time exit at exit_hour", len(tr) == 1 and tr[0]["reason"] == "time"
          and tr[0]["exit_time"].hour == 8, str(tr))
    tr2 = run(rows, spec(tp=5.0, mult=0.2, max_hold=2))
    check("max_hold closes early", len(tr2) == 1 and tr2[0]["exit_time"]
          == tr2[0]["entry_time"] + pd.Timedelta(hours=2) and tr2[0]["reason"] == "eod", str(tr2))


def test_cont_gap_fill_at_open():
    rows = WARM + [(100, 100.05, 99.95, 100.90),
                   (101.40, 102.20, 101.30, 101.80)]
    tr = run(rows, spec(mode="cont"))
    ok = len(tr) == 1 and tr[0]["side"] == "long" and abs(tr[0]["entry"] - 101.40) < 1e-9
    check("stop gaps: filled at open (worse)", ok, str(tr))


def test_cont_hole_fill_at_level():
    # hours 0,1,2,[hole 3-4],5,6: pre-break hour2, breakout hour5 opens far above level;
    # 3h data hole => crossing happened inside hole => resting stop fills AT LEVEL
    rows = WARM[:2] + [(100, 100.05, 99.95, 100.50),
                       (101.40, 102.20, 101.30, 101.80),
                       (101.60, 102.00, 101.40, 101.70),
                       (101.70, 102.10, 101.50, 101.90)]
    idx_hours = [0, 1, 2, 5, 6, 7]
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")[idx_hours]
    df = pd.DataFrame(np.array(rows, dtype=float), index=idx,
                      columns=["Open", "High", "Low", "Close"])
    tr = run_pattern(Data(df, atr_lens=(10,)), spec(mode="cont"), cost=0.0)
    ok = len(tr) == 1 and abs(tr[0]["entry"] - 100.06) < 1e-9
    check("data-hole: filled at level", ok, str(tr))


def test_rev_gap_improves():
    rows = WARM + [(100, 100.04, 99.96, 100),
                   (100.60, 101.20, 100.55, 100.70)]
    tr = run(rows, spec())
    ok = len(tr) == 1 and abs(tr[0]["entry"] - 100.60) < 1e-9
    check("limit gaps: filled at better open", ok, str(tr))


def test_confirm_mode_next_open():
    good = WARM + [(100, 100.04, 99.96, 100),
                   (100.02, 101.00, 100.01, 100.02),
                   (99.98, 100.00, 99.90, 99.95)]
    tr = run(good, spec(confirm=True))
    ok = len(tr) == 1 and abs(tr[0]["entry"] - 99.98) < 1e-9
    check("confirm: next-open entry", ok, str(tr))

    bad = WARM + [(100, 100.04, 99.96, 100),
                  (100.02, 101.00, 100.01, 100.50)]
    tr2 = run(bad, spec(confirm=True))
    check("confirm: close outside -> rejected", len(tr2) == 0, str(tr2))


def test_close_beyond_continuation_long():
    sp = spec(mode="cont", trigger="close_beyond")
    rows = WARM + [(100, 100.05, 99.95, 100.20),
                   (100.21, 100.30, 100.10, 100.28)]
    tr = run(rows, sp)
    check("close_beyond cont long", len(tr) == 1 and tr[0]["side"] == "long", str(tr))


def test_ref_fallback_across_missing_day():
    # day2 absent entirely; ref_off=1 falls back to day1's levels
    rows = WARM + [(100, 100.04, 99.96, 100)]
    idx1 = pd.date_range("2026-01-01", periods=len(rows), freq="1h", tz="UTC")
    idx2 = pd.date_range("2026-01-03", periods=5, freq="1h", tz="UTC")
    extra = [(100.10, 100.16, 100.04, 100.10),
             (100.10, 100.14, 100.06, 100.10),
             (100.10, 100.14, 100.06, 100.10),
             (100.12, 100.90, 100.11, 100.60),
             (100.12, 100.13, 99.98, 100.00)]
    df = pd.DataFrame(np.array(rows + extra, dtype=float), index=idx1.append(idx2),
                      columns=["Open", "High", "Low", "Close"])
    tr = run_pattern(Data(df, atr_lens=(10,)), spec(ref_off=1, act_win=(2, 5)), cost=0.0)
    # ref = day1 levels (skips absent day2); day3 opens gapped above -> better limit fill
    ok = len(tr) == 1 and abs(tr[0]["entry"] - 100.10) < 1e-9 \
        and tr[0]["side"] == "short" and str(tr[0]["date"]) == "2026-01-03"
    check("ref skips missing day -> older day levels", ok, str(tr))


def test_no_ref_same_day_future():
    # ref window AFTER action start must never be used
    sp = spec(ref_win=(4, 6), ref_off=0)
    rows = WARM[:2] + [(100, 100.90, 99.90, 100.10),
                       (100, 100.05, 99.95, 100.0)] * 4
    tr = run(rows, sp)
    check("no future reference leakage", len(tr) == 0, str(tr))


def test_phantom_tp_rejected_by_path_aware():
    # Bullish entry bar: dips to 99.97 first (below future tp), THEN sweeps the
    # level late in the bar. Naive engine credits the pre-entry dip as TP;
    # path-aware splits the bar at the fill and rejects it -> time exit later.
    rows = WARM + [(100, 100.04, 99.96, 100),
                   (100.00, 100.11, 99.97, 100.05)]
    rows += [(100.03, 100.05, 100.02, 100.04)] * 4      # clean drift through hour 8
    naive = run(rows, spec(mult=0.45))
    pa = run(rows, spec(mult=0.45, path_aware=True))
    check("phantom same-bar TP fired in naive", len(naive) == 1
          and naive[0]["reason"] == "tp", str(naive))
    check("path-aware rejects pre-entry TP", len(pa) == 1
          and pa[0]["reason"] == "time" and pa[0]["r"] < naive[0]["r"], str(pa))


def test_pathaware_keeps_genuine_samebar_tp():
    # Bar opens above the level (short fills at open), then declines through tp
    # without ever touching sl -> genuine same-bar TP path-aware must keep.
    warm = [(100.0, 100.06, 99.94, 100.0)] * 4
    rows = warm + [(100.00, 100.04, 99.96, 100.00),
                   (100.08, 100.09, 99.95, 99.96)]
    pa = run(rows, spec(mult=0.3))
    ok = len(pa) == 1 and pa[0]["reason"] == "tp"
    check("path-aware keeps genuine same-bar TP", ok, str(pa))


if __name__ == "__main__":
    print(f"{'=' * 62}\nENGINE GROUND-TRUTH TESTS v2\n{'=' * 62}")
    test_level_fill_short_tp()
    test_sl_priority_same_bar()
    test_ambiguous_double_sweep_skip()
    test_time_exit_and_max_hold()
    test_cont_gap_fill_at_open()
    test_cont_hole_fill_at_level()
    test_rev_gap_improves()
    test_confirm_mode_next_open()
    test_close_beyond_continuation_long()
    test_ref_fallback_across_missing_day()
    test_no_ref_same_day_future()
    test_phantom_tp_rejected_by_path_aware()
    test_pathaware_keeps_genuine_samebar_tp()
    print(f"\n{PASS} passed / {len(FAIL)} failed {FAIL if FAIL else ''}")
