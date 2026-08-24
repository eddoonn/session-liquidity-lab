"""Place GOLD NY-Judas v3 orders. Run 13:05 UTC Mon-Thu.
Ref: London high/low 06-13 UTC. Sell MIT @ H+0.25*ATR10 (SL 0.75*ATR, TP 1R),
Buy MIT @ L-0.25*ATR. BE handled by manage_nyjudas.py; flat by 19:00 UTC."""
import argparse
import sys

import pandas as pd

import lab_live as L

BUF, MULT, TP = 0.25, 0.75, 1.0
BOOK = "NYJUDAS"
SYMBOL_YF = "GC=F"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    L.load_env()

    now = pd.Timestamp.now(tz="UTC")
    if now.weekday() > 3:
        print(f"weekday={now.weekday()} — Mon-Thu book, skipping")
        return
    if L.load_session(BOOK):
        print("session already placed today — idempotent skip")
        return

    df = L.load_hourly(SYMBOL_YF)
    hi, lo = L.window_levels(df, 6, 13)
    atr = L.atr_last(df, 10)
    sell_trig = hi + BUF * atr          # fade the upside sweep
    buy_trig = lo - BUF * atr           # fade the downside sweep
    stop_d = MULT * atr
    tp_d = TP * stop_d
    account = float(L.os.environ.get("ACCOUNT_SIZE", "10000"))
    risk_usd = account * float(L.os.environ.get("RISK_PCT", "1.0")) / 100.0

    units = round(risk_usd / stop_d, 4)
    state = {"book": BOOK, "day": now.strftime("%Y-%m-%d"), "atr": atr,
             "ref_hi": hi, "ref_lo": lo,
             "orders": [
                 {"transaction": "sellShort", "trigger": sell_trig,
                  "sl": sell_trig + stop_d, "tp": sell_trig - tp_d, "units": units},
                 {"transaction": "buy", "trigger": buy_trig,
                  "sl": buy_trig - stop_d, "tp": buy_trig + tp_d, "units": units}],
             "be_trigger": 0.6, "status": "placed"}
    print(f"ATR10 {atr:.3f} | ref H {hi:.2f} L {lo:.2f}")
    print(f"SELL MIT {sell_trig:.2f} (SL {state['orders'][0]['sl']:.2f} "
          f"TP {state['orders'][0]['tp']:.2f})")
    print(f"BUY  MIT {buy_trig:.2f} (SL {state['orders'][1]['sl']:.2f} "
          f"TP {state['orders'][1]['tp']:.2f}) | units {units}")

    if args.dry_run:
        print("DRY RUN — nothing placed")
        return

    from etoro_client import EtoroClient
    cli = EtoroClient()
    inst = cli.resolve(os.environ.get("ETORO_GOLD_SYMBOL", "GOLD.24-7"))
    pf = cli.portfolio()
    pending = [o for o in (pf.get("clientPortfolio", {}).get("orders") or [])
               if o.get("instrumentID") == inst["instrumentId"]]
    if pending:
        print("orders already resting on eToro — skip")
        return

    for o in state["orders"]:
        r = cli.place_mit(inst, o["transaction"], o["trigger"], o["sl"], o["tp"], o["units"],
                          leverage=5)
        o["order_id"] = r.get("orderId")
        o["reference_id"] = r.get("referenceId")
        L.append_fill_log({"ts": pd.Timestamp.now(tz="UTC"), "book": BOOK, "event": "place",
                           "side": o["transaction"], "price": o["trigger"], "sl": o["sl"],
                           "tp": o["tp"], "units": o["units"], "order_id": o.get("order_id")})
    L.save_session(state, BOOK)
    L.notify(f"GOLD NY-Judas armed — {now.strftime('%a %d %b')}",
             f"SELL MIT {sell_trig:.2f} · BUY MIT {buy_trig:.2f}\n"
             f"SL 0.75×ATR ({stop_d:.2f}) · TP 1R · BE at +0.6R close · flat 19:00 UTC",
             fields=[{"name": "Risk", "value": f"{risk_usd:.0f} USD ({units} oz)", "inline": True},
                     {"name": "ATR10", "value": f"{atr:.2f}", "inline": True}])


if __name__ == "__main__":
    sys.exit(main())
