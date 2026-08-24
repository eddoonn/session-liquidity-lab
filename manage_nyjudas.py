"""Hourly manager for GOLD NY-Judas v3. Run at :05 after 14,15,16,17,18 UTC + 19:05.
- after each hourly close: if a bar CLOSED with >=0.6R progress and position open ->
  market-close (BE lock; documented deviation from backtest stop-move)
- final pass: cancel resting orders, close any open position (time exit 18:00 bar close)
Idempotent via session state."""
import argparse

import pandas as pd

import lab_live as L

BOOK = "NYJUDAS"


def order_state(cli, o):
    """resolve an order into ('pending'|'filled'|'gone', position_id)"""
    try:
        info = cli.lookup(o["order_id"], reference_id=o.get("reference_id"))
    except Exception as e:
        print(f"lookup failed {o['order_id']}: {e}")
        return "unknown", None
    if isinstance(info, list) and info:
        info = info[0]
    status = str(info.get("status", "")).lower()
    pid = (info.get("positionID") or info.get("positionId")
           or (info.get("position") or {}).get("positionID"))
    if status in ("open", "executed", "filled") or pid:
        return "filled", pid
    if status in ("pending", "waiting", "ordered"):
        return "pending", None
    return "gone", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    L.load_env()

    now = pd.Timestamp.now(tz="UTC")
    if not (14 <= now.hour <= 19):
        print("outside management window (14:05-19:05 UTC) — no-op")
        return
    st = L.load_session(BOOK)
    if not st:
        print("no session today — nothing to manage")
        return
    if st.get("status") == "closed":
        print("session already closed — idempotent skip")
        return
    final = now.hour >= 19

    df = L.load_hourly("GC=F")
    # latest fully-closed hourly bar (Yahoo may have rolled a forming bar already)
    if df.index[-1].hour == now.hour:
        bar = df.iloc[-2]
    else:
        bar = df.iloc[-1]

    from etoro_client import EtoroClient
    cli = None

    actions = []
    for o in st["orders"]:
        if o.get("done"):
            continue
        if args.dry_run:
            actions.append(f"(dry) would check {o['transaction']} {o['trigger']:.2f}")
            continue
        cli = cli or EtoroClient()
        state, pid = order_state(cli, o)
        if state == "pending":
            if final:
                cli.cancel_order(o["order_id"])
                actions.append(f"cancelled pending {o['transaction']}")
                o["done"] = True
            continue
        if state != "filled":
            o["done"] = True
            continue
        o["position_id"] = pid
        entry = float(o["trigger"])
        short = o["transaction"] == "sellShort"
        prog = ((entry - float(bar["Close"])) if short else (float(bar["Close"]) - entry))
        risk_d = abs(entry - float(o["sl"]))
        be_trig = st.get("be_trigger", 0.6) * risk_d
        if not final and prog >= be_trig and not o.get("be_done"):
            cli.close_position(pid, o.get("instrument_id"))
            actions.append(f"BE-close {o['transaction']} @~{bar['Close']:.2f} "
                           f"(+{prog/risk_d:.2f}R)")
            o.update(done=True, be_done=True, exit_reason="be")
            L.append_fill_log({"ts": now, "book": BOOK, "event": "be_close",
                               "side": o["transaction"], "price": float(bar["Close"]),
                               "r_approx": round(prog / risk_d, 3), "position_id": pid})
        elif final:
            cli.close_position(pid, o.get("instrument_id"))
            actions.append(f"time-exit {o['transaction']} @~{bar['Close']:.2f}")
            o.update(done=True, exit_reason="time")
            L.append_fill_log({"ts": now, "book": BOOK, "event": "time_close",
                               "side": o["transaction"], "price": float(bar["Close"]),
                               "position_id": pid})
    if final:
        st["status"] = "closed"
    L.save_session(st, BOOK)
    for a in actions:
        print(a)
    if args.dry_run:
        print("DRY RUN — no live actions")
    elif actions:
        L.notify(f"GOLD NY-Judas update {now.strftime('%H:%M')} UTC",
                 "\n".join(actions), color=0xF39C12)


if __name__ == "__main__":
    main()
