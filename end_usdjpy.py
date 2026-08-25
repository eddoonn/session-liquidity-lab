"""Morning cleanup for USDJPY book. Run 09:05 UTC daily.
Cancels resting orders, market-closes any filled position (time exit = 06:00 bar close
in backtest; live closes at next opportunity — documented deviation)."""
import argparse

import lab_live as L

BOOK = "USDJPY"


def order_state(cli, o):
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

    st = None
    for back in range(0, 3):   # today + last 2 days (overnight sessions)
        day = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=back)).strftime("%Y-%m-%d")
        cand = L.load_session(BOOK, day=day)
        if cand and cand.get("status") != "closed":
            st = cand
            break
    if not st:
        print("no open session on file — nothing to do")
        return
    if st.get("status") == "closed":
        print("session already closed — idempotent skip")
        return

    from etoro_client import EtoroClient
    cli = None if args.dry_run else EtoroClient()
    actions = []
    for o in st["orders"]:
        if o.get("done"):
            continue
        if args.dry_run:
            actions.append(f"(dry) would resolve {o['transaction']} {o['trigger']:.3f}")
            continue
        state, pid = order_state(cli, o)
        if state == "pending":
            cli.cancel_order(o["order_id"])
            actions.append(f"cancelled pending {o['transaction']}")
            o["done"] = True
        elif state == "filled":
            cli.close_position(pid, o.get("instrument_id"))
            actions.append(f"time-closed {o['transaction']}")
            o["done"] = True
            L.append_fill_log({"ts": pd.Timestamp.now(tz="UTC"), "book": BOOK,
                               "event": "time_close", "side": o["transaction"],
                               "position_id": pid})
        else:
            o["done"] = True
    st["status"] = "closed"
    L.save_session(st, BOOK)
    for a in actions:
        print(a)
    if args.dry_run:
        print("DRY RUN — no live actions")
    elif actions:
        L.notify("USDJPY session closed", "\n".join(actions), color=0xF39C12)


if __name__ == "__main__":
    import pandas as pd  # noqa: F401  (used in append_fill_log timestamps)
    main()
