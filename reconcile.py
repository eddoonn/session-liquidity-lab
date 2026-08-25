"""Daily reconciliation: pull closed trades from eToro demo and append to
results/live/closed_trades.csv. Run 10:30 UTC daily (or manually).
This is the fill-quality dataset: compare against sim_fine predictions to
calibrate your broker's real execution tax before scaling up."""
import argparse
import json
import os

import pandas as pd

import lab_live as L

BOOK = "RECONCILE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", default=(pd.Timestamp.now(tz="UTC") -
                                           pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
    args = ap.parse_args()
    L.load_env()

    from etoro_client import EtoroClient
    cli = EtoroClient()
    hist = cli.history(min_date=args.min_date)
    if isinstance(hist, dict):
        trades = (hist.get("trades") or hist.get("TradingProxies") or [])
    elif isinstance(hist, list):
        trades = hist
    else:
        trades = []
    rows = []
    for t in trades:
        rows.append({
            "position_id": t.get("positionId") or t.get("PositionID"),
            "instrument": t.get("instrument") or t.get("SymbolName"),
            "is_buy": t.get("isBuy", t.get("IsBuy")),
            "open_rate": t.get("openRate") or t.get("OpenRate"),
            "close_rate": t.get("closeRate") or t.get("CloseRate"),
            "open_date": t.get("openDate") or t.get("OpenDateTime"),
            "close_date": t.get("closeDate") or t.get("CloseDateTime"),
            "profit_usd": t.get("netProfit") or t.get("Profit"),
        })
    out_path = os.path.join(L.LIVE, "closed_trades.csv")
    if rows:
        new_df = pd.DataFrame(rows)
        if os.path.exists(out_path):
            old = pd.read_csv(out_path, dtype=str)
            combined = pd.concat([old, new_df.astype(str)]).drop_duplicates(
                subset=["position_id", "close_date"], keep="last")
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        print(f"reconciled {len(rows)} records -> {out_path} ({len(combined)} total)")

        # ---- daily closed-P&L notification ----
        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        recent = [r for r in rows if str(r.get("close_date", ""))[:10] >= cutoff]
        if recent:
            net = sum(float(r.get("profit_usd") or 0) for r in recent)
            fields = [{"name": f"{r.get('instrument', '?')} "
                               f"{'LONG' if str(r.get('is_buy')) in ('True', 'true', '1') else 'SHORT'}",
                       "value": f"{r.get('open_rate')} -> {r.get('close_rate')} · "
                                f"**{float(r.get('profit_usd') or 0):+.2f} USD** · "
                                f"closed {str(r.get('close_date'))[:16]}Z",
                       "inline": False}
                      for r in sorted(recent, key=lambda x: float(x.get("profit_usd") or 0))]
            L.notify(f"Closed-trades P&L — last 48h ({len(recent)} trades)",
                     f"net {net:+.2f} USD",
                     color=0x2ECC71 if net > 0 else (0xE74C3C if net < 0 else 0x95A5A6),
                     fields=fields[:20])
    else:
        print(f"no closed trades since {args.min_date}")

    with open(os.path.join(L.LIVE, "history_raw.json"), "w") as f:
        json.dump(hist, f, indent=2, default=str)


if __name__ == "__main__":
    main()
