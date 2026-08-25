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
    else:
        print(f"no closed trades since {args.min_date}")

    with open(os.path.join(L.LIVE, "history_raw.json"), "w") as f:
        json.dump(hist, f, indent=2, default=str)


if __name__ == "__main__":
    main()
