import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

mine = pd.read_csv("results/detailed/trades2026_EURUSD_BASE_A.csv")
mine["entry_time"] = pd.to_datetime(mine["entry_time"], utc=True)
mine["month"] = mine["entry_time"].dt.strftime("%Y-%m")
mm = mine.groupby("month")["r"].agg(["count", "sum"]).round(2)
print("MINE (config A generalized, cost 0.0002):")
print(mm.to_string())
theirs = pd.read_csv(r"..\asia-gold-reversal\results\months_2026_eurusd.csv")
print("\nTHEIRS (asia-gold-reversal results):")
print(theirs[["month", "trades", "total_r", "avg_win_r", "avg_loss_r"]].head(8).to_string(index=False))
t = pd.read_csv(r"..\asia-gold-reversal\results\trades_2026_eurusd.csv")
t["h"] = pd.to_datetime(t["entry_time"]).dt.hour
print("\ntheir entry-hour set:", sorted(t["h"].unique()))
mine["h2"] = pd.to_datetime(mine["entry_time"]).dt.hour
print("my entry-hour set:  ", sorted(mine["h2"].unique()))
