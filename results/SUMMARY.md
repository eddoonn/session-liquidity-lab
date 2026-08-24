# Pattern Lab — FINAL research summary (Aug 2026)

## Verification chain (all passed before any number was trusted)
1. Engine: 17/17 synthetic ground-truth tests (fills, ordering, gaps, holes, fallbacks).
2. Baseline reproduction: your GOLD book +35.78R vs published +35.79R.
3. Your EURUSD book reconciled exactly (same 125 trades) at rr=0.75, cost≈0.00005
   → FX books were backtested at near-zero cost; gold at realistic 0.3.
4. Data: 0 dupes / 0 bad ticks / 13–16 holes>3h per instrument in 2026;
   15m and 5m bars aggregate hourly bars EXACTLY (0 mismatches).

## The intrabar execution tax (measured on 5m ground truth, Jul–Aug 2026)
Naive hourly backtests credit TP when the target is touched anywhere in the
entry hour — including BEFORE the fill. True 5m-resolved tax per trade:
GOLD base −0.29 · GOLD NYJUDAS −0.32 · EUR base −0.28 · EUR NYJUDAS −0.23
GBP base −0.41 · GBP NYTREND −0.23 · JPY base −0.08 · LONBREAK −0.12…−0.35

## 2026 YTD leaderboard (naive → 5m-calibrated)
| Book | naive | calibrated |
|---|---|---|
| GOLD NY-Judas fade | +55.0 | **+12.0** |
| USDJPY Asia-NY (yours) | +19.9 (@my cost) | **+10.0** |
| GOLD Asia-NY (yours) | +35.5 | +1.3 |
| EURUSD NY-Judas | +23.5 | −7.3 |
| GBPUSD NY-Trend | +22.8 | −2.1 |
| London-break cont. (round-2 family, best cfgs) | +28…+40 | −22.7…+1.2 |
| EURUSD Asia-NY (yours, @your zero-cost) | +37.4 | ≈ +3.0 |
| GBPUSD Asia-NY (yours, @your zero-cost) | +30.6 | ≈ −23.8 |

Honest portfolio estimate: your 4-book table (+142.13R claimed) restates to
≈ **+50R ±20** under realistic costs + intrabar truth. Adding GOLD NY-Judas
lifts the honest total to ≈ **+62–70R**.

## Why most strategies die under fine resolution
Tight targets (≤1×ATR) relative to hourly ranges make win-rates depend on
intrabar path luck. Structural defenses that actually worked:
- rejection-confirmed / next-open entries (no level-fill ambiguity),
- wider risk anchors (JPY book's small tax),
- sessions with clean post-signal trends (gold NY afternoon).
Structural designs that STILL failed 5m validation: London-break continuation,
Asia-range breakout (fill artifact), daily sweep-rejection (no edge at all).

## Recommendations
1. Trade only what survived: GOLD NY-Judas as overlay; keep USDJPY Asia-NY;
   treat other FX books as unproven until live fills prove otherwise.
2. Run demo fills vs the 5m simulation for 4 weeks to calibrate your broker's
   real tax rate before sizing up.
3. Re-price all backtests with your actual all-in costs; demand ≥1.5 PF after
   costs AND after the measured tax.

## Key files
audit.py (17 tests) · explore.py · engine.py (path-aware capable) ·
fine15.py / fine5m.py / validate_pa.py (ground-truth studies) ·
search.py→search4.py (hunts v1–v4) · stress.py · detail26.py ·
results/detailed/* (trade-by-trade CSVs, scenarios, bootstrap)
