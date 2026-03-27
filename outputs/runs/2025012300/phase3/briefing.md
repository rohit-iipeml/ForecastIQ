# Phase 3 Briefing - Run 2025012300

- Init Time: `2025-01-23T00:00:00`
- Horizon Hours: `90`
- Risk Level: `SEVERE`
- Forecast Stability: `HIGHLY_UNSTABLE`
- Peak Timing Agreement: `STRONG`

## Operational Takeaway (Plain Summary)
- Risk for tomorrow is elevated.
- Operators should prepare now for tight hours.
- Forecasts are still shifting across recent updates.
- Weather impact could not be quantified for this run (insufficient overlap/update data).

## Executive Summary
- Operational risk level is SEVERE. Forecast stability is HIGHLY_UNSTABLE, and peak timing agreement is STRONG.
- Forecast peak is 364.1 MW at 2025-01-23T00:00:00.
- Capacity is 329.0 MW with 15 forecast hour(s) above capacity.
- Forecast variability across updates is 7.1 MW median spread.
- Weather impact could not be quantified for this run (insufficient overlap/update data).
- Significant ramp detected: up to 25.1 MW/h rise and 28.5 MW/h drop within the forecast window.

## Peak and Capacity
- Peak load is expected at `2025-01-23T00:00:00` with `364.1 MW` (capacity exceedance `35.1 MW`, exceeds capacity: `True`).
- Capacity is `329.0 MW`; forecast is above capacity for `15` hour(s); maximum exceedance is `35.1 MW`.
This suggests elevated load during key hours and possible strain on system capacity.

## Forecast Stability
- Forecast variability across updates is 7.1 MW (median spread across updates).
- Agreement across recent forecast updates on peak timing is `100%` (spread `0h`).
- How much forecasts changed between recent updates reached `23.7 MW` at `2025-01-23T03:00:00`.
Even if peak timing agrees, overall load levels can still shift between updates.

## Weather Impact
- Weather impact could not be quantified for this run (insufficient overlap/update data).
- No weather-driver claim is made without attribution evidence.

## Watchlist Hours
### Capacity Watchlist (Near/Above Capacity)
| Time | Expected Load (MW) | Exceedance (MW) | Reason |
|---|---:|---:|---|
| 2025-01-23 13:00:00 | 342.7 | 13.7 | OVER CAPACITY +13.7 MW (104%) |
| 2025-01-23 14:00:00 | 338.3 | 9.3 | OVER CAPACITY +9.3 MW (103%) |
| 2025-01-23 12:00:00 | 330.4 | 1.4 | OVER CAPACITY +1.4 MW (100%) |
| 2025-01-23 15:00:00 | 322.1 | 0.0 | High load watch (98%) |
| 2025-01-23 02:00:00 | 318.1 | 0.0 | High load watch (97%) |
| 2025-01-23 01:00:00 | 314.7 | 0.0 | High load watch (96%) |

### Stability Watchlist (Hours Most Likely To Shift)
| Time | Expected Load (MW) | Volatility (MW) | Range (MW) | Reason |
|---|---:|---:|---:|---|
| 2025-01-23 03:00:00 | 311.0 MW | 9.8 | 23.7 | Forecast shifting across updates |
| 2025-01-23 04:00:00 | 296.0 MW | 9.0 | 21.4 | Forecast shifting across updates |
| 2025-01-23 08:00:00 | 274.1 MW | 8.8 | 19.0 | Forecast shifting across updates |
| 2025-01-23 05:00:00 | 286.1 MW | 8.4 | 18.8 | Forecast shifting across updates |
| 2025-01-23 09:00:00 | 277.3 MW | 8.4 | 19.4 | Forecast shifting across updates |
| 2025-01-23 10:00:00 | 288.6 MW | 7.8 | 19.0 | Forecast shifting across updates |

## Ramp Risk & Energy at Risk
- **Ramp risk flagged**: max ramp up `25.1 MW/h` at `2025-01-24T12:00:00`, max ramp down `28.5 MW/h` at `2025-01-24T16:00:00` (threshold: `14.2 MW/h`).
- Total energy above capacity: `224.0 MWh` across 90-hour horizon.

## Backtest Quality
- **Recent model accuracy: POOR** — MAE is 16.5% of avg load — above 3% threshold.
- Yesterday MAE was `16.5%` of average load.

## Recommended Actions
- `ACT-001` (P0): Capacity Alert Readiness - Prepare reserve and monitor top risk hours for dispatch decisions.
- `ACT-002` (P1): Monitor Unstable Hours - Focus operator review on top unstable hours and rerun checks at next init.
- `ACT-003` (P1): Wait for Next Update - Defer irreversible actions until next forecast cycle if operationally possible.
- `ACT-005` (P2): Data Quality Review - Review missing/partial artifacts before final operational sign-off.
- `ACT-006` (P1): High Ramp Rate Detected - Forecast shows ramp up to 25.1 MW/h and down to 28.5 MW/h. Ensure fast-ramping generation is available during peak ramp windows.
- `ACT-007` (P1): Elevated Recent Forecast Error - Yesterday's MAE was 16.5% of average load. Apply wider operational margins and verify input data quality.

## Signal Meanings
- Risk Level: `SEVERE`
- Forecast Stability: `HIGHLY_UNSTABLE`
- Peak Timing Agreement: `STRONG`
- Legacy Confidence Grade (secondary): `D`
- Grade D means forecasts have been unstable or risk signals are elevated. Operators should plan conservatively.

## Notes
- No actual load beyond X 00:00 used in evaluation metrics.
