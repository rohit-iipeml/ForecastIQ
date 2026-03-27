# Phase 3 Briefing - Run 2025012500

- Init Time: `2025-01-25T00:00:00`
- Horizon Hours: `90`
- Risk Level: `SEVERE`
- Forecast Stability: `MODERATE`
- Peak Timing Agreement: `STRONG`

## Operational Takeaway (Plain Summary)
- Risk for tomorrow is elevated.
- Operators should prepare now for tight hours.
- Forecasts are fairly stable.
- Weather showed a 24% indicative association with forecast changes this run (R² 0.24, 19 samples — treat as directional only, not statistically definitive).

## Executive Summary
- Operational risk level is SEVERE. Forecast stability is MODERATE, and peak timing agreement is STRONG.
- Forecast peak is 354.9 MW at 2025-01-25T13:00:00.
- Capacity is 329.0 MW with 3 forecast hour(s) above capacity.
- Forecast variability across updates is 2.1 MW median spread.
- Weather showed a 24% indicative association with forecast changes this run (R² 0.24, 19 samples — treat as directional only, not statistically definitive).

## Peak and Capacity
- Peak load is expected at `2025-01-25T13:00:00` with `354.9 MW` (capacity exceedance `25.9 MW`, exceeds capacity: `True`).
- Capacity is `329.0 MW`; forecast is above capacity for `3` hour(s); maximum exceedance is `25.9 MW`.
This suggests elevated load during key hours and possible strain on system capacity.

## Forecast Stability
- Forecast variability across updates is 2.1 MW (median spread across updates).
- Agreement across recent forecast updates on peak timing is `100%` (spread `0h`).
- How much forecasts changed between recent updates reached `7.7 MW` at `2025-01-25T01:00:00`.
Even if peak timing agrees, overall load levels can still shift between updates.

## Weather Impact
- Weather showed a `24%` indicative association with forecast changes this run (R² `0.24`, `19` samples — treat as directional only, not statistically definitive).
- Top linked weather variable is `T2m` (correlation `-0.48`).
This explains forecast revisions, not causality.

## Watchlist Hours
### Capacity Watchlist (Near/Above Capacity)
| Time | Expected Load (MW) | Exceedance (MW) | Reason |
|---|---:|---:|---|
| 2025-01-25 13:00:00 | 344.9 | 15.9 | OVER CAPACITY +15.9 MW (105%) |
| 2025-01-25 14:00:00 | 343.8 | 14.8 | OVER CAPACITY +14.8 MW (105%) |
| 2025-01-25 12:00:00 | 331.3 | 2.3 | OVER CAPACITY +2.3 MW (101%) |
| 2025-01-25 15:00:00 | 320.4 | 0.0 | High load watch (97%) |
| 2025-01-25 11:00:00 | 312.7 | 0.0 | High load watch (95%) |
| 2025-01-25 10:00:00 | 296.1 | 0.0 | High load watch (90%) |

### Stability Watchlist (Hours Most Likely To Shift)
| Time | Expected Load (MW) | Volatility (MW) | Range (MW) | Reason |
|---|---:|---:|---:|---|
| 2025-01-25 23:00:00 | 232.1 MW | 3.4 | 6.8 | Forecast shifting across updates |
| 2025-01-25 00:00:00 | 272.2 MW | 3.2 | 7.3 | Forecast shifting across updates |
| 2025-01-25 01:00:00 | 283.9 MW | 3.2 | 7.7 | Forecast shifting across updates |
| 2025-01-25 04:00:00 | 283.7 MW | 2.8 | 6.6 | Forecast shifting across updates |
| 2025-01-25 08:00:00 | 274.8 MW | 2.7 | 6.5 | Forecast shifting across updates |
| 2025-01-25 16:00:00 | 293.2 MW | 2.5 | 6.0 | Forecast shifting across updates |

## Ramp Risk & Energy at Risk
- No significant ramp risk detected (max ramp up: `0.0 MW/h`, threshold: `N/A MW/h`).
- No energy above capacity in forecast horizon (all hours within capacity).

## Backtest Quality
- Recent model accuracy: Not available.

## Recommended Actions
- `ACT-001` (P0): Capacity Alert Readiness - Prepare reserve and monitor top risk hours for dispatch decisions.

## Signal Meanings
- Risk Level: `SEVERE`
- Forecast Stability: `MODERATE`
- Peak Timing Agreement: `STRONG`
- Legacy Confidence Grade (secondary): `B`
- Grade B means moderate confidence. Continue monitoring updates before major commitment.

## Notes
- No actual load beyond X 00:00 used in evaluation metrics.
