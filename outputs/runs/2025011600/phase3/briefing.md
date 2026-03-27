# Phase 3 Briefing - Run 2025011600

- Init Time: `2025-01-16T00:00:00`
- Horizon Hours: `90`
- Risk Level: `LOW`
- Forecast Stability: `HIGHLY_UNSTABLE`
- Peak Timing Agreement: `STRONG`

## Operational Takeaway (Plain Summary)
- Risk for tomorrow is manageable.
- Operators should continue routine monitoring and scheduled checks.
- Forecasts are still shifting across recent updates.
- Weather showed a 50% indicative association with forecast changes this run (R² 0.50, 19 samples — treat as directional only, not statistically definitive).

## Executive Summary
- Operational risk level is LOW. Forecast stability is HIGHLY_UNSTABLE, and peak timing agreement is STRONG.
- Forecast peak is 283.1 MW at 2025-01-16T13:00:00.
- Capacity is 329.0 MW with 0 forecast hour(s) above capacity.
- Forecast variability across updates is 4.9 MW median spread.
- Weather showed a 50% indicative association with forecast changes this run (R² 0.50, 19 samples — treat as directional only, not statistically definitive).

## Peak and Capacity
- Peak load is expected at `2025-01-16T13:00:00` with `283.1 MW` (capacity exceedance `0.0 MW`, exceeds capacity: `False`).
- Capacity is `329.0 MW`; forecast is above capacity for `0` hour(s); maximum exceedance is `0.0 MW`.
This suggests elevated load during key hours and possible strain on system capacity.

## Forecast Stability
- Forecast variability across updates is 4.9 MW (median spread across updates).
- Agreement across recent forecast updates on peak timing is `100%` (spread `0h`).
- How much forecasts changed between recent updates reached `15.5 MW` at `2025-01-16T11:00:00`.
Even if peak timing agrees, overall load levels can still shift between updates.

## Weather Impact
- Weather showed a `50%` indicative association with forecast changes this run (R² `0.50`, `19` samples — treat as directional only, not statistically definitive).
- Top linked weather variable is `Td2m` (correlation `-0.69`).
This explains forecast revisions, not causality.

## Watchlist Hours
### Capacity Watchlist (Near/Above Capacity)
| Time | Expected Load (MW) | Exceedance (MW) | Reason |
|---|---:|---:|---|
| 2025-01-16 13:00:00 | 280.5 | 0.0 | High load watch (85%) |
| 2025-01-16 14:00:00 | 273.5 | 0.0 | High load watch (83%) |
| 2025-01-16 12:00:00 | 268.7 | 0.0 | High load watch (82%) |
| 2025-01-16 15:00:00 | 256.2 | 0.0 | High load watch (78%) |
| 2025-01-16 02:00:00 | 248.6 | 0.0 | High load watch (76%) |
| 2025-01-16 01:00:00 | 248.5 | 0.0 | High load watch (76%) |

### Stability Watchlist (Hours Most Likely To Shift)
| Time | Expected Load (MW) | Volatility (MW) | Range (MW) | Reason |
|---|---:|---:|---:|---|
| 2025-01-16 13:00:00 | 280.5 MW | 7.0 | 15.2 | Forecast shifting across updates |
| 2025-01-16 12:00:00 | 268.7 MW | 6.8 | 15.5 | Forecast shifting across updates |
| 2025-01-16 11:00:00 | 242.5 MW | 6.6 | 15.5 | Forecast shifting across updates |
| 2025-01-16 10:00:00 | 224.0 MW | 5.6 | 12.7 | Forecast shifting across updates |
| 2025-01-16 14:00:00 | 273.5 MW | 5.3 | 11.6 | Forecast shifting across updates |
| 2025-01-16 09:00:00 | 213.2 MW | 4.8 | 10.6 | Forecast shifting across updates |

## Ramp Risk & Energy at Risk
- No significant ramp risk detected (max ramp up: `0.0 MW/h`, threshold: `N/A MW/h`).
- No energy above capacity in forecast horizon (all hours within capacity).

## Backtest Quality
- Recent model accuracy: Not available.

## Recommended Actions
- `ACT-002` (P1): Monitor Unstable Hours - Focus operator review on top unstable hours and rerun checks at next init.
- `ACT-003` (P1): Wait for Next Update - Defer irreversible actions until next forecast cycle if operationally possible.

## Signal Meanings
- Risk Level: `LOW`
- Forecast Stability: `HIGHLY_UNSTABLE`
- Peak Timing Agreement: `STRONG`
- Legacy Confidence Grade (secondary): `C`
- Grade C means moderate confidence. Continue monitoring updates before major commitment.

## Notes
- No actual load beyond X 00:00 used in evaluation metrics.
