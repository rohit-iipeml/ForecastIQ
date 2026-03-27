## Situation Summary
The risk tier is CRITICAL. The forecast peak load is 364.1 MW at 2025-01-23T00:00:00. The capacity margin is roughly 26 MW below capacity for about 15 hours. The forecast stability is HIGHLY_UNSTABLE. The forecast is still shifting. The ramp_risk_flag is true, with a ramp window around 2025-01-24T12:00:00. Total energy above capacity is 224.0 MWh.

## What To Watch
**13:00** — 342.7 MW expected, 13.7 MW above capacity (104%) — pre-position reserves
**14:00** — 338.3 MW expected, 9.3 MW above capacity (103%) — pre-position reserves
**12:00** — 330.4 MW expected, 1.4 MW above capacity (100%) — pre-position reserves
**03:00** — 311.0 MW expected, ±23.7 MW spread — forecast still shifting, defer firm commitments
**04:00** — 296.0 MW expected, ±21.4 MW spread — forecast still shifting, defer firm commitments

## Forecast Confidence
The model confidence grade is D, meaning forecasts are unreliable. Do not make firm commitments without the latest update. Weather impact could not be quantified for this run.

## Ramp Risk & Energy at Risk
System load is expected to ramp up 25.1 MW/h around 2025-01-24T12:00:00 — pre-position fast-ramping generation before that window. Maximum downward ramp is 28.5 MW/h — ensure regulating reserve covers the drop. Total energy above capacity across the forecast horizon is 224.0 MWh — reserve availability must cover this exposure.

## Backtest Quality
The backtest quality is POOR. Yesterday's model MAE was 16.5% of average load — apply at least a 16.5% buffer above peak forecast values.

## Recommended Actions
**ACT-001 P0** — Prepare reserve and monitor top risk hours for dispatch decisions.
**ACT-002 P1** — Focus operator review on top unstable hours and rerun checks at next init.
**ACT-003 P1** — Defer irreversible actions until next forecast cycle if operationally possible.
**ACT-005 P2** — Review missing/partial artifacts before final operational sign-off.
**ACT-006 P1** — Forecast shows ramp up to 25.1 MW/h and down to 28.5 MW/h. Ensure fast-ramping generation is available during peak ramp windows.
**ACT-007 P1** — Yesterday's MAE was 16.5% of average load. Apply wider operational margins and verify input data quality.