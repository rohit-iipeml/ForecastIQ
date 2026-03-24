from __future__ import annotations

from typing import Any

from src.constants import (
    ATTRIBUTION_R2_HIGH,
    BACKTEST_MODERATE_PCT,
    BACKTEST_POOR_PCT,
    CAPACITY_AT_RATIO,
    CAPACITY_NEAR_RATIO,
    DISAGREEMENT_P50,
    DISAGREEMENT_P75,
    DISAGREEMENT_P90,
    GRADE_A_PEAK_CONF_MIN,
    GRADE_A_SPREAD_MAX,
    GRADE_B_PEAK_CONF_MIN,
    GRADE_B_SPREAD_MAX,
    GRADE_C_PEAK_CONF_MIN,
    MAX_RANGE_HIGH,
    PEAK_OK_CONF,
    PEAK_OK_SPREAD,
    PEAK_STRONG_CONF,
    PEAK_STRONG_SPREAD,
    RAMP_HIGH_PCT,
    REV_VOL_HIGH,
    RISK_HIGH_EXCEED_MAJ,
    RISK_HIGH_HRS_ABOVE,
    RISK_HIGH_MAX_EX_MW,
    RISK_LOW_MAX_EX_LT,
    RISK_MEDIUM_MAX_EX_MIN,
    RISK_MEDIUM_HRS_MAX,
    RISK_MEDIUM_HRS_MIN,
    RISK_SEVERE_EXCEED_MAJ,
    RISK_SEVERE_HRS_ABOVE,
    RISK_SEVERE_MAX_EX_MW,
    STAB_PCT,
    T2M_DISAGREEMENT_HIGH,
    VOI_MAJOR_IMPROVEMENT_MW,
)


GRADE_ORDER = ["A", "B", "C", "D"]


def _downgrade(grade: str, steps: int = 1) -> str:
    idx = GRADE_ORDER.index(grade)
    return GRADE_ORDER[min(len(GRADE_ORDER) - 1, idx + steps)]


def classify_risk_level(
    hours_above_capacity: float | int | None,
    max_exceedance_mw: float | int | None,
    exceedance_hours_majority: float | int | None,
) -> str:
    h = float(hours_above_capacity or 0)
    ex = float(max_exceedance_mw or 0)
    maj = float(exceedance_hours_majority or 0)

    if maj >= RISK_SEVERE_EXCEED_MAJ or ex >= RISK_SEVERE_MAX_EX_MW or h >= RISK_SEVERE_HRS_ABOVE:
        return "SEVERE"
    if maj >= RISK_HIGH_EXCEED_MAJ or ex >= RISK_HIGH_MAX_EX_MW or h >= RISK_HIGH_HRS_ABOVE:
        return "HIGH"
    if (RISK_MEDIUM_HRS_MIN <= h <= RISK_MEDIUM_HRS_MAX) or (RISK_MEDIUM_MAX_EX_MIN <= ex < RISK_HIGH_MAX_EX_MW):
        return "MEDIUM"
    if h == 0 and maj == 0 and ex < RISK_LOW_MAX_EX_LT:
        return "LOW"
    return "MEDIUM"


def classify_forecast_stability(
    disagreement_index: float | None,
    revision_volatility: float | None,
    max_range: float | None,
    median_load_mw: float | None,
) -> str:
    if median_load_mw is None or float(median_load_mw) <= 0:
        return "UNKNOWN"
    d = float(disagreement_index or 0)
    v = float(revision_volatility or 0)
    r = float(max_range or 0)
    m = float(median_load_mw)

    # Percentage thresholds — tune these if grid behavior changes
    # All expressed as fraction of median load (e.g. 0.012 = 1.2%)
    PCT_MODERATE_DISAGREE = STAB_PCT["moderate"]["disagree"]
    PCT_UNSTABLE_DISAGREE = STAB_PCT["unstable"]["disagree"]
    PCT_HIGHLY_DISAGREE = STAB_PCT["highly"]["disagree"]

    PCT_MODERATE_VOL = STAB_PCT["moderate"]["vol"]
    PCT_UNSTABLE_VOL = STAB_PCT["unstable"]["vol"]
    PCT_HIGHLY_VOL = STAB_PCT["highly"]["vol"]

    PCT_MODERATE_RANGE = STAB_PCT["moderate"]["range"]
    PCT_UNSTABLE_RANGE = STAB_PCT["unstable"]["range"]
    PCT_HIGHLY_RANGE = STAB_PCT["highly"]["range"]

    # Derived thresholds — all in same MW units as inputs
    d_moderate = m * PCT_MODERATE_DISAGREE
    d_unstable = m * PCT_UNSTABLE_DISAGREE
    d_highly = m * PCT_HIGHLY_DISAGREE

    v_moderate = m * PCT_MODERATE_VOL
    v_unstable = m * PCT_UNSTABLE_VOL
    v_highly = m * PCT_HIGHLY_VOL

    r_moderate = m * PCT_MODERATE_RANGE
    r_unstable = m * PCT_UNSTABLE_RANGE
    r_highly = m * PCT_HIGHLY_RANGE

    # Classification — same logic structure as before
    if d >= d_highly or v >= v_highly or r >= r_highly:
        return "HIGHLY_UNSTABLE"
    if d >= d_unstable or v >= v_unstable or r >= r_unstable:
        return "UNSTABLE"
    if d >= d_moderate or v >= v_moderate or r >= r_moderate:
        return "MODERATE"
    return "STABLE"


def _capacity_reason(expected_mw: float, capacity_mw: float) -> str:
    """
    Return a precise reason label based on actual load vs capacity ratio.
    All thresholds are derived from capacity_mw — no hardcoded MW values.
    """
    if capacity_mw <= 0:
        return "Capacity unknown"
    ratio = expected_mw / capacity_mw
    exceedance = expected_mw - capacity_mw

    if exceedance > 0:
        return f"OVER CAPACITY +{exceedance:.1f} MW ({ratio*100:.0f}%)"
    if ratio >= CAPACITY_AT_RATIO:
        return f"At capacity ({ratio*100:.0f}%)"
    if ratio >= CAPACITY_NEAR_RATIO:
        return f"Near capacity ({ratio*100:.0f}%)"
    return f"High load watch ({ratio*100:.0f}%)"


def classify_peak_timing_agreement(
    peak_confidence: float | None,
    peak_time_spread_hours: float | int | None,
) -> str:
    if peak_confidence is None:
        return "UNKNOWN"
    pc = float(peak_confidence)
    spread = float(peak_time_spread_hours or 0)
    if pc >= PEAK_STRONG_CONF and spread <= PEAK_STRONG_SPREAD:
        return "STRONG"
    if pc >= PEAK_OK_CONF and spread <= PEAK_OK_SPREAD:
        return "OK"
    return "WEAK"


def compute_confidence_grade(phase3_input: dict[str, Any]) -> tuple[str, list[str]]:
    p2 = phase3_input.get("phase2", {})
    p25 = phase3_input.get("phase25", {})

    disagreement = p2.get("disagreement_index")
    peak_conf = p2.get("peak_confidence")
    spread = p2.get("peak_time_spread_hours")
    ex_majority = p2.get("exceedance_hours_majority")

    t2m_dis = (
        p25.get("weather_disagreement", {})
        .get("T2m", {})
        .get("disagreement_index")
    )
    r2 = p25.get("attribution", {}).get("r2")

    reasons: list[str] = []
    if peak_conf is None or disagreement is None or spread is None:
        return "D", ["Missing key stability metrics for confidence grading."]

    if peak_conf >= GRADE_A_PEAK_CONF_MIN and disagreement <= DISAGREEMENT_P50 and spread <= GRADE_A_SPREAD_MAX:
        grade = "A"
    elif peak_conf >= GRADE_B_PEAK_CONF_MIN and disagreement <= DISAGREEMENT_P75 and spread <= GRADE_B_SPREAD_MAX:
        grade = "B"
    elif peak_conf >= GRADE_C_PEAK_CONF_MIN or disagreement <= DISAGREEMENT_P90:
        grade = "C"
    else:
        grade = "D"
    reasons.append(
        f"Base grade from peak_confidence={peak_conf:.2f}, disagreement_index={disagreement:.2f}, "
        f"peak_time_spread_hours={spread}."
    )

    if t2m_dis is not None and r2 is not None and t2m_dis >= T2M_DISAGREEMENT_HIGH and r2 >= ATTRIBUTION_R2_HIGH:
        grade = _downgrade(grade, 1)
        reasons.append(
            f"Downgraded due to weather-driven volatility: T2m disagreement={t2m_dis:.2f}, attribution_r2={r2:.2f}."
        )

    if ex_majority is not None and ex_majority > 0 and disagreement > DISAGREEMENT_P75:
        grade = _downgrade(grade, 1)
        reasons.append(
            f"Downgraded due to combined capacity pressure and disagreement: "
            f"exceedance_hours_majority={ex_majority}, disagreement_index={disagreement:.2f}."
        )

    return grade, reasons


def _voi_has_major_improvement(phase3_input: dict[str, Any]) -> bool:
    bins = phase3_input.get("phase2", {}).get("voi_summary", {}).get("bins", [])
    mae_0_24 = None
    mae_48_72 = None
    for b in bins:
        if b.get("age_bin") == "0-24":
            mae_0_24 = b.get("mae")
        if b.get("age_bin") == "48-72":
            mae_48_72 = b.get("mae")
    if mae_0_24 is None or mae_48_72 is None:
        return False
    return (mae_48_72 - mae_0_24) >= VOI_MAJOR_IMPROVEMENT_MW


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def compute_ramp_risk(forecast_hours: list[dict[str, Any]], avg_load_mw: float | None = None) -> dict[str, Any]:
    """
    Identify rapid load ramps (up or down) across the 90-hour horizon.
    Threshold: RAMP_HIGH_PCT of avg_load_mw per hour.
    Falls back to a fixed absolute threshold (50 MW/h) if avg_load_mw is unavailable.
    """
    if not forecast_hours or len(forecast_hours) < 2:
        return {
            "ramp_risk_flag": False,
            "max_ramp_up_mw": None,
            "max_ramp_up_time": None,
            "max_ramp_down_mw": None,
            "max_ramp_down_time": None,
            "ramp_threshold_mw": None,
            "note": "Insufficient forecast points for ramp analysis.",
        }

    try:
        loads = [(h.get("timestamp"), float(h.get("predicted_load", 0.0))) for h in forecast_hours]
    except Exception:
        return {"ramp_risk_flag": False, "max_ramp_up_mw": None, "max_ramp_up_time": None,
                "max_ramp_down_mw": None, "max_ramp_down_time": None, "ramp_threshold_mw": None,
                "note": "Could not parse forecast hours for ramp analysis."}

    if avg_load_mw and float(avg_load_mw) > 0:
        threshold_mw = float(avg_load_mw) * RAMP_HIGH_PCT
    else:
        threshold_mw = 50.0

    max_up = 0.0
    max_up_time = None
    max_down = 0.0
    max_down_time = None

    for i in range(1, len(loads)):
        delta = loads[i][1] - loads[i - 1][1]
        if delta > max_up:
            max_up = delta
            max_up_time = loads[i][0]
        if (-delta) > max_down:
            max_down = -delta
            max_down_time = loads[i][0]

    ramp_flag = max_up >= threshold_mw or max_down >= threshold_mw
    return {
        "ramp_risk_flag": ramp_flag,
        "max_ramp_up_mw": round(max_up, 1),
        "max_ramp_up_time": max_up_time,
        "max_ramp_down_mw": round(max_down, 1),
        "max_ramp_down_time": max_down_time,
        "ramp_threshold_mw": round(threshold_mw, 1),
        "note": f"Threshold: {threshold_mw:.1f} MW/h ({RAMP_HIGH_PCT*100:.0f}% of avg load).",
    }


def compute_energy_at_risk(forecast_hours: list[dict[str, Any]], capacity_mw: float) -> dict[str, Any]:
    """
    Compute total MWh above capacity (energy-at-risk) across the forecast horizon.
    Each hourly exceedance contributes 1 hour × exceedance_mw MWh.
    """
    if not forecast_hours or capacity_mw <= 0:
        return {"energy_at_risk_mwh": 0.0, "hours_above_capacity": 0, "note": "No forecast data or invalid capacity."}

    total_mwh = 0.0
    hours_above = 0
    for h in forecast_hours:
        try:
            load = float(h.get("predicted_load", 0.0))
        except Exception:
            continue
        if load > capacity_mw:
            total_mwh += load - capacity_mw
            hours_above += 1

    return {
        "energy_at_risk_mwh": round(total_mwh, 1),
        "hours_above_capacity": hours_above,
        "note": f"Total energy above {capacity_mw:.1f} MW capacity threshold.",
    }


def classify_backtest_quality(mae_mw: float | None, avg_load_mw: float | None) -> dict[str, Any]:
    """
    Classify recent model accuracy based on MAE as a percentage of average load.
    poor: MAE > BACKTEST_POOR_PCT of avg load
    moderate: MAE > BACKTEST_MODERATE_PCT of avg load
    good: MAE <= BACKTEST_MODERATE_PCT
    """
    if mae_mw is None or avg_load_mw is None or float(avg_load_mw) <= 0:
        return {
            "backtest_quality_flag": "unknown",
            "mae_pct": None,
            "note": "Backtest MAE or average load not available.",
        }

    mae = float(mae_mw)
    avg = float(avg_load_mw)
    mae_pct = mae / avg

    if mae_pct > BACKTEST_POOR_PCT:
        flag = "poor"
        note = f"MAE is {mae_pct*100:.1f}% of avg load — above {BACKTEST_POOR_PCT*100:.0f}% threshold."
    elif mae_pct > BACKTEST_MODERATE_PCT:
        flag = "moderate"
        note = f"MAE is {mae_pct*100:.1f}% of avg load — moderate accuracy."
    else:
        flag = "good"
        note = f"MAE is {mae_pct*100:.1f}% of avg load — within acceptable range."

    return {
        "backtest_quality_flag": flag,
        "mae_pct": round(mae_pct, 4),
        "note": note,
    }


def build_capacity_watchlist_hours(phase3_input: dict[str, Any], max_items: int = 6) -> list[dict[str, Any]]:
    p2 = phase3_input.get("phase2", {})
    p1 = phase3_input.get("phase1", {})
    capacity = _as_float(p1.get("capacity_mw")) or 0.0
    out: list[dict[str, Any]] = []

    top_risky = p2.get("top_risky_hours", []) or []
    if top_risky:
        rows = sorted(
            top_risky,
            key=lambda r: (
                _as_float(r.get("exceed_prob_proxy")) or 0.0,
                _as_float(r.get("median_load_mw")) or 0.0,
            ),
            reverse=True,
        )
        for r in rows[:max_items]:
            expected = _as_float(r.get("median_load_mw"))
            if expected is None:
                continue
            out.append(
                {
                    "time": r.get("time"),
                    "expected_load_mw": expected,
                    "exceedance_mw": max(0.0, expected - capacity),
                    "reason": _capacity_reason(expected, capacity),
                    "exceed_prob_proxy": _as_float(r.get("exceed_prob_proxy")),
                }
            )
        if out:
            return out[:max_items]

    top_load = p1.get("top_load_hours", []) or []
    if top_load:
        rows = sorted(top_load, key=lambda r: _as_float(r.get("predicted_load_mw")) or 0.0, reverse=True)
        near = [r for r in rows if (_as_float(r.get("predicted_load_mw")) or 0.0) >= capacity * CAPACITY_NEAR_RATIO]
        rows = near + [r for r in rows if r not in near]
        for r in rows[:max_items]:
            expected = _as_float(r.get("predicted_load_mw"))
            if expected is None:
                continue
            out.append(
                {
                    "time": r.get("time"),
                    "expected_load_mw": expected,
                    "exceedance_mw": max(0.0, expected - capacity),
                    "reason": _capacity_reason(expected, capacity),
                    "exceed_prob_proxy": None,
                }
            )
        if out:
            return out[:max_items]

    peak = p1.get("peak", {})
    if peak.get("time"):
        expected = _as_float(peak.get("value_mw")) or 0.0
        out.append(
            {
                "time": peak.get("time"),
                "expected_load_mw": expected,
                "exceedance_mw": max(0.0, expected - capacity),
                "reason": _capacity_reason(expected, capacity),
                "exceed_prob_proxy": None,
            }
        )
    return out[:max_items]


def build_stability_watchlist_hours(phase3_input: dict[str, Any], max_items: int = 6) -> list[dict[str, Any]]:
    p2 = phase3_input.get("phase2", {})
    out: list[dict[str, Any]] = []

    unstable = p2.get("top_unstable_hours", []) or []
    if unstable:
        rows = sorted(
            unstable,
            key=lambda r: (
                _as_float(r.get("std_mw")) or -1.0,
                _as_float(r.get("range_mw")) or -1.0,
                _as_float(r.get("iqr_mw")) or -1.0,
            ),
            reverse=True,
        )
        for r in rows[:max_items]:
            out.append(
                {
                    "time": r.get("time"),
                    "median_load_mw": None,
                    "volatility_mw": _as_float(r.get("std_mw")),
                    "range_mw": _as_float(r.get("range_mw")),
                    "reason": "Forecast shifting across updates",
                }
            )
        if out:
            return out[:max_items]

    max_range_time = p2.get("max_range_time")
    if max_range_time:
        out.append(
            {
                "time": max_range_time,
                "median_load_mw": None,
                "volatility_mw": _as_float(p2.get("avg_revision_volatility")),
                "range_mw": _as_float(p2.get("max_range_mw")),
                "reason": "Forecast shifting across updates",
            }
        )
    return out[:max_items]


def build_watchlist_hours(phase3_input: dict[str, Any], max_items: int = 10) -> list[dict[str, Any]]:
    cap = build_capacity_watchlist_hours(phase3_input, max_items=max_items)
    stab = build_stability_watchlist_hours(phase3_input, max_items=max_items)
    combined = []
    for row in cap:
        combined.append(
            {
                "time": row.get("time"),
                "reason": row.get("reason"),
                "expected_load_mw": row.get("expected_load_mw"),
                "notes": f"exceedance_mw={row.get('exceedance_mw')}",
            }
        )
    for row in stab:
        combined.append(
            {
                "time": row.get("time"),
                "reason": row.get("reason"),
                "expected_load_mw": row.get("median_load_mw"),
                "notes": f"volatility_mw={row.get('volatility_mw')}, range_mw={row.get('range_mw')}",
            }
        )
    return combined[:max_items]


def build_action_items(
    phase3_input: dict[str, Any],
    confidence_grade: str,
    metadata_warnings: list[str],
) -> list[dict[str, Any]]:
    p1 = phase3_input.get("phase1", {})
    p2 = phase3_input.get("phase2", {})
    p25 = phase3_input.get("phase25", {})
    actions: list[dict[str, Any]] = []

    if (p2.get("exceedance_hours_majority", 0) or 0) > 0 or (p1.get("hours_above_capacity", 0) or 0) > 0:
        actions.append(
            {
                "id": "ACT-001",
                "title": "Capacity Alert Readiness",
                "priority": "P0",
                "confidence": 0.9,
                "triggered_by": [
                    {
                        "metric": "exceedance_hours_majority",
                        "value": p2.get("exceedance_hours_majority"),
                        "threshold": "> 0",
                        "direction": "above",
                    }
                ],
                "recommended_next_step": "Prepare reserve and monitor top risk hours for dispatch decisions.",
            }
        )

    if (p2.get("avg_revision_volatility", 0) or 0) >= REV_VOL_HIGH or (p2.get("max_range_mw", 0) or 0) >= MAX_RANGE_HIGH:
        actions.append(
            {
                "id": "ACT-002",
                "title": "Monitor Unstable Hours",
                "priority": "P1",
                "confidence": 0.8,
                "triggered_by": [
                    {
                        "metric": "avg_revision_volatility",
                        "value": p2.get("avg_revision_volatility"),
                        "threshold": f">= {REV_VOL_HIGH}",
                        "direction": "above",
                    }
                ],
                "recommended_next_step": "Focus operator review on top unstable hours and rerun checks at next init.",
            }
        )

    if confidence_grade in {"C", "D"} and _voi_has_major_improvement(phase3_input):
        actions.append(
            {
                "id": "ACT-003",
                "title": "Wait for Next Update",
                "priority": "P1",
                "confidence": 0.75,
                "triggered_by": [
                    {
                        "metric": "VOI_MAE_improvement",
                        "value": "0-24h better than 48-72h",
                        "threshold": f">= {VOI_MAJOR_IMPROVEMENT_MW} MW",
                        "direction": "improves_with_fresh_forecast",
                    }
                ],
                "recommended_next_step": "Defer irreversible actions until next forecast cycle if operationally possible.",
            }
        )

    t2m_dis = p25.get("weather_disagreement", {}).get("T2m", {}).get("disagreement_index")
    r2 = p25.get("attribution", {}).get("r2")
    if t2m_dis is not None and r2 is not None and t2m_dis >= T2M_DISAGREEMENT_HIGH and r2 >= ATTRIBUTION_R2_HIGH:
        actions.append(
            {
                "id": "ACT-004",
                "title": "Weather-driven Volatility Watch",
                "priority": "P2",
                "confidence": 0.7,
                "triggered_by": [
                    {
                        "metric": "attribution_r2",
                        "value": r2,
                        "threshold": f">= {ATTRIBUTION_R2_HIGH}",
                        "direction": "above",
                    }
                ],
                "recommended_next_step": "Track weather model updates closely; reassess load risk after next weather refresh.",
            }
        )

    if metadata_warnings:
        actions.append(
            {
                "id": "ACT-005",
                "title": "Data Quality Review",
                "priority": "P2",
                "confidence": 0.6,
                "triggered_by": [
                    {
                        "metric": "metadata_warnings_count",
                        "value": len(metadata_warnings),
                        "threshold": "> 0",
                        "direction": "above",
                    }
                ],
                "recommended_next_step": "Review missing/partial artifacts before final operational sign-off.",
            }
        )

    ramp_risk = (p1.get("ramp_risk") or {})
    if ramp_risk.get("ramp_risk_flag"):
        max_up = ramp_risk.get("max_ramp_up_mw")
        max_down = ramp_risk.get("max_ramp_down_mw")
        threshold = ramp_risk.get("ramp_threshold_mw")
        actions.append(
            {
                "id": "ACT-006",
                "title": "High Ramp Rate Detected",
                "priority": "P1",
                "confidence": 0.8,
                "triggered_by": [
                    {
                        "metric": "max_ramp_up_mw",
                        "value": max_up,
                        "threshold": f">= {threshold} MW/h",
                        "direction": "above",
                    }
                ],
                "recommended_next_step": (
                    f"Forecast shows ramp up to {max_up:.1f} MW/h and down to {max_down:.1f} MW/h. "
                    "Ensure fast-ramping generation is available during peak ramp windows."
                ),
            }
        )

    backtest = (p1.get("backtest_quality") or {})
    if backtest.get("backtest_quality_flag") == "poor":
        mae_pct = backtest.get("mae_pct")
        actions.append(
            {
                "id": "ACT-007",
                "title": "Elevated Recent Forecast Error",
                "priority": "P1",
                "confidence": 0.75,
                "triggered_by": [
                    {
                        "metric": "backtest_mae_pct",
                        "value": f"{float(mae_pct or 0)*100:.1f}%",
                        "threshold": f"> {BACKTEST_POOR_PCT*100:.0f}% of avg load",
                        "direction": "above",
                    }
                ],
                "recommended_next_step": (
                    f"Yesterday's MAE was {float(mae_pct or 0)*100:.1f}% of average load. "
                    "Apply wider operational margins and verify input data quality."
                ),
            }
        )

    # max 7 actions
    return actions[:7]
