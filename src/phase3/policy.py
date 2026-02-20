from __future__ import annotations

from typing import Any


# Configurable thresholds
RISK_SEVERE_EXCEED_MAJ = 3
RISK_SEVERE_MAX_EX_MW = 25
RISK_SEVERE_HRS_ABOVE = 8
RISK_HIGH_EXCEED_MAJ = 1
RISK_HIGH_MAX_EX_MW = 10
RISK_HIGH_HRS_ABOVE = 4
RISK_MEDIUM_MAX_EX_MIN = 3
RISK_LOW_MAX_EX_LT = 3

STAB_HIGHLY_REV_VOL = 5
STAB_HIGHLY_MAX_RANGE = 20
STAB_HIGHLY_DISAGREE = 6
STAB_UNSTABLE_REV_VOL = 2
STAB_UNSTABLE_MAX_RANGE = 10
STAB_UNSTABLE_DISAGREE = 4
STAB_MODERATE_REV_VOL = 1
STAB_MODERATE_MAX_RANGE = 5
STAB_MODERATE_DISAGREE = 2

PEAK_STRONG_CONF = 0.8
PEAK_STRONG_SPREAD = 1
PEAK_OK_CONF = 0.6
PEAK_OK_SPREAD = 3

DISAGREEMENT_P50 = 1.8
DISAGREEMENT_P75 = 2.6
DISAGREEMENT_P90 = 3.8
T2M_DISAGREEMENT_HIGH = 0.9
ATTRIBUTION_R2_HIGH = 0.25
REV_VOL_HIGH = 2.0
MAX_RANGE_HIGH = 8.0
VOI_MAJOR_IMPROVEMENT_MW = 2.0


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
    if (1 <= h <= 3) or (RISK_MEDIUM_MAX_EX_MIN <= ex < RISK_HIGH_MAX_EX_MW):
        return "MEDIUM"
    if h == 0 and maj == 0 and ex < RISK_LOW_MAX_EX_LT:
        return "LOW"
    return "MEDIUM"


def classify_forecast_stability(
    disagreement_index: float | None,
    avg_revision_volatility: float | None,
    max_range_mw: float | None,
) -> str:
    d = float(disagreement_index or 0)
    v = float(avg_revision_volatility or 0)
    r = float(max_range_mw or 0)

    if v >= STAB_HIGHLY_REV_VOL or r >= STAB_HIGHLY_MAX_RANGE or d >= STAB_HIGHLY_DISAGREE:
        return "HIGHLY_UNSTABLE"
    if v >= STAB_UNSTABLE_REV_VOL or r >= STAB_UNSTABLE_MAX_RANGE or d >= STAB_UNSTABLE_DISAGREE:
        return "UNSTABLE"
    if v >= STAB_MODERATE_REV_VOL or r >= STAB_MODERATE_MAX_RANGE or d >= STAB_MODERATE_DISAGREE:
        return "MODERATE"
    return "STABLE"


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

    if peak_conf >= 0.70 and disagreement <= DISAGREEMENT_P50 and spread <= 2:
        grade = "A"
    elif peak_conf >= 0.55 and disagreement <= DISAGREEMENT_P75 and spread <= 4:
        grade = "B"
    elif peak_conf >= 0.40 or disagreement <= DISAGREEMENT_P90:
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
                    "reason": "Near/above capacity",
                    "exceed_prob_proxy": _as_float(r.get("exceed_prob_proxy")),
                }
            )
        if out:
            return out[:max_items]

    top_load = p1.get("top_load_hours", []) or []
    if top_load:
        rows = sorted(top_load, key=lambda r: _as_float(r.get("predicted_load_mw")) or 0.0, reverse=True)
        near = [r for r in rows if (_as_float(r.get("predicted_load_mw")) or 0.0) >= capacity * 0.98]
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
                    "reason": "Near/above capacity",
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
                "reason": "Near/above capacity",
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

    # 3-6 actions max
    return actions[:6]
