from __future__ import annotations

from typing import Any


BRIEFING_SCHEMA: dict[str, Any] = {
    "required": {
        "run_id": str,
        "risk_level": str,
        "forecast_stability_level": str,
        "peak_timing_agreement": str,
        "confidence_grade": str,
        "confidence_reasons": list,
        "executive_summary": list,
        "peak": dict,
        "capacity": dict,
        "stability": dict,
        "weather_load_link": dict,
        "capacity_watchlist_hours": list,
        "stability_watchlist_hours": list,
        "recommended_actions": list,
        "limits": dict,
    },
    "enum": {
        "risk_level": {"LOW", "MEDIUM", "HIGH", "SEVERE"},
        "forecast_stability_level": {"STABLE", "MODERATE", "UNSTABLE", "HIGHLY_UNSTABLE"},
        "peak_timing_agreement": {"STRONG", "OK", "WEAK", "UNKNOWN"},
        "confidence_grade": {"A", "B", "C", "D"},
    },
}


ACTION_ITEMS_SCHEMA: dict[str, Any] = {
    "required": {
        "run_id": str,
        "items": list,
    },
}


def _require(payload: dict[str, Any], key: str, typ: type, where: str) -> None:
    if key not in payload:
        raise ValueError(f"Missing required field '{key}' in {where}.")
    if not isinstance(payload[key], typ):
        raise ValueError(f"Invalid type for '{key}' in {where}: expected {typ.__name__}.")


def validate_briefing_json(payload: dict[str, Any]) -> None:
    for key, typ in BRIEFING_SCHEMA["required"].items():
        _require(payload, key, typ, "briefing.json")
    if payload["risk_level"] not in BRIEFING_SCHEMA["enum"]["risk_level"]:
        raise ValueError("briefing.json: risk_level must be LOW/MEDIUM/HIGH/SEVERE")
    if payload["forecast_stability_level"] not in BRIEFING_SCHEMA["enum"]["forecast_stability_level"]:
        raise ValueError("briefing.json: forecast_stability_level must be STABLE/MODERATE/UNSTABLE/HIGHLY_UNSTABLE")
    if payload["peak_timing_agreement"] not in BRIEFING_SCHEMA["enum"]["peak_timing_agreement"]:
        raise ValueError("briefing.json: peak_timing_agreement must be STRONG/OK/WEAK/UNKNOWN")
    if payload["confidence_grade"] not in BRIEFING_SCHEMA["enum"]["confidence_grade"]:
        raise ValueError("briefing.json: confidence_grade must be one of A/B/C/D")

    peak = payload["peak"]
    for k in ["time", "value_mw", "exceeds_capacity", "exceedance_mw"]:
        if k not in peak:
            raise ValueError(f"briefing.json: missing peak.{k}")

    capacity = payload["capacity"]
    for k in ["capacity_mw", "hours_above_capacity", "max_exceedance_mw"]:
        if k not in capacity:
            raise ValueError(f"briefing.json: missing capacity.{k}")

    stability = payload["stability"]
    for k in [
        "disagreement_index",
        "peak_confidence",
        "peak_time_spread_hours",
        "max_range_mw",
        "max_range_time",
    ]:
        if k not in stability:
            raise ValueError(f"briefing.json: missing stability.{k}")

    wl = payload["weather_load_link"]
    for k in ["attribution_r2", "top_driver_var", "top_driver_corr"]:
        if k not in wl:
            raise ValueError(f"briefing.json: missing weather_load_link.{k}")

    for field in ["capacity_watchlist_hours", "stability_watchlist_hours"]:
        if not isinstance(payload.get(field), list):
            raise ValueError(f"briefing.json: {field} must be a list")

    limits = payload["limits"]
    if "real_time_cut_note" not in limits:
        raise ValueError("briefing.json: missing limits.real_time_cut_note")


def validate_action_items_json(payload: dict[str, Any]) -> None:
    for key, typ in ACTION_ITEMS_SCHEMA["required"].items():
        _require(payload, key, typ, "action_items.json")

    for i, item in enumerate(payload["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"action_items.json: items[{i}] must be object")
        for k in ["id", "title", "priority", "confidence", "triggered_by", "recommended_next_step"]:
            if k not in item:
                raise ValueError(f"action_items.json: missing items[{i}].{k}")
        if item["priority"] not in {"P0", "P1", "P2"}:
            raise ValueError(f"action_items.json: items[{i}].priority must be P0/P1/P2")
        c = item["confidence"]
        if not isinstance(c, (int, float)) or c < 0 or c > 1:
            raise ValueError(f"action_items.json: items[{i}].confidence must be in [0,1]")
