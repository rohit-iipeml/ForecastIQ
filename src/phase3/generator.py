from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.phase3.contract import build_phase3_input, load_phase3_input
from src.phase3.llm_writer import generate_briefing_llm_md
from src.phase3.policy import (
    build_action_items,
    build_capacity_watchlist_hours,
    build_stability_watchlist_hours,
    compute_confidence_grade,
    classify_forecast_stability,
    classify_peak_timing_agreement,
    classify_risk_level,
)
from src.phase3.schemas import validate_action_items_json, validate_briefing_json


RUNS_ROOT = Path("outputs/runs")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _parse_ts(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts


def _nearest_value(target_ts: pd.Timestamp, entries: list[tuple[pd.Timestamp, float]]) -> float | None:
    if not entries:
        return None
    exact = [val for ts, val in entries if ts == target_ts]
    if exact:
        return exact[0]
    nearest: tuple[pd.Timedelta, float] | None = None
    for ts, val in entries:
        delta = abs(ts - target_ts)
        if nearest is None or delta < nearest[0]:
            nearest = (delta, val)
    if nearest is None or nearest[0] > pd.Timedelta(hours=1):
        return None
    return nearest[1]


def _load_consensus_entries(run_id: str) -> list[tuple[pd.Timestamp, float]]:
    path = RUNS_ROOT / run_id / "phase2" / "consensus_series.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    ts_col = next((c for c in ["target_timestamp", "timestamp", "ForeTime", "time"] if c in df.columns), None)
    val_col = next((c for c in ["median", "consensus_median"] if c in df.columns), None)
    if ts_col is None or val_col is None:
        return []
    out: list[tuple[pd.Timestamp, float]] = []
    for _, row in df.iterrows():
        ts = _parse_ts(row.get(ts_col))
        if ts is None:
            continue
        try:
            val = float(row.get(val_col))
        except Exception:
            continue
        out.append((ts, val))
    return out


def _load_forecast_entries(run_id: str) -> list[tuple[pd.Timestamp, float]]:
    path = RUNS_ROOT / run_id / "forecast.csv"
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    year = int(run_id[:4])
    out: list[tuple[pd.Timestamp, float]] = []
    if "ForeTime" in df.columns and "Load" in df.columns:
        for _, row in df.iterrows():
            raw = str(row.get("ForeTime", "")).strip()
            ts = _parse_ts(f"{year}/{raw}")
            if ts is None:
                ts = _parse_ts(raw)
            if ts is None:
                continue
            try:
                val = float(row.get("Load"))
            except Exception:
                continue
            out.append((ts, val))
        return out
    ts_col = next((c for c in ["timestamp", "target_timestamp", "ForeTime", "time"] if c in df.columns), None)
    val_col = next((c for c in ["predicted_load", "Load"] if c in df.columns), None)
    if ts_col is None or val_col is None:
        return []
    for _, row in df.iterrows():
        ts = _parse_ts(row.get(ts_col))
        if ts is None:
            continue
        try:
            val = float(row.get(val_col))
        except Exception:
            continue
        out.append((ts, val))
    return out


def fill_expected_load_for_stability_watchlist(
    run_id: str,
    stability_watchlist_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    consensus_entries = _load_consensus_entries(run_id)
    forecast_entries = _load_forecast_entries(run_id)
    out: list[dict[str, Any]] = []
    for row in stability_watchlist_rows:
        new_row = dict(row)
        ts = _parse_ts(new_row.get("time"))
        expected: float | None = None
        if ts is not None:
            expected = _nearest_value(ts, consensus_entries)
            if expected is None:
                expected = _nearest_value(ts, forecast_entries)
        new_row["expected_load_mw"] = expected
        new_row["expected_load_display"] = "—" if expected is None else f"{expected:.1f} MW"
        if new_row.get("median_load_mw") is None and expected is not None:
            new_row["median_load_mw"] = expected
        out.append(new_row)
    return out


def _build_executive_summary(
    phase3_input: dict[str, Any],
    risk_level: str,
    stability_level: str,
    peak_timing_agreement: str,
) -> list[str]:
    p1 = phase3_input["phase1"]
    p2 = phase3_input["phase2"]
    p25 = phase3_input["phase25"]
    bullets = [
        f"Operational risk level is {risk_level}. Forecast stability is {stability_level}, and peak timing agreement is {peak_timing_agreement}.",
        f"Forecast peak is {float(p1['peak']['value_mw']):.1f} MW at {p1['peak']['time']}.",
        f"Capacity is {float(p1['capacity_mw']):.1f} MW with {p1['hours_above_capacity']} forecast hour(s) above capacity.",
        f"Forecast variability across updates is {float(p2.get('disagreement_index') or 0):.1f} MW median spread.",
    ]
    r2 = p25.get("attribution", {}).get("r2")
    if r2 is None:
        bullets.append("Weather impact could not be quantified for this run (insufficient overlap/update data).")
    else:
        bullets.append(
            f"Weather explains about {float(r2) * 100:.0f}% of recent forecast changes (explains revisions, not causality)."
        )
    return bullets[:6]


def _build_briefing_json(phase3_input: dict[str, Any]) -> dict[str, Any]:
    grade, reasons = compute_confidence_grade(phase3_input)
    p1 = phase3_input["phase1"]
    p2 = phase3_input["phase2"]
    p25 = phase3_input["phase25"]
    metadata_warnings = phase3_input.get("metadata", {}).get("warnings", []) or []
    risk_level = classify_risk_level(
        p1.get("hours_above_capacity"),
        p1.get("max_exceedance_mw"),
        p2.get("exceedance_hours_majority"),
    )
    stability_level = classify_forecast_stability(
        p2.get("disagreement_index"),
        p2.get("avg_revision_volatility"),
        p2.get("max_range_mw"),
    )
    peak_timing_agreement = classify_peak_timing_agreement(
        p2.get("peak_confidence"),
        p2.get("peak_time_spread_hours"),
    )

    capacity_watchlist = build_capacity_watchlist_hours(phase3_input, max_items=6)
    stability_watchlist = build_stability_watchlist_hours(phase3_input, max_items=6)
    stability_watchlist = fill_expected_load_for_stability_watchlist(phase3_input["run_id"], stability_watchlist)
    actions = build_action_items(phase3_input, grade, metadata_warnings)

    peak_ex = max(0.0, float(p1["peak"]["value_mw"]) - float(p1["capacity_mw"]))
    r2 = p25.get("attribution", {}).get("r2")
    top_driver_corr = p25.get("attribution", {}).get("top_driver_corr")
    weather_display = "Not available" if r2 is None else f"{float(r2) * 100:.0f}%"
    rec_actions = [
        {
            "id": a["id"],
            "title": a["title"],
            "priority": a["priority"],
            "rationale": "; ".join(
                [
                    f"{t['metric']}={t['value']} ({t['direction']} {t['threshold']})"
                    for t in a.get("triggered_by", [])
                ]
            ),
        }
        for a in actions
    ]

    briefing = {
        "run_id": phase3_input["run_id"],
        "risk_level": risk_level,
        "forecast_stability_level": stability_level,
        "peak_timing_agreement": peak_timing_agreement,
        "confidence_grade": grade,
        "confidence_reasons": reasons,
        "executive_summary": _build_executive_summary(
            phase3_input,
            risk_level,
            stability_level,
            peak_timing_agreement,
        ),
        "peak": {
            "time": p1["peak"]["time"],
            "value_mw": p1["peak"]["value_mw"],
            "exceeds_capacity": bool(peak_ex > 0),
            "exceedance_mw": peak_ex,
        },
        "capacity": {
            "capacity_mw": p1["capacity_mw"],
            "hours_above_capacity": p1["hours_above_capacity"],
            "max_exceedance_mw": p1["max_exceedance_mw"],
        },
        "stability": {
            "disagreement_index": p2.get("disagreement_index"),
            "peak_confidence": p2.get("peak_confidence"),
            "peak_time_spread_hours": p2.get("peak_time_spread_hours"),
            "max_range_mw": p2.get("max_range_mw"),
            "max_range_time": p2.get("max_range_time"),
        },
        "weather_load_link": {
            "attribution_r2": r2,
            "top_driver_var": p25.get("attribution", {}).get("top_driver_var"),
            "top_driver_corr": top_driver_corr,
        },
        "capacity_watchlist_hours": capacity_watchlist,
        "stability_watchlist_hours": stability_watchlist,
        "watchlist_hours": capacity_watchlist + stability_watchlist,
        "display": {
            "risk_level_display": risk_level,
            "stability_level_display": stability_level,
            "peak_timing_display": peak_timing_agreement,
            "peak_value_display": f"{float(p1['peak']['value_mw']):.1f} MW",
            "capacity_display": f"{float(p1['capacity_mw']):.1f} MW",
            "max_exceedance_display": f"{float(p1['max_exceedance_mw']):.1f} MW",
            "disagreement_display": f"{float(p2.get('disagreement_index') or 0):.1f} MW",
            "peak_agreement_display": peak_timing_agreement,
            "weather_revision_explains_display": weather_display,
            "top_driver_corr_display": "Not available"
            if top_driver_corr is None
            else f"{float(top_driver_corr):.2f}",
        },
        "recommended_actions": rec_actions,
        "limits": {
            "real_time_cut_note": "No actual load beyond X 00:00 used in evaluation metrics."
        },
    }
    validate_briefing_json(briefing)
    return briefing


def _build_action_items_json(phase3_input: dict[str, Any], grade: str) -> dict[str, Any]:
    warnings = phase3_input.get("metadata", {}).get("warnings", []) or []
    items = build_action_items(phase3_input, grade, warnings)
    payload = {
        "run_id": phase3_input["run_id"],
        "items": items,
    }
    validate_action_items_json(payload)
    return payload


def _build_briefing_md(phase3_input: dict[str, Any], briefing: dict[str, Any], actions: dict[str, Any]) -> str:
    def _mw(x: Any) -> str:
        try:
            return f"{float(x):.1f}"
        except Exception:
            return "NA"

    def _num2(x: Any) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "NA"

    def _pct(x: Any) -> str:
        try:
            return f"{float(x) * 100:.0f}%"
        except Exception:
            return "NA"

    p1 = briefing["capacity"]
    peak = briefing["peak"]
    st = briefing["stability"]
    wl = briefing["weather_load_link"]
    risk_level = briefing.get("risk_level", "NA")
    stability_level = briefing.get("forecast_stability_level", "NA")
    peak_timing_agreement = briefing.get("peak_timing_agreement", "NA")
    cap_watch = briefing.get("capacity_watchlist_hours", []) or []
    stab_watch = briefing.get("stability_watchlist_hours", []) or []
    has_attr = wl.get("attribution_r2") is not None

    lines: list[str] = []
    lines.append(f"# Phase 3 Briefing - Run {phase3_input['run_id']}")
    lines.append("")
    lines.append(f"- Init Time: `{phase3_input['init_time']}`")
    lines.append(f"- Horizon Hours: `{phase3_input['horizon_hours']}`")
    lines.append(f"- Risk Level: `{risk_level}`")
    lines.append(f"- Forecast Stability: `{stability_level}`")
    lines.append(f"- Peak Timing Agreement: `{peak_timing_agreement}`")
    lines.append("")
    lines.append("## Operational Takeaway (Plain Summary)")
    high_risk = risk_level in {"HIGH", "SEVERE"}
    stable = stability_level in {"STABLE", "MODERATE"}
    lines.append(f"- Risk for tomorrow is {'elevated' if high_risk else 'manageable'}.")
    lines.append(
        f"- Operators should {'prepare now for tight hours' if high_risk else 'continue routine monitoring and scheduled checks'}."
    )
    lines.append(f"- Forecasts are {'fairly stable' if stable else 'still shifting across recent updates'}.")
    if has_attr:
        lines.append(f"- Weather explains about {_pct(wl.get('attribution_r2'))} of recent forecast changes.")
    else:
        lines.append("- Weather impact could not be quantified for this run (insufficient overlap/update data).")
    lines.append("")
    lines.append("## Executive Summary")
    lines.extend([f"- {b}" for b in briefing.get("executive_summary", [])[:6]])
    lines.append("")
    lines.append("## Peak and Capacity")
    lines.append(
        f"- Peak load is expected at `{peak['time']}` with `{_mw(peak['value_mw'])} MW` "
        f"(capacity exceedance `{_mw(peak['exceedance_mw'])} MW`, exceeds capacity: `{peak['exceeds_capacity']}`)."
    )
    lines.append(
        f"- Capacity is `{_mw(p1['capacity_mw'])} MW`; forecast is above capacity for "
        f"`{p1['hours_above_capacity']}` hour(s); maximum exceedance is `{_mw(p1['max_exceedance_mw'])} MW`."
    )
    lines.append("This suggests elevated load during key hours and possible strain on system capacity.")
    lines.append("")
    lines.append("## Forecast Stability")
    lines.append(
        f"- Forecast variability across updates is {_mw(st.get('disagreement_index'))} MW "
        f"(median spread across updates)."
    )
    lines.append(
        f"- Agreement across recent forecast updates on peak timing is `{_pct(st.get('peak_confidence'))}` "
        f"(spread `{st.get('peak_time_spread_hours')}h`)."
    )
    lines.append(
        f"- How much forecasts changed between recent updates reached `{_mw(st.get('max_range_mw'))} MW` "
        f"at `{st.get('max_range_time')}`."
    )
    lines.append("Even if peak timing agrees, overall load levels can still shift between updates.")
    lines.append("")
    lines.append("## Weather Impact")
    if has_attr:
        lines.append(
            f"- Weather explains about `{_pct(wl['attribution_r2'])}` of recent forecast changes "
            f"(R² `{_num2(wl['attribution_r2'])}`)."
        )
        lines.append(
            f"- Top linked weather variable is `{wl.get('top_driver_var')}` "
            f"(correlation `{_num2(wl.get('top_driver_corr'))}`)."
        )
        lines.append("This explains forecast revisions, not causality.")
    else:
        lines.append("- Weather impact could not be quantified for this run (insufficient overlap/update data).")
        lines.append("- No weather-driver claim is made without attribution evidence.")
    lines.append("")
    lines.append("## Watchlist Hours")
    lines.append("### Capacity Watchlist (Near/Above Capacity)")
    lines.append("| Time | Expected Load (MW) | Exceedance (MW) | Reason |")
    lines.append("|---|---:|---:|---|")
    for w in cap_watch[:10]:
        lines.append(
            f"| {w.get('time')} | {_mw(w.get('expected_load_mw'))} | {_mw(w.get('exceedance_mw'))} | {w.get('reason')} |"
        )
    if not cap_watch:
        lines.append("| NA | NA | NA | No capacity-critical hours available |")
    lines.append("")
    lines.append("### Stability Watchlist (Hours Most Likely To Shift)")
    lines.append("| Time | Expected Load (MW) | Volatility (MW) | Range (MW) | Reason |")
    lines.append("|---|---:|---:|---:|---|")
    for w in stab_watch[:10]:
        expected_display = w.get("expected_load_display") or ("—" if w.get("expected_load_mw") is None else f"{float(w.get('expected_load_mw')):.1f} MW")
        lines.append(
            f"| {w.get('time')} | {expected_display} | {_mw(w.get('volatility_mw'))} | {_mw(w.get('range_mw'))} | {w.get('reason')} |"
        )
    if not stab_watch:
        lines.append("| — | — | — | — | No stability hours available |")
    lines.append("")
    lines.append("## Recommended Actions")
    for a in actions["items"]:
        lines.append(f"- `{a['id']}` ({a['priority']}): {a['title']} - {a['recommended_next_step']}")
    lines.append("")
    lines.append("## Signal Meanings")
    lines.append(f"- Risk Level: `{risk_level}`")
    lines.append(f"- Forecast Stability: `{stability_level}`")
    lines.append(f"- Peak Timing Agreement: `{peak_timing_agreement}`")
    lines.append(f"- Legacy Confidence Grade (secondary): `{briefing['confidence_grade']}`")
    if briefing["confidence_grade"] == "A":
        lines.append("- Grade A means forecasts are stable across recent updates and risk signals are low.")
    elif briefing["confidence_grade"] == "D":
        lines.append("- Grade D means forecasts have been unstable or risk signals are elevated. Operators should plan conservatively.")
    else:
        lines.append(
            f"- Grade {briefing['confidence_grade']} means moderate confidence. Continue monitoring updates before major commitment."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- {briefing['limits']['real_time_cut_note']}")
    return "\n".join(lines).strip() + "\n"


def generate_phase3_outputs(
    run_id: str,
    force: bool = False,
    use_llm_writer: bool = False,
) -> dict[str, Any]:
    phase3_dir = RUNS_ROOT / run_id / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    p_input = phase3_dir / "phase3_input.json"
    p_briefing = phase3_dir / "briefing.json"
    p_actions = phase3_dir / "action_items.json"
    p_md = phase3_dir / "briefing.md"
    p_summary = phase3_dir / "phase3_summary.json"

    if not force and p_briefing.exists() and p_actions.exists() and p_input.exists() and p_md.exists():
        try:
            briefing = json.loads(p_briefing.read_text(encoding="utf-8"))
            actions = json.loads(p_actions.read_text(encoding="utf-8"))
            validate_briefing_json(briefing)
            validate_action_items_json(actions)
            out = {
                "run_id": run_id,
                "cached": True,
                "confidence_grade": briefing["confidence_grade"],
                "risk_level": briefing.get("risk_level"),
                "forecast_stability_level": briefing.get("forecast_stability_level"),
                "peak_timing_agreement": briefing.get("peak_timing_agreement"),
                "files": {
                    "phase3_input_json": str(p_input),
                    "briefing_json": str(p_briefing),
                    "action_items_json": str(p_actions),
                    "briefing_md": str(p_md),
                    "phase3_summary_json": str(p_summary) if p_summary.exists() else None,
                },
            }
            if use_llm_writer:
                llm = generate_briefing_llm_md(run_id, force=False)
                out["llm"] = llm
                out["files"]["briefing_llm_md"] = llm["path"]
            return out
        except Exception:
            # Backward compatibility: regenerate if cached schema is outdated.
            pass

    phase3_input = load_phase3_input(run_id) if (p_input.exists() and not force) else None
    if phase3_input is None:
        phase3_input = build_phase3_input(run_id)

    briefing = _build_briefing_json(phase3_input)
    actions = _build_action_items_json(phase3_input, briefing["confidence_grade"])
    md = _build_briefing_md(phase3_input, briefing, actions)

    _write_json(p_briefing, briefing)
    _write_json(p_actions, actions)
    p_md.write_text(md, encoding="utf-8")
    summary = {
        "run_id": run_id,
        "confidence_grade": briefing["confidence_grade"],
        "risk_level": briefing.get("risk_level"),
        "forecast_stability_level": briefing.get("forecast_stability_level"),
        "peak_timing_agreement": briefing.get("peak_timing_agreement"),
        "files": {
            "phase3_input_json": str(p_input),
            "briefing_json": str(p_briefing),
            "action_items_json": str(p_actions),
            "briefing_md": str(p_md),
        },
    }
    _write_json(p_summary, summary)
    summary["files"]["phase3_summary_json"] = str(p_summary)
    out = {
        "run_id": run_id,
        "cached": False,
        "confidence_grade": briefing["confidence_grade"],
        "risk_level": briefing.get("risk_level"),
        "forecast_stability_level": briefing.get("forecast_stability_level"),
        "peak_timing_agreement": briefing.get("peak_timing_agreement"),
        "files": summary["files"],
    }
    if use_llm_writer:
        llm = generate_briefing_llm_md(run_id, force=force)
        out["llm"] = llm
        out["files"]["briefing_llm_md"] = llm["path"]
    return out
