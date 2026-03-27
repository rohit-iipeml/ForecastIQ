from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.client import llm_generate_text


RUNS_ROOT = Path("outputs/runs")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_numeric_tokens(text: str) -> set[str]:
    toks = re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)", text)
    return {t.lstrip("+") for t in toks}


def _has_new_numbers(output_text: str, allowed_source_text: str) -> bool:
    return not _extract_numeric_tokens(output_text).issubset(_extract_numeric_tokens(allowed_source_text))


def build_facts_pack(run_id: str) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    briefing = _read_json(run_dir / "phase3" / "briefing.json")
    phase3_input = _read_json(run_dir / "phase3" / "phase3_input.json")

    p2 = phase3_input.get("phase2", {})
    weather = briefing.get("weather_load_link", {})

    p1 = phase3_input.get("phase1", {})

    # Human-readable recommended actions for chatbot grounding
    raw_actions = briefing.get("recommended_actions", []) or []
    readable_actions = [
        {
            "id": a.get("id", ""),
            "priority": a.get("priority", ""),
            "title": a.get("title", ""),
            "recommended_next_step": a.get("recommended_next_step", ""),
            "rationale": a.get("rationale", ""),
        }
        for a in raw_actions
    ]

    facts = {
        "run_id": run_id,
        "init_time": phase3_input.get("init_time"),
        "horizon_hours": phase3_input.get("horizon_hours", 90),
        "risk_level": briefing.get("risk_level"),
        "forecast_stability_level": briefing.get("forecast_stability_level"),
        "stability_label": briefing.get("forecast_stability_level", "").replace("_", " ").title(),
        "peak_timing_agreement": briefing.get("peak_timing_agreement"),
        "confidence_grade": briefing.get("confidence_grade"),
        "confidence_reasons": briefing.get("confidence_reasons", []),
        "peak_mw": briefing.get("peak", {}).get("value_mw"),
        "peak_time": briefing.get("peak", {}).get("time"),
        "peak": {
            "time": briefing.get("peak", {}).get("time"),
            "value_mw": briefing.get("peak", {}).get("value_mw"),
            "capacity_mw": briefing.get("capacity", {}).get("capacity_mw"),
            "max_exceedance_mw": briefing.get("capacity", {}).get("max_exceedance_mw"),
        },
        "capacity_mw": briefing.get("capacity", {}).get("capacity_mw"),
        "exceedance_hours": briefing.get("capacity", {}).get("hours_above_capacity"),
        "max_exceedance_mw": briefing.get("capacity", {}).get("max_exceedance_mw"),
        "capacity": {
            "hours_above_capacity": briefing.get("capacity", {}).get("hours_above_capacity"),
            "exceedance_hours_majority": p2.get("exceedance_hours_majority"),
        },
        "avg_load_mw": p1.get("avg_predicted_load_mw") or p2.get("median_load_mw"),
        "stability": {
            "disagreement_index": briefing.get("stability", {}).get("disagreement_index"),
            "avg_revision_volatility": p2.get("avg_revision_volatility"),
            "max_range_mw": briefing.get("stability", {}).get("max_range_mw"),
            "peak_confidence": briefing.get("stability", {}).get("peak_confidence"),
            "peak_time_spread_hours": briefing.get("stability", {}).get("peak_time_spread_hours"),
        },
        "attribution_r2": weather.get("attribution_r2"),
        "weather": {
            "attribution_r2": weather.get("attribution_r2"),
            "top_variable": weather.get("top_driver_var"),
            "correlation": weather.get("top_driver_corr"),
            "r2_reliable": weather.get("r2_reliable"),
            "n_samples": weather.get("n_samples"),
        },
        "capacity_watchlist_hours": briefing.get("capacity_watchlist_hours", []) or [],
        "stability_watchlist_hours": briefing.get("stability_watchlist_hours", []) or [],
        "recommended_actions": readable_actions,
        "ramp_risk": briefing.get("ramp_risk") or {},
        "energy_at_risk_mwh": briefing.get("energy_at_risk_mwh", 0.0),
        "backtest_quality": briefing.get("backtest_quality") or {},
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "phase3_briefing_json": str(run_dir / "phase3" / "briefing.json"),
                "phase3_input_json": str(run_dir / "phase3" / "phase3_input.json"),
            },
        },
    }
    return facts


def _fmt_mw(value: Any) -> str:
    return "NA" if value is None else f"{value} MW"


def _deterministic_simple_summary(facts: dict[str, Any]) -> str:
    r2 = facts.get("weather", {}).get("attribution_r2")
    def _fmt_mw_r(v: Any) -> str:
        try:
            return f"{float(v):.1f} MW"
        except Exception:
            return "NA"

    def _fmt_peak_time(v: Any) -> str:
        import pandas as pd
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return str(v) if v else "NA"
        return ts.strftime("%b %d %H:%M")

    overall_lines = [
        f"Run {facts.get('run_id')} is classified as {facts.get('risk_level')} risk with {facts.get('forecast_stability_level')} forecast stability.",
        f"Peak load is {_fmt_mw_r(facts.get('peak', {}).get('value_mw'))} at {_fmt_peak_time(facts.get('peak', {}).get('time'))} against capacity {_fmt_mw_r(facts.get('peak', {}).get('capacity_mw'))}.",
        f"Forecast is above capacity for {facts.get('capacity', {}).get('hours_above_capacity')} hour(s), with max exceedance {_fmt_mw_r(facts.get('peak', {}).get('max_exceedance_mw'))}.",
    ]
    if r2 is None:
        overall_lines.append("Weather impact could not be quantified for this run.")
    else:
        overall_lines.append(f"Weather attribution R² is {float(r2):.2f} for this run.")

    ramp = facts.get("ramp_risk") or {}
    if ramp.get("ramp_risk_flag"):
        overall_lines.append(
            f"High ramp rate detected: up to {ramp.get('max_ramp_up_mw', 0):.1f} MW/h rise, "
            f"{ramp.get('max_ramp_down_mw', 0):.1f} MW/h drop."
        )
    ear_mwh = float(facts.get("energy_at_risk_mwh") or 0)
    if ear_mwh > 0:
        overall_lines.append(f"Energy-at-risk above capacity: {ear_mwh:.1f} MWh.")
    bt = facts.get("backtest_quality") or {}
    if bt.get("backtest_quality_flag") == "poor":
        overall_lines.append(f"Recent model accuracy is POOR ({bt.get('note', '')})")

    meaning = [
        f"- Tomorrow has {facts.get('risk_level')} operational risk, so reserve and dispatch decisions should track watchlist hours closely.",
        f"- Forecast stability is {facts.get('forecast_stability_level')}, so updates may still shift load levels before delivery.",
        "- Operators should stay cautious around capacity-critical hours and revisit decisions at the next forecast update.",
    ]

    def _r(v: Any, digits: int = 1) -> str:
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return str(v) if v is not None else "—"

    def _fmt_time(v: Any) -> str:
        import pandas as pd
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return str(v) if v else "—"
        return ts.strftime("%b %d %H:%M")

    cap_top = facts.get("capacity_watchlist_hours", [])[:3]
    stab_top = facts.get("stability_watchlist_hours", [])[:3]

    lines: list[str] = []
    lines.append(f"# Simple Summary — Run {facts.get('run_id')}")
    lines.append("")
    lines.append("## 1. Overall Situation")
    lines.extend([f"{x}" for x in overall_lines])
    lines.append("")
    lines.append("## 2. What This Means")
    lines.extend(meaning[:3])
    lines.append("")
    lines.append("## 3. Top Things To Watch")
    lines.append("Capacity watchlist (top 3):")
    for row in cap_top:
        lines.append(
            f"- {_fmt_time(row.get('time'))}: expected {_r(row.get('expected_load_mw'))} MW,"
            f" exceedance {_r(row.get('exceedance_mw'))} MW"
        )
    if not cap_top:
        lines.append("- No capacity watchlist hours available.")
    lines.append("Stability watchlist (top 3):")
    for row in stab_top:
        expected = row.get("expected_load_display") or (f"{_r(row.get('expected_load_mw'))} MW" if row.get("expected_load_mw") is not None else "—")
        lines.append(
            f"- {_fmt_time(row.get('time'))}: expected {expected},"
            f" volatility {_r(row.get('volatility_mw'))} MW, range {_r(row.get('range_mw'))} MW"
        )
    if not stab_top:
        lines.append("- No stability watchlist hours available.")
    return "\n".join(lines).strip() + "\n"


def _build_simple_summary_prompt(facts: dict[str, Any]) -> str:
    return (
        "Write markdown for sections exactly:\n"
        "## 1. Overall Situation\n"
        "## 2. What This Means\n\n"
        "Rules:\n"
        "- Plain English, short sentences.\n"
        "- Section 1 must be 3-4 sentences max.\n"
        "- Section 2 must be max 3 bullets.\n"
        "- Do not introduce any new numbers.\n"
        "- Copy numeric values exactly from JSON.\n"
        "- If weather.attribution_r2 is null, include exactly: 'Weather impact could not be quantified for this run.'\n"
        "Return markdown only for these two sections, no extra headings.\n\n"
        f"facts_pack.json:\n{json.dumps(facts, ensure_ascii=False)}\n"
    )


def generate_simple_summary(run_id: str, facts: dict[str, Any], use_llm: bool = True) -> tuple[str, dict[str, Any]]:
    deterministic = _deterministic_simple_summary(facts)
    lines = deterministic.splitlines()
    top_watch_start = lines.index("## 3. Top Things To Watch")
    deterministic_top_watch = "\n".join(lines[top_watch_start:]).strip()

    llm_status = "deterministic"
    llm_reason = "disabled"
    summary_head = "\n".join(lines[:top_watch_start]).strip()

    if use_llm:
        prompt = _build_simple_summary_prompt(facts)
        allowed = json.dumps(facts, ensure_ascii=False)
        try:
            llm_text = llm_generate_text(
                prompt=prompt,
                system=(
                    "You are a power grid shift briefing assistant.\n"
                    "Write for both operators and managers — clear, confident, human.\n"
                    "Use short sentences. Lead with what matters most (risk level, peak load, key concern).\n"
                    "Never invent numbers. Use only the facts provided. Round naturally in prose.\n"
                    "Output clean markdown with no code fences."
                ),
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=800,
            ).strip()
            if _has_new_numbers(llm_text, allowed):
                # one retry
                llm_text = llm_generate_text(
                    prompt=prompt,
                    system=(
                        "Rewrite the forecast summary in plain, confident English for grid operators and managers.\n"
                        "Use only the numbers already present. Do not add any new figures."
                    ),
                    model="llama-3.3-70b-versatile",
                    temperature=0.0,
                    max_tokens=800,
                ).strip()
            if _has_new_numbers(llm_text, allowed):
                llm_status = "fallback"
                llm_reason = "new_numbers"
            else:
                llm_status = "ok"
                llm_reason = None
                summary_head = llm_text
        except Exception as exc:
            llm_status = "fallback"
            llm_reason = f"llm_error: {exc}"

    full_summary = f"{summary_head}\n\n{deterministic_top_watch}\n"
    meta = {
        "status": llm_status,
        "reason": llm_reason,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return full_summary, meta


def get_or_build_phase4(run_id: str, force: bool = False, use_llm_summary: bool = True) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    phase4_dir = run_dir / "phase4"
    phase4_dir.mkdir(parents=True, exist_ok=True)

    facts_path = phase4_dir / "facts_pack.json"
    summary_path = phase4_dir / "simple_summary.md"
    summary_meta_path = phase4_dir / "simple_summary.meta.json"

    if not force and facts_path.exists() and summary_path.exists():
        return {
            "run_id": run_id,
            "cached": True,
            "facts_pack": _read_json(facts_path),
            "files": {
                "facts_pack_json": str(facts_path),
                "simple_summary_md": str(summary_path),
                "simple_summary_meta_json": str(summary_meta_path) if summary_meta_path.exists() else None,
            },
        }

    facts = build_facts_pack(run_id)
    summary_md, summary_meta = generate_simple_summary(run_id, facts, use_llm=use_llm_summary)

    _write_json(facts_path, facts)
    summary_path.write_text(summary_md, encoding="utf-8")
    _write_json(summary_meta_path, summary_meta)

    return {
        "run_id": run_id,
        "cached": False,
        "facts_pack": facts,
        "files": {
            "facts_pack_json": str(facts_path),
            "simple_summary_md": str(summary_path),
            "simple_summary_meta_json": str(summary_meta_path),
        },
    }
