from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.llm.client import llm_generate_text
from src.phase4.generator import RUNS_ROOT, get_or_build_phase4


def _extract_numeric_tokens(text: str) -> set[str]:
    toks = re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)", text)
    return {t.lstrip("+") for t in toks}


def _contains_new_numbers(output: str, facts: dict[str, Any]) -> bool:
    allowed = _extract_numeric_tokens(json.dumps(facts, ensure_ascii=False))
    out = _extract_numeric_tokens(output)
    return not out.issubset(allowed)


def classify_question_intent(question: str) -> str:
    q = (question or "").lower()
    if ("why" in q and "risk" in q) or "why severe" in q:
        return "risk_reason"
    if "why unstable" in q or "stable" in q or "stability" in q:
        return "stability_reason"
    if "weather" in q:
        return "weather_reason"
    if "which hour" in q or "which hours" in q or "hours" in q:
        return "hours_watchlist"
    return "general"


def _fmt_num(value: Any) -> str:
    return "NA" if value is None else str(value)


def _deterministic_answer(intent: str, facts: dict[str, Any]) -> tuple[str, list[str]]:
    peak = facts.get("peak", {})
    cap = facts.get("capacity", {})
    st = facts.get("stability", {})
    w = facts.get("weather", {})

    if intent == "risk_reason":
        text = (
            f"Risk is {facts.get('risk_level')} because the peak is {_fmt_num(peak.get('value_mw'))} MW "
            f"against capacity {_fmt_num(peak.get('capacity_mw'))} MW, with {cap.get('hours_above_capacity')} "
            f"forecast hour(s) above capacity and max exceedance {_fmt_num(peak.get('max_exceedance_mw'))} MW."
        )
        return text, [
            "risk_level",
            "peak.value_mw",
            "peak.capacity_mw",
            "capacity.hours_above_capacity",
            "peak.max_exceedance_mw",
        ]

    if intent == "stability_reason":
        text = (
            f"Forecast stability is {facts.get('forecast_stability_level')} with disagreement index "
            f"{_fmt_num(st.get('disagreement_index'))} MW, average revision volatility "
            f"{_fmt_num(st.get('avg_revision_volatility'))} MW, and max range {_fmt_num(st.get('max_range_mw'))} MW."
        )
        return text, [
            "forecast_stability_level",
            "stability.disagreement_index",
            "stability.avg_revision_volatility",
            "stability.max_range_mw",
        ]

    if intent == "weather_reason":
        if w.get("attribution_r2") is None:
            return (
                "Weather impact could not be quantified for this run.",
                ["weather.attribution_r2"],
            )
        text = (
            f"Weather attribution_r2 is {w.get('attribution_r2')}. "
            f"Top variable is {w.get('top_variable')} with correlation {w.get('correlation')}. "
            "This explains revisions, not causality."
        )
        return text, ["weather.attribution_r2", "weather.top_variable", "weather.correlation"]

    if intent == "hours_watchlist":
        cap_rows = facts.get("capacity_watchlist_hours", [])[:3]
        if cap_rows:
            parts = []
            for r in cap_rows:
                parts.append(f"{r.get('time')} ({_fmt_num(r.get('expected_load_mw'))} MW)")
            return (
                "Most capacity-critical hours are: " + ", ".join(parts) + ".",
                ["capacity_watchlist_hours"],
            )
        return (
            "That information is not available in the current forecast data.",
            ["capacity_watchlist_hours"],
        )

    return (
        f"Run {facts.get('run_id')} is {facts.get('risk_level')} risk, forecast stability is "
        f"{facts.get('forecast_stability_level')}, and peak timing agreement is {facts.get('peak_timing_agreement')}.",
        ["risk_level", "forecast_stability_level", "peak_timing_agreement"],
    )


def _build_llm_prompt(question: str, facts: dict[str, Any], draft: str, sources: list[str]) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Use only this JSON:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"Draft answer (fact-checked):\n{draft}\n\n"
        f"Source fields:\n{', '.join(sources)}\n\n"
        "Rewrite the draft answer in clear plain English, keep exact numbers and values unchanged."
    )


def answer_forecast_question(run_id: str, question: str) -> dict[str, Any]:
    phase4 = get_or_build_phase4(run_id, force=False, use_llm_summary=True)
    facts = phase4["facts_pack"]
    intent = classify_question_intent(question)
    draft, sources = _deterministic_answer(intent, facts)

    if draft == "That information is not available in the current forecast data.":
        return {
            "answer": draft,
            "sources": sources,
            "intent": intent,
            "mode": "deterministic",
            "status": "not_available",
        }

    prompt = _build_llm_prompt(question, facts, draft, sources)
    system = (
        "You are an assistant explaining a load forecast. "
        "You are not allowed to invent numbers. "
        "You must only use values present in the provided JSON. "
        "If a question cannot be answered from the JSON, say: "
        "'That information is not available in the current forecast data.'"
    )

    llm_text = None
    llm_err = None
    for _ in range(2):
        try:
            candidate = llm_generate_text(
                prompt=prompt,
                system=system,
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=350,
            ).strip()
            if _contains_new_numbers(candidate, facts):
                llm_text = None
                continue
            llm_text = candidate
            break
        except Exception as exc:
            llm_err = str(exc)
            llm_text = None
            break

    if llm_text is None:
        if llm_err:
            return {
                "answer": draft,
                "sources": sources,
                "intent": intent,
                "mode": "deterministic_fallback",
                "status": "llm_error",
                "error": llm_err,
            }
        return {
            "answer": "I cannot answer reliably based on available data.",
            "sources": sources,
            "intent": intent,
            "mode": "safe_reject",
            "status": "invalid_numbers",
        }

    return {
        "answer": llm_text,
        "sources": sources,
        "intent": intent,
        "mode": "llm",
        "status": "ok",
    }


def load_facts_pack(run_id: str) -> dict[str, Any]:
    p = RUNS_ROOT / run_id / "phase4" / "facts_pack.json"
    if not p.exists():
        return get_or_build_phase4(run_id, force=False, use_llm_summary=True)["facts_pack"]
    return json.loads(p.read_text(encoding="utf-8"))
