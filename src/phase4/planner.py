from __future__ import annotations

import json
import re
from typing import Any

from src.llm.client import llm_generate_text
from src.phase4.tools import ALLOWED_TOOLS


def _extract_weather_vars(question: str) -> list[str]:
    q = question.lower()
    vars = ["T2m"]
    if "humidity" in q or "rh" in q:
        vars.append("RH2m")
    if "dewpoint" in q or "dew point" in q or "td2m" in q:
        vars.append("Td2m")
    return vars


def _normalize_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for step in plan:
        tool = step.get("tool")
        args = step.get("args") or {}
        if tool not in ALLOWED_TOOLS:
            continue
        if not isinstance(args, dict):
            args = {}
        if tool == "tool_weather_at_times":
            args = {
                "timestamps_source_csv": str(args.get("timestamps_source_csv", "exceedance_hours.csv")),
                "vars": args.get("vars", ["T2m"]),
            }
        else:
            args = {}
        out.append({"tool": tool, "args": args})
    return out


def _deterministic_plan(question: str) -> list[dict[str, Any]]:
    q = question.lower()
    has_weather = any(k in q for k in ["temperature", "t2m", "weather", "humidity", "dewpoint", "dew point", "rh"])
    has_ex = any(k in q for k in ["above capacity", "over capacity", "exceedance"])
    if has_weather and has_ex:
        return [
            {"tool": "tool_exceedance_hours", "args": {}},
            {
                "tool": "tool_weather_at_times",
                "args": {
                    "timestamps_source_csv": "exceedance_hours.csv",
                    "vars": _extract_weather_vars(question),
                },
            },
        ]

    if any(k in q for k in ["most risky hours", "risky hours", "hours are most risky", "watchlist", "top hours"]):
        return [{"tool": "tool_top_risk_hours", "args": {}}]

    if any(k in q for k in ["unstable", "shifting", "revisions", "confidence", "agreement"]):
        return [{"tool": "tool_compare_revisions", "args": {}}]

    if any(k in q for k in ["why severe", "why high risk", "why risk"]):
        return [{"tool": "tool_top_risk_hours", "args": {}}]

    return []


def _llm_planner_fallback(question: str) -> list[dict[str, Any]]:
    tool_desc = {
        "tool_exceedance_hours": "Find timestamps where forecast load is at/above capacity.",
        "tool_weather_at_times": "Attach weather variables to provided timestamps source CSV.",
        "tool_top_risk_hours": "Return capacity and stability watchlist hours.",
        "tool_compare_revisions": "Return top unstable revision hours from phase2 metrics.",
    }
    prompt = (
        "Choose tools for this user question. Return JSON only in format "
        '{"plan":[{"tool":"...","args":{...}}]}. '
        "Do not include tools outside allowed list.\n"
        f"Allowed tools: {json.dumps(tool_desc)}\n"
        f"Question: {question}\n"
    )
    try:
        text = llm_generate_text(
            prompt=prompt,
            system="You are a tool planner. Output strict JSON only.",
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=300,
        ).strip()
    except Exception:
        return []

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return []
    try:
        payload = json.loads(m.group(0))
    except Exception:
        return []
    return _normalize_plan(payload.get("plan") or [])


def plan_tools(question: str, allow_llm_fallback: bool = True) -> list[dict[str, Any]]:
    plan = _deterministic_plan(question)
    if plan:
        return _normalize_plan(plan)
    if allow_llm_fallback:
        plan = _llm_planner_fallback(question)
        if plan:
            return plan
    return []
