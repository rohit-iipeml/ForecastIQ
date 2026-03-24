from __future__ import annotations

import os
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from src.phase4.generator import RUNS_ROOT, get_or_build_phase4
from src.phase4.planner import plan_tools
from src.phase4.tools import ToolResult, execute_tool


NOT_AVAILABLE = "That information is not available in the saved forecast data for this run."
CHAT_SYSTEM_PROMPT = (
    "You are a helpful forecast analyst assistant for a power grid operations team.\n"
    "You explain load forecast results in plain, clear English — like a knowledgeable colleague.\n"
    "Your audience includes both technical operators and non-technical managers.\n"
    "Use short sentences. Be direct. Answer the actual question first, then add context.\n"
    "Use only the evidence provided below. Do not invent any numbers or metrics.\n"
    "If something is not in the evidence, say: \"I don't have that detail in the saved data for this run.\"\n"
    "Do not end every response with a robotic Sources list unless the user asks where data came from."
)


def _extract_numeric_tokens(text: str) -> set[str]:
    toks = re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)", text)
    return {t.lstrip("+") for t in toks}


def _has_new_numbers(output: str, allowed_source_text: str) -> bool:
    return not _extract_numeric_tokens(output).issubset(_extract_numeric_tokens(allowed_source_text))


def _source_item(path: str, fields: list[str]) -> dict[str, Any]:
    return {"path": path, "fields": fields}


def _result_sources(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tr in tool_results:
        for name, path in tr.created_paths.items():
            fields = []
            if "exceedance" in name:
                fields = ["timestamp", "load_mw", "capacity_mw", "exceedance_mw"]
            elif "weather" in name:
                fields = ["timestamp", "T2m", "RH2m", "Td2m"]
            elif "risk" in name:
                fields = ["kind", "timestamp", "expected_load_mw", "volatility_mw", "range_mw"]
            elif "revision" in name:
                fields = ["timestamp", "revision_volatility", "range", "consensus_median"]
            out.append(_source_item(path, fields))
    return out


def _build_readable_context(
    facts_pack: dict[str, Any],
    tool_summaries: dict[str, Any] | None,
    run_id: str,
) -> str:
    """Convert facts_pack and tool outputs into plain English context."""
    fp = facts_pack or {}
    lines: list[str] = []
    peak = fp.get("peak", {}) or {}
    cap = fp.get("capacity", {}) or {}
    weather = fp.get("weather", {}) or {}

    lines.append(f"RUN: {run_id}")
    lines.append(f"Risk Level: {fp.get('risk_level', 'N/A')}")
    lines.append(f"Forecast Stability: {fp.get('stability_label', fp.get('forecast_stability_level', 'N/A'))}")
    lines.append(f"Peak Load: {fp.get('peak_mw', peak.get('value_mw', 'N/A'))} MW at {fp.get('peak_time', peak.get('time', 'N/A'))}")
    lines.append(f"Grid Capacity: {fp.get('capacity_mw', peak.get('capacity_mw', 'N/A'))} MW")
    lines.append(f"Hours Above Capacity: {fp.get('exceedance_hours', cap.get('hours_above_capacity', 'N/A'))}")
    lines.append(f"Max Exceedance: {fp.get('max_exceedance_mw', peak.get('max_exceedance_mw', 'N/A'))} MW")
    lines.append(f"Weather Attribution R²: {fp.get('attribution_r2', weather.get('attribution_r2', 'N/A'))}")

    cap_watch = fp.get("capacity_watchlist", fp.get("capacity_watchlist_hours", [])) or []
    if cap_watch:
        lines.append("\nTop capacity risk hours:")
        for h in cap_watch[:3]:
            lines.append(
                f"  - {h.get('time','?')}: "
                f"{h.get('expected_mw', h.get('expected_load_mw', '?'))} MW, "
                f"exceedance {h.get('exceedance_mw','?')} MW"
            )

    stab_watch = fp.get("stability_watchlist", fp.get("stability_watchlist_hours", [])) or []
    if stab_watch:
        lines.append("\nTop stability risk hours (most likely to shift):")
        for h in stab_watch[:3]:
            lines.append(
                f"  - {h.get('time','?')}: "
                f"{h.get('expected_mw', h.get('expected_load_mw', '?'))} MW, "
                f"volatility {h.get('volatility_mw','?')} MW"
            )

    for tool_name, tool_out in (tool_summaries or {}).items():
        if tool_out:
            lines.append(f"\nTool result ({tool_name}): {str(tool_out)[:300]}")

    return "\n".join(lines)


def _build_prompt(context_text: str, question: str) -> str:
    return (
        f"Forecast data for this run:\n{context_text}\n\n"
        f"User question: {question}\n\n"
        "Answer the question directly in plain English. "
        "Be concise. Use only the data above."
    )


def _generate_with_messages(messages: list[dict[str, str]], temperature: float) -> str:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "api_key")
    if not key or key.strip() in {"", "api_key"}:
        raise RuntimeError("GROQ_API_KEY missing or still placeholder. Update .env with your real key.")
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
        max_completion_tokens=1000,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Groq returned empty response text.")
    return text


def _deterministic_fallback(question: str, plan: list[dict[str, Any]], facts: dict[str, Any], tool_results: list[ToolResult]) -> str:
    q = question.lower()
    if not plan:
        return NOT_AVAILABLE
    if any(step.get("tool") == "tool_exceedance_hours" for step in plan):
        ex = next((t for t in tool_results if t.tool_name == "tool_exceedance_hours"), None)
        w = next((t for t in tool_results if t.tool_name == "tool_weather_at_times"), None)
        if ex and int(ex.summary.get("exceedance_hours_count", 0)) == 0:
            return "No forecast hours are at or above capacity in the saved data, so exceedance weather rows are not available."
        if w:
            return (
                f"Forecast exceedance-weather rows: {w.summary.get('row_count')} based on {w.created_paths.get('exceedance_weather_csv')}. "
                f"Requested variables: {w.summary.get('requested_vars')}."
            )

    if any(step.get("tool") == "tool_top_risk_hours" for step in plan):
        cap = facts.get("capacity_watchlist_hours", [])[:3]
        stab = facts.get("stability_watchlist_hours", [])[:3]
        return (
            f"Top capacity watch hours: {[r.get('time') for r in cap]}. "
            f"Top stability watch hours: {[r.get('time') for r in stab]}."
        )

    if any(step.get("tool") == "tool_compare_revisions" for step in plan):
        tr = next((t for t in tool_results if t.tool_name == "tool_compare_revisions"), None)
        if tr:
            return (
                f"Max range is {tr.summary.get('max_range')} at {tr.summary.get('max_range_time')}; "
                f"average revision volatility is {tr.summary.get('avg_revision_volatility')}."
            )

    if "risk" in q:
        return (
            f"Risk level is {facts.get('risk_level')} with hours_above_capacity "
            f"{facts.get('capacity', {}).get('hours_above_capacity')} and max_exceedance_mw "
            f"{facts.get('peak', {}).get('max_exceedance_mw')}."
        )
    return NOT_AVAILABLE


def _write_chat_log(run_id: str, payload: dict[str, Any]) -> str:
    logs_dir = RUNS_ROOT / run_id / "phase4" / "chat_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    path = logs_dir / f"{ts}__{h}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


def answer_question(run_id: str, question: str, chat_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    phase4 = get_or_build_phase4(run_id, force=False, use_llm_summary=True)
    facts = phase4["facts_pack"]
    facts_path = phase4["files"]["facts_pack_json"]
    if chat_history:
        recent_history = chat_history[-6:]  # last 3 user+assistant pairs
    else:
        recent_history = []

    plan = plan_tools(question)
    tool_results: list[ToolResult] = []
    execution_errors: list[str] = []

    for step in plan:
        tool = step.get("tool")
        args = step.get("args") or {}
        if tool == "tool_weather_at_times":
            if not args.get("timestamps_source_csv"):
                ex = next((t for t in tool_results if t.tool_name == "tool_exceedance_hours"), None)
                if ex:
                    args["timestamps_source_csv"] = ex.created_paths.get("exceedance_hours_csv", "exceedance_hours.csv")
            elif args.get("timestamps_source_csv") == "exceedance_hours.csv":
                ex = next((t for t in tool_results if t.tool_name == "tool_exceedance_hours"), None)
                if ex:
                    args["timestamps_source_csv"] = ex.created_paths.get("exceedance_hours_csv", "exceedance_hours.csv")
        try:
            tr = execute_tool(run_id, tool, args)
            tool_results.append(tr)
        except Exception as exc:
            execution_errors.append(f"{tool}: {exc}")

    tool_summaries = {
        tr.tool_name: {
            "summary": tr.summary,
            "preview_markdown": tr.preview_markdown,
            "errors": tr.errors,
            "created_paths": tr.created_paths,
        }
        for tr in tool_results
    }
    context_text = _build_readable_context(facts, tool_summaries, run_id)
    prompt = _build_prompt(context_text, question)
    allowed_text = "\n".join([context_text, question, json.dumps(tool_summaries, ensure_ascii=False)])

    final = None
    status = "ok"
    for attempt in range(2):
        try:
            messages: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
            for turn in recent_history:
                role = str(turn.get("role", "")).strip()
                content = str(turn.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": prompt})
            text = _generate_with_messages(messages, temperature=0.2 if attempt == 0 else 0.0)
            if _has_new_numbers(text, allowed_text):
                final = None
                continue
            final = text
            break
        except Exception:
            final = None
            break

    if final is None:
        det = _deterministic_fallback(question, plan, facts, tool_results)
        if det == NOT_AVAILABLE:
            status = "not_available"
            final = NOT_AVAILABLE
        else:
            status = "ok"
            final = det

    sources = [
        _source_item(facts_path, [
            "risk_level",
            "forecast_stability_level",
            "peak_timing_agreement",
            "peak",
            "capacity",
            "stability",
            "weather",
            "capacity_watchlist_hours",
            "stability_watchlist_hours",
        ])
    ] + _result_sources(tool_results)

    sources_md_lines = ["", "Sources"]
    for s in sources:
        fields = ", ".join(s.get("fields", []))
        sources_md_lines.append(f"- {s.get('path')} (fields: {fields})")
    if any(w in question.lower() for w in ["source", "where did", "which file", "how do you know"]) and "sources" not in final.lower():
        final = final.rstrip() + "\n\n" + "\n".join(sources_md_lines) + "\n"

    log_payload = {
        "run_id": run_id,
        "question": question,
        "plan": plan,
        "tool_results": [
            {
                "tool_name": t.tool_name,
                "created_paths": t.created_paths,
                "summary": t.summary,
                "errors": t.errors,
            }
            for t in tool_results
        ],
        "execution_errors": execution_errors,
        "final_markdown": final,
        "sources": sources,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    log_path = _write_chat_log(run_id, log_payload)

    return {
        "final_markdown": final,
        "sources": sources,
        "tool_results": [
            {
                "tool_name": t.tool_name,
                "summary": t.summary,
                "created_paths": t.created_paths,
                "preview_markdown": t.preview_markdown,
                "errors": t.errors,
            }
            for t in tool_results
        ],
        "plan": plan,
        "status": status,
        "log_path": log_path,
        "execution_errors": execution_errors,
    }
