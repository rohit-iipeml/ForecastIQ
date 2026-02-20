from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.client import llm_generate_text
from src.phase4.generator import RUNS_ROOT, get_or_build_phase4
from src.phase4.planner import plan_tools
from src.phase4.tools import ToolResult, execute_tool


NOT_AVAILABLE = "That information is not available in the saved forecast data for this run."


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


def _evidence_bundle(
    facts_pack: dict[str, Any],
    facts_path: str,
    question: str,
    plan: list[dict[str, Any]],
    tool_results: list[ToolResult],
) -> tuple[str, str]:
    tool_summaries = []
    for tr in tool_results:
        tool_summaries.append(
            {
                "tool_name": tr.tool_name,
                "summary": tr.summary,
                "preview_markdown": tr.preview_markdown,
                "errors": tr.errors,
                "created_paths": tr.created_paths,
            }
        )

    evidence_json = {
        "question": question,
        "facts_pack_path": facts_path,
        "facts_pack": facts_pack,
        "plan": plan,
        "tool_outputs": tool_summaries,
    }
    allowed_text = json.dumps(evidence_json, ensure_ascii=False)
    prompt = (
        "Answer the user question using ONLY the evidence below.\n"
        "Do not invent metrics or numbers.\n"
        "If evidence is missing, answer exactly: 'That information is not available in the saved forecast data for this run.'\n"
        "Use plain, concise language.\n"
        "End with section header 'Sources' and bullet list of source paths/fields used.\n\n"
        f"Evidence:\n{json.dumps(evidence_json, ensure_ascii=False)}\n"
    )
    return prompt, allowed_text


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


def answer_question(run_id: str, question: str) -> dict[str, Any]:
    phase4 = get_or_build_phase4(run_id, force=False, use_llm_summary=True)
    facts = phase4["facts_pack"]
    facts_path = phase4["files"]["facts_pack_json"]

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

    prompt, allowed_text = _evidence_bundle(facts, facts_path, question, plan, tool_results)
    system = (
        "You are an assistant explaining a load forecast. "
        "Answer using ONLY provided evidence. "
        "Do not introduce any numbers not present in evidence. "
        "If evidence is missing, answer exactly: 'That information is not available in the saved forecast data for this run.'"
    )

    final = None
    status = "ok"
    for attempt in range(2):
        try:
            text = llm_generate_text(
                prompt=prompt,
                system=system,
                model="llama-3.1-8b-instant",
                temperature=0.2 if attempt == 0 else 0.0,
                max_tokens=700,
            ).strip()
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
    if "sources" not in final.lower():
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
