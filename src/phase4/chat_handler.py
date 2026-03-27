from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.phase4.generator import RUNS_ROOT, get_or_build_phase4
from src.phase4.planner import plan_tools
from src.phase4.tools import ToolResult, execute_tool


NOT_AVAILABLE = "That information is not available in the saved forecast data for this run."
CHAT_SYSTEM_PROMPT = """\
You are an expert power grid operations analyst embedded in the ForecastIQ situational awareness system.

ROLE:
You have direct access to the current run's facts_pack (peak load, capacity status, risk level, watchlist hours, \
weather attribution, ramp risk, backtest quality, and recommended actions) plus tool query results and retrieved \
historical briefing context. Answer like a seasoned NERC-certified grid analyst: precise, direct, and quantitative.

CORE BEHAVIOR RULES:
1. Lead with the number or status, then the context. Never bury the answer in prose.
2. Use ONLY values from the provided facts_pack and tool results. Never estimate or generalize beyond the data.
3. The facts_pack is your primary ground truth — if a value is there, you have access to it. \
   Never say "I don't have access to" when the data is provided below.
4. If data is genuinely absent from the provided context, say: \
   "That metric is not in the saved data for run [run_id]." Do not guess.
5. For multi-part questions, use labeled sections (**Peak Demand**, **Risk Status**, etc.).
6. Keep answers under 150 words unless a full briefing is explicitly requested.

HISTORICAL CONTEXT USAGE:
When RELEVANT HISTORICAL CONTEXT FROM PAST BRIEFINGS is provided, you MUST compare it to the current run explicitly.
Example: "In run 2025-01-23, risk was SEVERE with 15 hours above capacity. Current run 2025-01-25 shows 3 hours — \
a significant improvement, though still rated SEVERE due to 25.9 MW max exceedance."
Always cite the run_id when referencing historical data.

QUERY ROUTING — use the matching data source for each question type:
- Peak demand / timing         → facts_pack.peak (value_mw, time)
- Risk level / why SEVERE?     → facts_pack.risk_level + capacity.hours_above_capacity + peak.max_exceedance_mw
- Which hours are at risk?     → facts_pack.capacity_watchlist_hours (top entries with MW values)
- Which hours are uncertain?   → facts_pack.stability_watchlist_hours (volatility_mw, range_mw)
- Weather / load driver?       → facts_pack.weather (attribution_r2, top_variable, correlation)
- Historical comparison?       → HISTORICAL CONTEXT chunks (cite run_id, compare numbers directly)
- Operator actions?            → facts_pack.recommended_actions (recommended_next_step field)
- Model accuracy / backtest?   → facts_pack.backtest_quality (flag, mae_pct)
- Ramp risk?                   → facts_pack.ramp_risk (ramp_risk_flag, max_ramp_up_mw, max_ramp_down_mw)
- Energy above capacity?       → facts_pack.energy_at_risk_mwh
- Forecast stability?          → facts_pack.forecast_stability_level + stability.disagreement_index

ANSWER FORMAT BY QUERY TYPE:
- Single-number question: "[Value] MW / [Status]" → one sentence context.
- Risk / status question: State tier → then 2–3 supporting metrics as brief bullets.
- Watchlist question: Bullet each hour with "HH:00 — X MW, Y MW above capacity".
- Historical comparison: "Current run [id]: X — Historical run [id]: Y (±Z% difference)."
- What-if / scenario: Reason from facts explicitly; label it "Model-based estimate, not a live forecast."

Do not append a Sources list at the end unless the user explicitly asks where data came from.\
"""


# ---------------------------------------------------------------------------
# LLM factory — instantiated per call so temperature can vary on retry
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.3) -> ChatGroq:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY", "api_key")
    if not key or key.strip() in {"", "api_key"}:
        raise RuntimeError("GROQ_API_KEY missing or still placeholder. Update .env with your real key.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=key,
        temperature=temperature,
        # streaming=True is set so the model is ready for .stream() calls;
        # answer_question() currently collects the full response via .invoke()
        # TODO: refactor answer_question() into a generator and use
        #       st.write_stream() in app.py for real-time streaming output.
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Hallucination guard (unchanged)
# ---------------------------------------------------------------------------

def _extract_numeric_tokens(text: str) -> set[str]:
    toks = re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)", text)
    return {t.lstrip("+") for t in toks}


def _has_new_numbers(output: str, allowed_source_text: str) -> bool:
    return not _extract_numeric_tokens(output).issubset(_extract_numeric_tokens(allowed_source_text))


# ---------------------------------------------------------------------------
# Source helpers (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Context builders (unchanged)
# ---------------------------------------------------------------------------

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


def _build_rag_context(retrieved: list[dict[str, Any]]) -> str:
    """Format retrieved RAG chunks into a labeled context block."""
    if not retrieved:
        return ""
    lines = ["RELEVANT HISTORICAL CONTEXT FROM PAST BRIEFINGS:"]
    for item in retrieved:
        meta = item.get("metadata") or {}
        run = meta.get("run_id", "unknown")
        section = meta.get("section_title", "")
        risk = meta.get("risk_level", "")
        lines.append(f"\n--- Run {run} | {section} | Risk: {risk} ---")
        lines.append(item.get("text", "").strip())
    lines.append("\n---END HISTORICAL CONTEXT---")
    return "\n".join(lines)


def _build_prompt(context_text: str, question: str, rag_context: str = "") -> str:
    """Build the human message body. RAG context is prepended when non-empty."""
    parts = []
    if rag_context:
        parts.append(rag_context + "\n")
    parts.append(
        f"Forecast data for this run:\n{context_text}\n\n"
        f"User question: {question}\n\n"
        "Answer the question directly in plain English. "
        "Be concise. Use only the data above."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LangChain message construction
# ---------------------------------------------------------------------------

def _build_lc_messages(
    prompt: str,
    lc_memory: Any | None,
    chat_history: list[dict[str, str]] | None,
    window: int = 6,
) -> list:
    """Build a list of LangChain message objects for the LLM call.

    Priority: lc_memory (ChatMessageHistory) > chat_history (raw list fallback).
    Applies a sliding window of `window` messages from history.
    """
    msgs: list = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]

    if lc_memory is not None:
        # Load from LangChain ChatMessageHistory, apply window
        history_msgs = lc_memory.messages[-window:]
        msgs.extend(history_msgs)
    elif chat_history:
        for turn in chat_history[-window:]:
            role = str(turn.get("role", "")).strip()
            content = str(turn.get("content", "")).strip()
            if role == "user" and content:
                msgs.append(HumanMessage(content=content))
            elif role == "assistant" and content:
                msgs.append(AIMessage(content=content))

    msgs.append(HumanMessage(content=prompt))
    return msgs


# ---------------------------------------------------------------------------
# LLM invocation (replaces _generate_with_messages)
# ---------------------------------------------------------------------------

def _invoke_llm(lc_messages: list, temperature: float) -> str:
    """Invoke ChatGroq with the given message list and return the text response."""
    llm = _get_llm(temperature=temperature)
    result = llm.invoke(lc_messages)
    text = (result.content or "").strip()
    if not text:
        raise RuntimeError("LLM returned empty response.")
    return text


# ---------------------------------------------------------------------------
# Deterministic fallback (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chat log writer (unchanged)
# ---------------------------------------------------------------------------

def _write_chat_log(run_id: str, payload: dict[str, Any]) -> str:
    logs_dir = RUNS_ROOT / run_id / "phase4" / "chat_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    path = logs_dir / f"{ts}__{h}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(
    run_id: str,
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    lc_memory: Any | None = None,
) -> dict[str, Any]:
    """Answer a forecast question using tool grounding + LangChain LLM + RAG.

    Args:
        run_id: The forecast run identifier.
        question: The user's question.
        chat_history: Optional raw list of {"role", "content"} dicts (legacy path).
        lc_memory: Optional ChatMessageHistory instance (preferred; managed by UI
                   in st.session_state["lc_memory_<run_id>"]). When provided,
                   the Q&A pair is appended to it after answering.
    """
    phase4 = get_or_build_phase4(run_id, force=False, use_llm_summary=True)
    facts = phase4["facts_pack"]
    facts_path = phase4["files"]["facts_pack_json"]

    # --- Tool planning and execution (unchanged) ---
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

    # --- RAG retrieval via LangChain retriever ---
    rag_chunks: list[dict[str, Any]] = []
    try:
        from src.rag_store import RAGStore
        retriever = RAGStore().get_langchain_retriever(k=3)
        lc_docs = retriever.invoke(question)
        rag_chunks = [
            {"text": doc.page_content, "metadata": doc.metadata, "distance": 0.0}
            for doc in lc_docs
        ]
    except Exception:
        pass
    rag_context = _build_rag_context(rag_chunks)
    # Collect distinct run_ids retrieved (for UI display)
    rag_retrieved_run_ids: list[str] = list(dict.fromkeys(
        c["metadata"].get("run_id", "") for c in rag_chunks if c.get("metadata", {}).get("run_id")
    ))

    # --- Prompt and message construction ---
    context_text = _build_readable_context(facts, tool_summaries, run_id)
    prompt = _build_prompt(context_text, question, rag_context=rag_context)
    allowed_text = "\n".join([context_text, question, json.dumps(tool_summaries, ensure_ascii=False), rag_context])

    # --- LangChain LLM call with retry ---
    final = None
    status = "ok"
    for attempt in range(2):
        try:
            lc_messages = _build_lc_messages(
                prompt=prompt,
                lc_memory=lc_memory,
                chat_history=chat_history,
            )
            text = _invoke_llm(lc_messages, temperature=0.2 if attempt == 0 else 0.0)
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

    # --- Update LangChain memory with this Q&A turn ---
    if lc_memory is not None:
        lc_memory.add_user_message(question)
        lc_memory.add_ai_message(final)

    # --- Sources (unchanged) ---
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

    # --- Log (unchanged) ---
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
        "rag_retrieved_count": len(rag_chunks),
        "rag_retrieved_run_ids": rag_retrieved_run_ids,
    }
