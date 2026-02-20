from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.client import llm_generate_text


RUNS_ROOT = Path("outputs/runs")
PROMPT_PATH = Path("prompts/briefing_writer.md")


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt template missing: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_llm_writer_prompt(
    phase3_input: dict[str, Any],
    briefing_json: dict[str, Any],
    action_items_json: dict[str, Any],
    baseline_md: str,
) -> str:
    template = load_prompt_template()
    payload = (
        f"{template}\n\n"
        "### phase3_input.json\n"
        f"{json.dumps(phase3_input, separators=(',', ':'), ensure_ascii=False)}\n\n"
        "### briefing.json\n"
        f"{json.dumps(briefing_json, separators=(',', ':'), ensure_ascii=False)}\n\n"
        "### action_items.json\n"
        f"{json.dumps(action_items_json, separators=(',', ':'), ensure_ascii=False)}\n\n"
        "### baseline briefing.md\n"
        f"{baseline_md}\n"
    )
    return payload


def _extract_numeric_tokens(text: str) -> set[str]:
    # Capture ints/floats including negatives; normalize leading + signs.
    toks = re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)", text)
    return {t.lstrip("+") for t in toks}


def _has_new_numbers(output_text: str, allowed_sources_text: str) -> bool:
    out_nums = _extract_numeric_tokens(output_text)
    allowed = _extract_numeric_tokens(allowed_sources_text)
    return not out_nums.issubset(allowed)


def _has_required_sections(md_text: str) -> bool:
    lower = md_text.lower()
    required = [
        "operational takeaway",
        "executive summary",
        "peak and capacity",
        "forecast stability",
        "weather impact",
        "watchlist hours",
        "recommended actions",
        "notes",
    ]
    return all(r in lower for r in required)


def _has_invalid_weather_claim(md_text: str, briefing_json: dict[str, Any]) -> bool:
    r2 = (briefing_json.get("weather_load_link") or {}).get("attribution_r2")
    if r2 is not None:
        return False
    lower = md_text.lower()
    banned = [
        "major reason",
        "major driver",
        "driving most",
        "weather explains",
        "explains about",
    ]
    return any(token in lower for token in banned)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_briefing_llm_md(run_id: str, force: bool = False) -> dict[str, Any]:
    phase3_dir = RUNS_ROOT / run_id / "phase3"
    if not phase3_dir.exists():
        raise FileNotFoundError(f"Phase3 directory not found: {phase3_dir}")

    p_input = phase3_dir / "phase3_input.json"
    p_briefing = phase3_dir / "briefing.json"
    p_actions = phase3_dir / "action_items.json"
    p_baseline = phase3_dir / "briefing.md"
    p_out = phase3_dir / "briefing_llm.md"
    p_meta = phase3_dir / "briefing_llm.meta.json"

    for p in [p_input, p_briefing, p_actions, p_baseline]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing for LLM writer: {p}")

    phase3_input = _read_json(p_input)
    briefing_json = _read_json(p_briefing)
    action_items = _read_json(p_actions)
    baseline_md = p_baseline.read_text(encoding="utf-8")

    prompt_text = build_llm_writer_prompt(phase3_input, briefing_json, action_items, baseline_md)
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    model = "llama-3.1-8b-instant"

    if not force and p_out.exists() and p_meta.exists():
        try:
            meta = _read_json(p_meta)
        except Exception:
            meta = {}
        if meta.get("prompt_hash") == prompt_hash and meta.get("run_id") == run_id:
            return {
                "run_id": run_id,
                "cached": True,
                "status": meta.get("status", "ok"),
                "path": str(p_out),
                "meta_path": str(p_meta),
                "model": meta.get("model", model),
            }

    status = "ok"
    reason = None
    try:
        llm_md = llm_generate_text(
            prompt=prompt_text,
            system=(
                "You rewrite operational markdown. Preserve all numbers exactly. "
                "Do not add any new numeric values."
            ),
            model=model,
            temperature=0.2,
            max_tokens=1200,
        ).strip()

        allowed_sources = "\n".join(
            [
                json.dumps(phase3_input, ensure_ascii=False),
                json.dumps(briefing_json, ensure_ascii=False),
                json.dumps(action_items, ensure_ascii=False),
                baseline_md,
            ]
        )
        if _has_new_numbers(llm_md, allowed_sources):
            status = "fallback"
            reason = "new_numbers"
        elif not _has_required_sections(llm_md):
            status = "fallback"
            reason = "missing_sections"
        elif _has_invalid_weather_claim(llm_md, briefing_json):
            status = "fallback"
            reason = "weather_claim_on_na"
    except Exception as exc:
        status = "fallback"
        reason = f"llm_error: {exc}"
        llm_md = ""

    if status == "fallback":
        footer = "\n\n---\nNote: LLM polish unavailable for this run; showing deterministic briefing.\n"
        out_md = baseline_md.rstrip() + footer
    else:
        out_md = llm_md

    p_out.write_text(out_md, encoding="utf-8")
    meta = {
        "run_id": run_id,
        "model": model,
        "prompt_hash": prompt_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
    }
    p_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "cached": False,
        "status": status,
        "reason": reason,
        "path": str(p_out),
        "meta_path": str(p_meta),
        "model": model,
    }
