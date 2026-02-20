from __future__ import annotations

import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase4.generator import get_or_build_phase4  # noqa: E402
from src.phase4.chatbot import answer_forecast_question  # noqa: E402


def _nums(text: str) -> set[str]:
    return {t.lstrip('+') for t in re.findall(r"(?<!\w)[+-]?(?:\d+\.\d+|\d+)(?!\w)", text)}


def main() -> None:
    run_id = "2025010900"
    out = get_or_build_phase4(run_id, force=True, use_llm_summary=True)
    facts_path = Path(out["files"]["facts_pack_json"])
    summary_path = Path(out["files"]["simple_summary_md"])
    assert facts_path.exists(), "facts_pack.json missing"
    assert summary_path.exists(), "simple_summary.md missing"

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    allowed = _nums(json.dumps(facts, ensure_ascii=False))

    questions = [
        "Why is risk severe?",
        "Is tomorrow stable?",
        "Which hours are most risky?",
        "Is weather driving this?",
    ]
    print("facts_pack:", facts_path)
    print("simple_summary:", summary_path)
    for q in questions:
        ans = answer_forecast_question(run_id, q)
        text = ans.get("answer", "")
        bad = _nums(text) - allowed
        print(f"Q: {q}")
        print(f"A: {text}")
        print(f"mode={ans.get('mode')} status={ans.get('status')} sources={ans.get('sources')}")
        if bad:
            raise AssertionError(f"Answer has numbers not in facts_pack.json: {bad}")

    print("Phase 4 checks passed.")


if __name__ == "__main__":
    main()
