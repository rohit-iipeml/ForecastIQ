from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3.generator import generate_phase3_outputs  # noqa: E402
from src.phase3.schemas import validate_action_items_json, validate_briefing_json  # noqa: E402


def main() -> None:
    run_id = "2025012500"
    out = generate_phase3_outputs(run_id, force=False)

    phase3_dir = Path(f"outputs/runs/{run_id}/phase3")
    required = [
        phase3_dir / "phase3_input.json",
        phase3_dir / "briefing.json",
        phase3_dir / "briefing.md",
        phase3_dir / "action_items.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing Phase3 files: {missing}")

    briefing = json.loads((phase3_dir / "briefing.json").read_text(encoding="utf-8"))
    actions = json.loads((phase3_dir / "action_items.json").read_text(encoding="utf-8"))
    validate_briefing_json(briefing)
    validate_action_items_json(actions)

    print("confidence_grade:", briefing["confidence_grade"])
    bullets = briefing.get("executive_summary", [])[:3]
    print("executive_summary_top3:")
    for b in bullets:
        print("-", b)

    print("action_titles:")
    for a in actions.get("items", []):
        print("-", a["title"])


if __name__ == "__main__":
    main()
