from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    run_id = "2025010900"
    phase3_dir = Path(f"outputs/runs/{run_id}/phase3")

    llm_md_path = phase3_dir / "briefing_llm.md"
    briefing_json_path = phase3_dir / "briefing.json"

    if not llm_md_path.exists():
        raise AssertionError(f"Missing file: {llm_md_path}")
    if not briefing_json_path.exists():
        raise AssertionError(f"Missing file: {briefing_json_path}")

    llm_md = llm_md_path.read_text(encoding="utf-8")
    first_line = llm_md.splitlines()[0] if llm_md.splitlines() else ""
    if first_line.strip().startswith("⚠️"):
        raise AssertionError("briefing_llm.md should not start with warning banner")

    briefing = json.loads(briefing_json_path.read_text(encoding="utf-8"))
    rows = briefing.get("stability_watchlist_hours", [])
    if not rows:
        raise AssertionError("stability_watchlist_hours is empty")
    for i, row in enumerate(rows):
        if "expected_load_display" not in row:
            raise AssertionError(f"Row {i} missing expected_load_display")
        if str(row.get("expected_load_display", "")).strip().upper() == "NA":
            raise AssertionError(f"Row {i} has literal NA in expected_load_display")

    print("Phase3-C.1 patch checks passed.")


if __name__ == "__main__":
    main()
