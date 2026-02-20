from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3.generator import generate_phase3_outputs  # noqa: E402
from src.phase3.schemas import validate_briefing_json  # noqa: E402


RUN_IDS = ["2025010900", "2025011000"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_new_fields(briefing: dict) -> None:
    required = [
        "risk_level",
        "forecast_stability_level",
        "peak_timing_agreement",
        "capacity_watchlist_hours",
        "stability_watchlist_hours",
    ]
    for key in required:
        if key not in briefing:
            raise AssertionError(f"Missing briefing field: {key}")
    if len(briefing.get("capacity_watchlist_hours", [])) == 0:
        raise AssertionError("capacity_watchlist_hours must not be empty")
    if len(briefing.get("stability_watchlist_hours", [])) == 0:
        raise AssertionError("stability_watchlist_hours must not be empty")


def _assert_weather_narrative_rule(briefing: dict, briefing_md: str) -> None:
    r2 = (briefing.get("weather_load_link") or {}).get("attribution_r2")
    if r2 is None:
        if "could not be quantified" not in briefing_md:
            raise AssertionError("Expected missing-attribution wording in briefing.md")
        if "major reason" in briefing_md.lower():
            raise AssertionError("briefing.md must not claim 'major reason' when attribution is missing")


def _assert_llm_structure(llm_md: str) -> None:
    for token in ["Risk Level", "Forecast Stability", "Peak Timing Agreement", "Watchlist Hours"]:
        if token.lower() not in llm_md.lower():
            raise AssertionError(f"briefing_llm.md missing expected section/token: {token}")


def run_for(run_id: str) -> None:
    out = generate_phase3_outputs(run_id, force=True, use_llm_writer=False)
    phase3_dir = Path(out["files"]["briefing_json"]).parent

    briefing = _load_json(phase3_dir / "briefing.json")
    validate_briefing_json(briefing)
    _assert_new_fields(briefing)

    briefing_md = (phase3_dir / "briefing.md").read_text(encoding="utf-8")
    _assert_weather_narrative_rule(briefing, briefing_md)

    out_llm = generate_phase3_outputs(run_id, force=True, use_llm_writer=True)
    llm_path = Path(out_llm["files"].get("briefing_llm_md", ""))
    if not llm_path.exists():
        raise AssertionError(f"Missing briefing_llm.md for run {run_id}")
    llm_md = llm_path.read_text(encoding="utf-8")
    _assert_llm_structure(llm_md)

    print(f"run_id={run_id}")
    print(f"  risk_level={briefing.get('risk_level')}")
    print(f"  forecast_stability_level={briefing.get('forecast_stability_level')}")
    print(f"  peak_timing_agreement={briefing.get('peak_timing_agreement')}")
    print(f"  llm_status={out_llm.get('llm', {}).get('status')}")


def main() -> None:
    for run_id in RUN_IDS:
        run_for(run_id)
    print("Phase3-C checks passed.")


if __name__ == "__main__":
    main()
