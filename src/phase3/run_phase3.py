from __future__ import annotations

import argparse

from src.phase3.generator import generate_phase3_outputs


def main() -> None:
    p = argparse.ArgumentParser(description="Run Phase 3 deterministic briefing generation.")
    p.add_argument("--run_id", required=True, help="Run id in YYYYMMDDHH format")
    p.add_argument("--force", action="store_true", help="Force regenerate even if cached outputs exist")
    p.add_argument("--llm", action="store_true", help="Generate LLM-polished briefing_llm.md")
    args = p.parse_args()

    out = generate_phase3_outputs(args.run_id, force=args.force, use_llm_writer=args.llm)
    print(f"briefing_md: {out['files']['briefing_md']}")
    print(f"risk_level: {out.get('risk_level')}")
    print(f"forecast_stability_level: {out.get('forecast_stability_level')}")
    print(f"peak_timing_agreement: {out.get('peak_timing_agreement')}")
    print(f"confidence_grade: {out['confidence_grade']}")
    print(f"num_actions_file: {out['files']['action_items_json']}")
    if args.llm:
        llm = out.get("llm", {})
        print(f"llm_status: {llm.get('status')}")
        print(f"briefing_llm_md: {llm.get('path')}")


if __name__ == "__main__":
    main()
