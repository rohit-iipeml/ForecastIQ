from __future__ import annotations

import argparse

from src.phase4.generator import get_or_build_phase4


def main() -> None:
    p = argparse.ArgumentParser(description="Build Phase 4 artifacts.")
    p.add_argument("--run_id", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = get_or_build_phase4(args.run_id, force=args.force, use_llm_summary=True)
    print(f"facts_pack_json: {out['files']['facts_pack_json']}")
    print(f"simple_summary_md: {out['files']['simple_summary_md']}")


if __name__ == "__main__":
    main()
