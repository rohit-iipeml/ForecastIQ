from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase3.llm_writer import generate_briefing_llm_md  # noqa: E402


def main() -> None:
    run_id = "2025010900"
    result = generate_briefing_llm_md(run_id, force=True)
    out = Path(result["path"])
    print("status:", result["status"])
    print("path:", out)
    if not out.exists():
        raise SystemExit("briefing_llm.md was not created")
    print("--- first 10 lines ---")
    lines = out.read_text(encoding="utf-8").splitlines()
    for line in lines[:10]:
        print(line)


if __name__ == "__main__":
    main()
