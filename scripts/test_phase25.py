from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase25_backend import get_or_build_phase25  # noqa: E402


def main() -> None:
    result = get_or_build_phase25("2025-01-25", "00")
    print("target_run_id:", result["target_run_id"])
    print("cached:", result["cached"])

    weather_day = result.get("weather_day_metrics", {})
    t2m = weather_day.get("variables", {}).get("T2m", {})
    print("T2m disagreement index:", t2m.get("variable_disagreement_index_iqr"))
    print("attribution R^2:", result.get("attribution_fit", {}).get("r2"))
    high_risk = int(result["joint_ops_risk_df"]["HighOpsRisk"].sum()) if not result["joint_ops_risk_df"].empty else 0
    print("high ops risk hours:", high_risk)

    files = result.get("files", {})
    required = [
        "weather_per_hour_metrics_csv",
        "weather_day_metrics_json",
        "revision_pairs_csv",
        "attribution_fit_json",
        "correlation_table_csv",
        "joint_ops_risk_csv",
        "phase25_summary_json",
    ]
    missing = [k for k in required if not files.get(k) or not Path(files[k]).exists()]
    if missing:
        raise SystemExit(f"Missing phase25 artifacts: {missing}")
    print("phase25 artifacts: OK")

    if t2m.get("n_init_runs", 0) < 2:
        print("WARNING: limited weather overlap; analysis completed with reduced confidence.")


if __name__ == "__main__":
    main()
