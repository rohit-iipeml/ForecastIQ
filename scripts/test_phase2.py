from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase2_backend import get_or_build_phase2  # noqa: E402


def main() -> None:
    target_date = "2025-01-25"
    init_hour = "00"
    result = get_or_build_phase2(target_date, init_hour)

    print("target_run_id:", result["target_run_id"])
    print("cached:", result["cached"])
    print("day_metrics:")
    for k, v in result["day_metrics"].items():
        print(f"  - {k}: {v}")

    files = result.get("files", {})
    required = [
        "forecast_matrix_csv",
        "per_hour_metrics_csv",
        "day_metrics_json",
        "peak_by_init_csv",
        "consensus_series_csv",
        "exceedance_proxy_csv",
        "phase2_summary_json",
    ]
    missing = [k for k in required if not files.get(k) or not Path(files[k]).exists()]
    if missing:
        raise SystemExit(f"Missing expected phase2 artifacts: {missing}")
    print("artifacts: OK")

    matrix = result["forecast_matrix_df"]
    n_init_cols = len([c for c in matrix.columns if str(c).startswith("pred_init_")])
    if n_init_cols < 2:
        print(f"WARNING: forecast_matrix has only {n_init_cols} init columns (expected >=2).")
    else:
        print(f"forecast_matrix init columns: {n_init_cols} (OK)")


if __name__ == "__main__":
    main()
