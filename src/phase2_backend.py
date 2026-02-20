from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.revision_analysis import (
    build_forecast_matrix,
    compute_revision_metrics,
    compute_value_of_information,
    list_overlapping_inits,
    save_phase2_artifacts,
)
from src.gru_universal import ingest_load_csv
from src.phase1_backend import load_config, run_for_date


def _phase2_dir(runs_dir: Path, target_run_id: str) -> Path:
    return runs_dir / target_run_id / "phase2"


def _phase2_cache_exists(runs_dir: Path, target_run_id: str) -> bool:
    p = _phase2_dir(runs_dir, target_run_id)
    required = [
        p / "day_metrics.json",
        p / "forecast_matrix.csv",
        p / "per_hour_metrics.csv",
        p / "consensus_series.csv",
        p / "exceedance_proxy.csv",
        p / "peak_by_init.csv",
        p / "phase2_summary.json",
    ]
    return all(x.exists() for x in required)


def _is_phase2_cache_compatible(runs_dir: Path, target_run_id: str, init_hour: str) -> bool:
    """Ensure cached matrix uses only requested init hour columns."""
    matrix_path = _phase2_dir(runs_dir, target_run_id) / "forecast_matrix.csv"
    if not matrix_path.exists():
        return False
    try:
        m = pd.read_csv(matrix_path, nrows=1)
    except Exception:
        return False
    pred_cols = [c for c in m.columns if c.startswith("pred_init_")]
    if not pred_cols:
        return False
    return all(c[-2:] == init_hour for c in pred_cols)


def _load_phase2_cache(runs_dir: Path, target_run_id: str) -> dict[str, Any]:
    p = _phase2_dir(runs_dir, target_run_id)
    day_metrics = json.loads((p / "day_metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((p / "phase2_summary.json").read_text(encoding="utf-8"))
    files = dict(summary.get("files", {}))
    files["phase2_summary_json"] = str(p / "phase2_summary.json")
    out = {
        "cached": True,
        "target_run_id": target_run_id,
        "day_metrics": day_metrics,
        "forecast_matrix_df": pd.read_csv(p / "forecast_matrix.csv", index_col=0, parse_dates=True),
        "per_hour_metrics_df": pd.read_csv(p / "per_hour_metrics.csv", parse_dates=["target_timestamp"]),
        "consensus_series_df": pd.read_csv(p / "consensus_series.csv", parse_dates=["target_timestamp"]),
        "exceedance_proxy_df": pd.read_csv(p / "exceedance_proxy.csv", parse_dates=["target_timestamp"]),
        "peak_table_df": pd.read_csv(p / "peak_by_init.csv", parse_dates=["peak_timestamp"]),
        "voi_df": pd.read_csv(p / "voi.csv") if (p / "voi.csv").exists() else pd.DataFrame(),
        "files": files,
        "phase2_summary_path": str(p / "phase2_summary.json"),
    }
    return out


def _required_prior_run_ids(target_date: pd.Timestamp, init_hour: str, days_back: int = 4) -> list[str]:
    req = []
    for d in range(1, days_back + 1):
        t = target_date - pd.Timedelta(days=d)
        req.append(t.strftime("%Y%m%d") + init_hour)
    return req


def _build_actuals_history_df(history_csv: Path, load_daily_dir: Path) -> pd.DataFrame:
    hist = pd.read_csv(history_csv, parse_dates=["timestamp", "init_time"])
    load_col = "load_MW" if "load_MW" in hist.columns else ("Load" if "Load" in hist.columns else None)
    if load_col is None:
        for c in ["load", "value"]:
            if c in hist.columns:
                load_col = c
                break
    if load_col is None:
        raise ValueError("No recognized load column found in history.")
    hist_part = hist[["timestamp", load_col]].rename(columns={load_col: "load_MW"})

    daily_parts = []
    for f in sorted(load_daily_dir.glob("GVL_D_*.csv")):
        try:
            daily_parts.append(ingest_load_csv(f))
        except Exception:
            continue
    if daily_parts:
        daily = pd.concat(daily_parts, ignore_index=True)
        merged = pd.concat([hist_part, daily], ignore_index=True)
    else:
        merged = hist_part.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    merged["load_MW"] = pd.to_numeric(merged["load_MW"], errors="coerce")
    merged = merged.dropna(subset=["timestamp", "load_MW"])
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return merged.reset_index(drop=True)


def get_or_build_phase2(target_date: str, init_hour: str = "00") -> dict[str, Any]:
    cfg = load_config()
    target_day = pd.to_datetime(target_date).normalize()
    target_run_id = target_day.strftime("%Y%m%d") + init_hour

    runs_dir = Path(cfg["paths"]["runs_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    history_csv = Path(cfg["paths"]["history_00"] if init_hour == "00" else cfg["paths"]["history_12"])
    load_daily_dir = Path(cfg["paths"]["load_daily_dir"])
    horizon_hours = int(cfg.get("horizon_hours", 90))

    # Ensure target run exists (cache-first in phase1).
    target_phase1 = run_for_date(target_day.strftime("%Y-%m-%d"), init_hour)
    capacity_mw = float(target_phase1["metrics"]["capacity_mw"])

    if _phase2_cache_exists(runs_dir, target_run_id) and _is_phase2_cache_compatible(runs_dir, target_run_id, init_hour):
        return _load_phase2_cache(runs_dir, target_run_id)

    # Discover overlaps from disk first.
    overlap_ids = list_overlapping_inits(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        runs_dir=str(runs_dir),
        outputs_dir=str(outputs_dir),
        horizon_hours=horizon_hours,
    )

    # Generate missing required prior runs (same init hour, up to 4 days back) if in allowed range.
    allowed_min = pd.to_datetime(cfg["allowed_date_min"]).normalize()
    allowed_max = pd.to_datetime(cfg["allowed_date_max"]).normalize()
    required = _required_prior_run_ids(target_day, init_hour, days_back=4)
    for rid in required:
        rid_ts = pd.to_datetime(rid, format="%Y%m%d%H")
        if rid in overlap_ids:
            continue
        if not (allowed_min <= rid_ts.normalize() <= allowed_max):
            continue
        run_for_date(rid_ts.strftime("%Y-%m-%d"), init_hour)

    # Refresh overlap after optional generation.
    overlap_ids = list_overlapping_inits(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        runs_dir=str(runs_dir),
        outputs_dir=str(outputs_dir),
        horizon_hours=horizon_hours,
    )

    forecast_matrix = build_forecast_matrix(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        available_run_ids=overlap_ids,
        runs_dir=str(runs_dir),
        outputs_dir=str(outputs_dir),
    )
    if forecast_matrix.shape[1] < 2:
        day_metrics = {
            "n_init_runs": int(forecast_matrix.shape[1]),
            "message": "Insufficient overlapping inits (<2) for full revision analytics.",
            "capacity_mw": capacity_mw,
        }
        consensus = pd.DataFrame(columns=["target_timestamp", "median", "q1", "q3", "min", "max"])
        exceed = pd.DataFrame(columns=["target_timestamp", "exceed_prob_proxy", "median", "capacity_mw"])
        per_hour = pd.DataFrame(columns=["target_timestamp"])
        peak = pd.DataFrame(columns=["init_run", "peak_timestamp", "peak_hour_of_day", "peak_value"])
    else:
        rev = compute_revision_metrics(forecast_matrix, capacity_mw)
        day_metrics = rev["day_metrics"]
        per_hour = rev["per_hour_metrics"]
        consensus = rev["consensus_series"]
        exceed = rev["exceedance_proxy"]
        peak = rev["peak_table"]

    # Real-time cut rule: actuals only for timestamps < X 00:00.
    realtime_cut = target_day
    history_actuals = _build_actuals_history_df(history_csv, load_daily_dir)
    voi_df = compute_value_of_information(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        forecast_matrix_df=forecast_matrix,
        history_df=history_actuals.rename(columns={"load_MW": "load_MW"}),
        realtime_cut_ts=realtime_cut,
    )

    files = save_phase2_artifacts(
        run_id_for_target_day=target_run_id,
        outputs_runs_dir=str(runs_dir),
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        forecast_matrix_df=forecast_matrix,
        per_hour_metrics_df=per_hour,
        day_metrics_json=day_metrics,
        voi_df=voi_df,
        peak_table_df=peak,
        consensus_series_df=consensus,
        exceedance_proxy_df=exceed,
    )

    return {
        "cached": False,
        "target_run_id": target_run_id,
        "day_metrics": day_metrics,
        "forecast_matrix_df": forecast_matrix,
        "per_hour_metrics_df": per_hour,
        "consensus_series_df": consensus,
        "exceedance_proxy_df": exceed,
        "peak_table_df": peak,
        "voi_df": voi_df,
        "files": files,
        "phase2_summary_path": files.get("phase2_summary_json"),
    }
