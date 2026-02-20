from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.revision_analysis import list_overlapping_inits
from src.analysis.weather_revision_analysis import (
    build_weather_matrix,
    compute_joint_risk_flags,
    compute_weather_load_attribution,
    compute_weather_revision_metrics,
    save_phase25_artifacts,
)
from src.phase1_backend import load_config, run_for_date
from src.phase2_backend import get_or_build_phase2


def _phase25_dir(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id / "phase25"


def _cache_exists(runs_dir: Path, run_id: str) -> bool:
    p = _phase25_dir(runs_dir, run_id)
    req = [
        p / "phase25_summary.json",
        p / "weather_per_hour_metrics.csv",
        p / "weather_day_metrics.json",
        p / "revision_pairs.csv",
        p / "attribution_fit.json",
        p / "correlation_table.csv",
        p / "joint_ops_risk.csv",
    ]
    return all(x.exists() for x in req)


def _load_cache(runs_dir: Path, run_id: str) -> dict[str, Any]:
    p = _phase25_dir(runs_dir, run_id)
    summary = json.loads((p / "phase25_summary.json").read_text(encoding="utf-8"))
    weather_day_metrics = json.loads((p / "weather_day_metrics.json").read_text(encoding="utf-8"))
    attribution_fit = json.loads((p / "attribution_fit.json").read_text(encoding="utf-8"))

    weather_matrices = {}
    weather_dir = p / "weather_matrices"
    if weather_dir.exists():
        for f in weather_dir.glob("matrix_*.csv"):
            var = f.stem.replace("matrix_", "")
            weather_matrices[var] = pd.read_csv(f, index_col=0, parse_dates=True)

    consensus = {}
    for f in p.glob("weather_consensus_*.csv"):
        var = f.stem.replace("weather_consensus_", "")
        consensus[var] = pd.read_csv(f, parse_dates=["timestamp"])

    files = dict(summary.get("files", {}))
    files["phase25_summary_json"] = str(p / "phase25_summary.json")
    return {
        "cached": True,
        "target_run_id": run_id,
        "weather_matrices": weather_matrices,
        "weather_per_hour_metrics_df": pd.read_csv(p / "weather_per_hour_metrics.csv", parse_dates=["timestamp"]),
        "weather_day_metrics": weather_day_metrics,
        "weather_consensus": consensus,
        "revision_pairs_df": pd.read_csv(p / "revision_pairs.csv", parse_dates=["timestamp"]),
        "attribution_fit": attribution_fit,
        "correlation_table_df": pd.read_csv(p / "correlation_table.csv"),
        "joint_ops_risk_df": pd.read_csv(p / "joint_ops_risk.csv", parse_dates=["timestamp"]),
        "files": files,
    }


def get_or_build_phase25(target_date: str, init_hour: str = "00") -> dict[str, Any]:
    cfg = load_config()
    target_day = pd.to_datetime(target_date).normalize()
    run_id = target_day.strftime("%Y%m%d") + init_hour
    runs_dir = Path(cfg["paths"]["runs_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    variables = list(cfg.get("variables_to_plot", ["T2m", "Td2m", "RH2m"]))
    horizon = int(cfg.get("horizon_hours", 90))

    # Ensure base run exists.
    run_for_date(target_day.strftime("%Y-%m-%d"), init_hour)
    # Ensure phase2 exists / get load matrices and metrics.
    p2 = get_or_build_phase2(target_day.strftime("%Y-%m-%d"), init_hour)

    if _cache_exists(runs_dir, run_id):
        return _load_cache(runs_dir, run_id)

    overlap_run_ids = list_overlapping_inits(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        runs_dir=str(runs_dir),
        outputs_dir=str(outputs_dir),
        horizon_hours=horizon,
    )
    weather_matrices = build_weather_matrix(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        run_ids=overlap_run_ids,
        variables=variables,
        runs_dir=str(runs_dir),
    )
    weather_per_hour, weather_day, weather_consensus = compute_weather_revision_metrics(weather_matrices)

    attribution = compute_weather_load_attribution(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        load_matrix_df=p2["forecast_matrix_df"],
        weather_matrices_dict=weather_matrices,
    )
    joint_risk = compute_joint_risk_flags(
        target_date=target_day.strftime("%Y-%m-%d"),
        init_hour=init_hour,
        load_per_hour_metrics_df=p2["per_hour_metrics_df"],
        weather_per_hour_metrics_df=weather_per_hour,
        capacity_mw=float(p2["day_metrics"].get("capacity_mw", 0.0)),
        exceedance_proxy_df=p2["exceedance_proxy_df"],
    )

    files = save_phase25_artifacts(
        run_id_for_target_day=run_id,
        outputs_runs_dir=str(runs_dir),
        weather_matrices=weather_matrices,
        weather_per_hour_metrics_df=weather_per_hour,
        weather_day_metrics_json=weather_day,
        weather_consensus=weather_consensus,
        revision_pairs_df=attribution["revision_pairs"],
        attribution_fit_json=attribution["attribution_fit"],
        correlation_table_df=attribution["correlation_table"],
        joint_ops_risk_df=joint_risk,
        summary_extras={
            "phase2_day_metrics": p2["day_metrics"],
            "overlap_run_ids": overlap_run_ids,
            "realtime_cut_rule": "No actual load used beyond target day 00:00.",
        },
    )

    return {
        "cached": False,
        "target_run_id": run_id,
        "weather_matrices": weather_matrices,
        "weather_per_hour_metrics_df": weather_per_hour,
        "weather_day_metrics": weather_day,
        "weather_consensus": weather_consensus,
        "revision_pairs_df": attribution["revision_pairs"],
        "attribution_fit": attribution["attribution_fit"],
        "correlation_table_df": attribution["correlation_table"],
        "joint_ops_risk_df": joint_risk,
        "files": files,
        "phase2_day_metrics": p2["day_metrics"],
    }
