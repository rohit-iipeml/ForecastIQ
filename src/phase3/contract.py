from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RUNS_ROOT = Path("outputs/runs")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_json(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        warnings.append(f"Missing file: {path}")
        return None
    try:
        return _read_json(path)
    except Exception as exc:
        warnings.append(f"Failed to parse JSON {path}: {exc}")
        return None


def _safe_read_csv(path: Path, warnings: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        warnings.append(f"Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.append(f"Failed to parse CSV {path}: {exc}")
        return None


def _require(obj: dict[str, Any], key: str, typ: type, where: str) -> None:
    if key not in obj:
        raise ValueError(f"Missing required field '{key}' in {where}.")
    if not isinstance(obj[key], typ):
        raise ValueError(f"Invalid type for '{key}' in {where}: expected {typ.__name__}.")


def validate_phase3_input(payload: dict[str, Any]) -> None:
    _require(payload, "run_id", str, "phase3_input")
    _require(payload, "init_time", str, "phase3_input")
    _require(payload, "horizon_hours", int, "phase3_input")
    _require(payload, "phase1", dict, "phase3_input")
    _require(payload, "phase2", dict, "phase3_input")
    _require(payload, "phase25", dict, "phase3_input")
    _require(payload, "metadata", dict, "phase3_input")

    p1 = payload["phase1"]
    for k in [
        "avg_load_mw",
        "max_load_mw",
        "max_load_time",
        "min_load_mw",
        "min_load_time",
        "capacity_mw",
        "hours_above_capacity",
        "max_exceedance_mw",
    ]:
        if k not in p1:
            raise ValueError(f"Missing phase1.{k}")
    if "peak" not in p1 or not isinstance(p1["peak"], dict):
        raise ValueError("Missing phase1.peak")

    p2 = payload["phase2"]
    for k in ["disagreement_index", "peak_confidence", "peak_time_spread_hours", "avg_revision_volatility"]:
        if k not in p2:
            raise ValueError(f"Missing phase2.{k}")

    meta = payload["metadata"]
    if "source_files" not in meta or not isinstance(meta["source_files"], dict):
        raise ValueError("Missing metadata.source_files")


def load_phase3_input(run_id: str) -> dict[str, Any] | None:
    p = RUNS_ROOT / run_id / "phase3" / "phase3_input.json"
    if not p.exists():
        return None
    return _read_json(p)


def build_phase3_input(run_id: str) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    phase3_dir = run_dir / "phase3"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    src_forecast = run_dir / "forecast.json"
    src_metrics = run_dir / "metrics.json"
    src_p2_summary = run_dir / "phase2" / "phase2_summary.json"
    src_p25_summary = run_dir / "phase25" / "phase25_summary.json"

    forecast = _safe_read_json(src_forecast, warnings) or {}
    metrics = _safe_read_json(src_metrics, warnings) or {}
    p2_summary = _safe_read_json(src_p2_summary, warnings) or {}
    p25_summary = _safe_read_json(src_p25_summary, warnings) or {}

    p2_files = (p2_summary or {}).get("files", {}) or {}
    p25_files = (p25_summary or {}).get("files", {}) or {}

    per_hour = _safe_read_csv(Path(p2_files.get("per_hour_metrics_csv", run_dir / "phase2" / "per_hour_metrics.csv")), warnings)
    exceed_proxy = _safe_read_csv(Path(p2_files.get("exceedance_proxy_csv", run_dir / "phase2" / "exceedance_proxy.csv")), warnings)
    voi = _safe_read_csv(Path(p2_files.get("voi_csv", run_dir / "phase2" / "voi.csv")), warnings)
    voi_meta = _safe_read_json(Path(p2_files.get("voi_json", run_dir / "phase2" / "voi.json")), warnings) or {}
    peak_by_init = _safe_read_csv(Path(p2_files.get("peak_by_init_csv", run_dir / "phase2" / "peak_by_init.csv")), warnings)

    joint_risk = _safe_read_csv(Path(p25_files.get("joint_ops_risk_csv", run_dir / "phase25" / "joint_ops_risk.csv")), warnings)
    attribution_fit = _safe_read_json(Path(p25_files.get("attribution_fit_json", run_dir / "phase25" / "attribution_fit.json")), warnings) or {}
    correlation = _safe_read_csv(Path(p25_files.get("correlation_table_csv", run_dir / "phase25" / "correlation_table.csv")), warnings)
    weather_day = _safe_read_json(Path(p25_files.get("weather_day_metrics_json", run_dir / "phase25" / "weather_day_metrics.json")), warnings) or {}

    forecast_points = forecast.get("forecast", []) or []
    if forecast_points:
        peak_pt = max(forecast_points, key=lambda x: float(x.get("predicted_load", float("-inf"))))
        peak = {"time": peak_pt.get("timestamp"), "value_mw": float(peak_pt.get("predicted_load", 0.0))}
    else:
        peak = {"time": metrics.get("max_predicted_load_ts"), "value_mw": float(metrics.get("max_predicted_load_mw", 0.0))}

    top_unstable = []
    if per_hour is not None and not per_hour.empty:
        temp = per_hour.copy()
        sort_col = "range" if "range" in temp.columns else ("revision_volatility" if "revision_volatility" in temp.columns else None)
        if sort_col:
            temp = temp.sort_values(sort_col, ascending=False).head(10)
            for _, r in temp.iterrows():
                top_unstable.append(
                    {
                        "time": str(r.get("target_timestamp")),
                        "range_mw": float(r["range"]) if "range" in r and pd.notna(r["range"]) else None,
                        "iqr_mw": float(r["consensus_iqr"]) if "consensus_iqr" in r and pd.notna(r["consensus_iqr"]) else None,
                        "std_mw": float(r["revision_volatility"]) if "revision_volatility" in r and pd.notna(r["revision_volatility"]) else None,
                    }
                )

    top_risky = []
    if exceed_proxy is not None and not exceed_proxy.empty:
        temp = exceed_proxy.sort_values(["exceed_prob_proxy", "median"], ascending=[False, False]).head(10)
        for _, r in temp.iterrows():
            top_risky.append(
                {
                    "time": str(r.get("target_timestamp")),
                    "median_load_mw": float(r["median"]) if "median" in r and pd.notna(r["median"]) else None,
                    "exceed_prob_proxy": float(r["exceed_prob_proxy"]) if "exceed_prob_proxy" in r and pd.notna(r["exceed_prob_proxy"]) else None,
                }
            )

    voi_bins = []
    if voi is not None and not voi.empty:
        for _, r in voi.iterrows():
            voi_bins.append(
                {
                    "age_bin": str(r.get("age_bin")),
                    "mae": float(r["mae_mw"]) if "mae_mw" in r and pd.notna(r["mae_mw"]) else None,
                    "bias": float(r["bias_mw"]) if "bias_mw" in r and pd.notna(r["bias_mw"]) else None,
                }
            )

    weather_disagreement = {}
    for var, vals in (weather_day.get("variables", {}) or {}).items():
        weather_disagreement[var] = {
            "disagreement_index": vals.get("variable_disagreement_index_iqr"),
            "max_range": vals.get("max_range"),
            "max_range_time": vals.get("max_range_timestamp"),
        }

    top_driver_var = None
    top_driver_corr = None
    if correlation is not None and not correlation.empty and "corr_with_delta_load" in correlation.columns:
        c = correlation.copy().dropna(subset=["corr_with_delta_load"])
        if not c.empty:
            c["_abs_corr"] = c["corr_with_delta_load"].abs()
            best = c.sort_values("_abs_corr", ascending=False).iloc[0]
            top_driver_var = str(best.get("variable"))
            top_driver_corr = float(best.get("corr_with_delta_load"))

    high_ops = []
    if joint_risk is not None and not joint_risk.empty:
        temp = joint_risk[joint_risk.get("HighOpsRisk", False) == True].copy()
        if "ops_risk_score" in temp.columns:
            temp = temp.sort_values("ops_risk_score", ascending=False)
        temp = temp.head(10)
        for _, r in temp.iterrows():
            flags = []
            if bool(r.get("HighLoadDisagreement", False)):
                flags.append("HighLoadDisagreement")
            if bool(r.get("HighWeatherDisagreement", False)):
                flags.append("HighWeatherDisagreement")
            if float(r.get("exceed_prob_proxy", 0.0) or 0.0) >= 0.5:
                flags.append("MajorityCapacityExceedance")
            high_ops.append(
                {
                    "time": str(r.get("timestamp")),
                    "reason_flags": flags,
                    "median_load_mw": float(r["median_load"]) if "median_load" in r and pd.notna(r["median_load"]) else None,
                    "load_range_mw": float(r["load_range"]) if "load_range" in r and pd.notna(r["load_range"]) else None,
                    "T2m_range": float(r["max_weather_range"]) if "max_weather_range" in r and pd.notna(r["max_weather_range"]) else None,
                }
            )

    top_load_hours = []
    if forecast_points:
        top_pts = sorted(
            forecast_points,
            key=lambda x: float(x.get("predicted_load", float("-inf"))),
            reverse=True,
        )[:10]
        for pt in top_pts:
            top_load_hours.append(
                {
                    "time": pt.get("timestamp"),
                    "predicted_load_mw": float(pt.get("predicted_load", 0.0)),
                }
            )

    phase3_input: dict[str, Any] = {
        "run_id": run_id,
        "init_time": forecast.get("init_time"),
        "horizon_hours": int(forecast.get("horizon_hours", 90)),
        "capacity_definition": "p90_of_history_before_init_time",
        "phase1": {
            "peak": peak,
            "avg_load_mw": float(metrics.get("avg_predicted_load_mw", 0.0)),
            "max_load_mw": float(metrics.get("max_predicted_load_mw", 0.0)),
            "max_load_time": metrics.get("max_predicted_load_ts"),
            "min_load_mw": float(metrics.get("min_predicted_load_mw", 0.0)),
            "min_load_time": metrics.get("min_predicted_load_ts"),
            "capacity_mw": float(metrics.get("capacity_mw", 0.0)),
            "hours_above_capacity": int(metrics.get("hours_above_capacity", 0)),
            "max_exceedance_mw": float(metrics.get("max_exceedance_mw", 0.0)),
            "top_load_hours": top_load_hours,
        },
        "phase2": {
            "disagreement_index": (p2_summary.get("day_metrics") or {}).get("disagreement_index_day"),
            "peak_confidence": (p2_summary.get("day_metrics") or {}).get("peak_confidence"),
            "peak_time_spread_hours": (p2_summary.get("day_metrics") or {}).get("peak_time_spread_hours"),
            "avg_revision_volatility": (p2_summary.get("day_metrics") or {}).get("avg_revision_volatility"),
            "max_range_mw": (p2_summary.get("day_metrics") or {}).get("max_range"),
            "max_range_time": (p2_summary.get("day_metrics") or {}).get("max_range_timestamp"),
            "exceedance_hours_consensus": (p2_summary.get("day_metrics") or {}).get("day_exceedance_hours_consensus"),
            "exceedance_hours_majority": (p2_summary.get("day_metrics") or {}).get("day_exceedance_hours_majority"),
            "top_unstable_hours": top_unstable,
            "top_risky_hours": top_risky,
            "voi_summary": {
                "bins": voi_bins,
                "note": voi_meta.get("realtime_cut_rule", "computed using real-time cut < X 00:00"),
            },
        },
        "phase25": {
            "weather_disagreement": weather_disagreement,
            "attribution": {
                "r2": attribution_fit.get("r2"),
                "vars_used": attribution_fit.get("vars_used", []),
                "top_driver_var": top_driver_var,
                "top_driver_corr": top_driver_corr,
            },
            "high_ops_risk_hours": high_ops,
        },
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_files": {
                "forecast_json": str(src_forecast),
                "metrics_json": str(src_metrics),
                "phase2_summary": str(src_p2_summary),
                "phase25_summary": str(src_p25_summary),
            },
            "warnings": warnings,
            "optional_sources": {
                "phase2_peak_by_init_present": peak_by_init is not None,
                "phase2_voi_present": voi is not None,
                "phase25_joint_ops_present": joint_risk is not None,
            },
        },
    }

    validate_phase3_input(phase3_input)
    out_path = phase3_dir / "phase3_input.json"
    out_path.write_text(json.dumps(phase3_input, indent=2), encoding="utf-8")
    return phase3_input
