from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Optional

import numpy as np
import pandas as pd


def _parse_run_id(run_id: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(run_id, format="%Y%m%d%H")
    except Exception as exc:
        raise ValueError(f"Invalid run_id '{run_id}', expected YYYYMMDDHH.") from exc


def _resolve_forecast_path(
    run_id: str,
    runs_dir: Path = Path("outputs/runs"),
    outputs_dir: Path = Path("outputs"),
) -> Path:
    run_csv = runs_dir / run_id / "forecast.csv"
    if run_csv.exists():
        return run_csv
    out_csv = outputs_dir / f"forecast_{run_id}.csv"
    if out_csv.exists():
        return out_csv
    raise FileNotFoundError(
        f"Could not find forecast CSV for run_id={run_id} in {run_csv} or {out_csv}."
    )


def _parse_foretime_series(df: pd.DataFrame, init_time: pd.Timestamp) -> pd.Series:
    """Parse ForeTime formatted as MM/DD HH robustly, including year rollover."""
    if "ForeTime" not in df.columns:
        raise ValueError("ForeTime column missing.")

    raw = df["ForeTime"].astype(str).str.strip()
    mdh = raw.str.extract(r"^(?P<month>\d{1,2})/(?P<day>\d{1,2})\s+(?P<hour>\d{1,2})$")
    if mdh.isna().any().any():
        raise ValueError("ForeTime values are not in expected 'MM/DD HH' format.")

    month = mdh["month"].astype(int).tolist()
    day = mdh["day"].astype(int).tolist()
    hour = mdh["hour"].astype(int).tolist()

    years = []
    year = int(init_time.year)
    prev_key: Optional[tuple[int, int, int]] = None
    for m, d, h in zip(month, day, hour):
        cur_key = (m, d, h)
        if prev_key is not None and cur_key < prev_key:
            # Handles year rollover (e.g. 12/31 23 -> 01/01 00).
            year += 1
        years.append(year)
        prev_key = cur_key

    return pd.to_datetime(
        pd.DataFrame({"year": years, "month": month, "day": day, "hour": hour}),
        errors="coerce",
    )


def load_forecast_series(
    run_id: str,
    runs_dir: str = "outputs/runs",
    outputs_dir: str = "outputs",
) -> pd.DataFrame:
    """Load one run forecast and return target_timestamp + predicted_load + init metadata.

    Output columns:
    - target_timestamp
    - predicted_load
    - init_time
    - run_id
    """
    init_time = _parse_run_id(run_id)
    csv_path = _resolve_forecast_path(
        run_id=run_id,
        runs_dir=Path(runs_dir),
        outputs_dir=Path(outputs_dir),
    )
    df = pd.read_csv(csv_path)

    if {"ForeTime", "Load"}.issubset(df.columns):
        target_ts = _parse_foretime_series(df, init_time)
        pred = pd.to_numeric(df["Load"], errors="coerce")
    elif {"timestamp", "predicted_load"}.issubset(df.columns):
        target_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        pred = pd.to_numeric(df["predicted_load"], errors="coerce")
    else:
        raise ValueError(
            f"Unsupported forecast schema in {csv_path}. "
            "Expected (ForeTime, Load) or (timestamp, predicted_load)."
        )

    out = pd.DataFrame(
        {
            "target_timestamp": target_ts,
            "predicted_load": pred,
            "init_time": init_time,
            "run_id": run_id,
        }
    ).dropna(subset=["target_timestamp", "predicted_load"])
    out = out.sort_values("target_timestamp").reset_index(drop=True)
    return out


def _discover_available_run_ids(
    runs_dir: str = "outputs/runs",
    outputs_dir: str = "outputs",
) -> list[str]:
    run_ids = set()
    runs_base = Path(runs_dir)
    if runs_base.exists():
        for d in runs_base.iterdir():
            if d.is_dir() and len(d.name) == 10 and d.name.isdigit():
                if (d / "forecast.csv").exists():
                    run_ids.add(d.name)

    out_base = Path(outputs_dir)
    if out_base.exists():
        for f in out_base.glob("forecast_*.csv"):
            rid = f.stem.replace("forecast_", "")
            if len(rid) == 10 and rid.isdigit():
                run_ids.add(rid)
    return sorted(run_ids)


def list_overlapping_inits(
    target_date: str,
    init_hour: str,
    runs_dir: str,
    horizon_hours: int,
    outputs_dir: str = "outputs",
) -> list[str]:
    """Return available prior init run_ids whose forecast window overlaps target_date."""
    if init_hour not in ("00", "12"):
        raise ValueError("init_hour must be '00' or '12'.")
    target_day = pd.to_datetime(target_date).normalize()
    day_start = target_day
    day_end = target_day + pd.Timedelta(days=1)

    out: list[str] = []
    for rid in _discover_available_run_ids(runs_dir=runs_dir, outputs_dir=outputs_dir):
        if rid[-2:] != init_hour:
            continue
        init_ts = _parse_run_id(rid)
        if init_ts >= day_start:
            continue  # only prior init runs for revision analysis
        try:
            s = load_forecast_series(rid, runs_dir=runs_dir, outputs_dir=outputs_dir)
        except Exception:
            continue
        if s.empty:
            continue
        # Overlap with day X by timestamp intersection, not by assumed row positions.
        overlaps = ((s["target_timestamp"] >= day_start) & (s["target_timestamp"] < day_end)).any()
        # Guard for malformed/short files: also check expected window constraint.
        max_allowed = init_ts + pd.Timedelta(hours=horizon_hours + 1)
        window_plausible = s["target_timestamp"].max() <= max_allowed
        if overlaps and window_plausible:
            out.append(rid)
    return sorted(out)


def build_forecast_matrix(
    target_date: str,
    init_hour: str,
    available_run_ids: list[str],
    runs_dir: str = "outputs/runs",
    outputs_dir: str = "outputs",
) -> pd.DataFrame:
    target_day = pd.to_datetime(target_date).normalize()
    target_hours = pd.date_range(target_day, target_day + pd.Timedelta(hours=23), freq="h")
    matrix = pd.DataFrame(index=target_hours)
    matrix.index.name = "target_timestamp"

    for rid in sorted(available_run_ids):
        s = load_forecast_series(rid, runs_dir=runs_dir, outputs_dir=outputs_dir)
        s = s[(s["target_timestamp"] >= target_day) & (s["target_timestamp"] < target_day + pd.Timedelta(days=1))]
        if s.empty:
            continue
        col = f"pred_init_{rid}"
        matrix[col] = s.set_index("target_timestamp")["predicted_load"]

    # Keep hours that have at least one available run forecast.
    matrix = matrix.dropna(axis=0, how="all")
    return matrix


def _directionality_score(row: pd.Series) -> float:
    vals = row.dropna().to_numpy(dtype=float)
    if vals.size < 3:
        return np.nan
    diffs = np.diff(vals)
    signs = np.sign(diffs)
    signs = signs[signs != 0]
    if signs.size < 2:
        return np.nan
    return float(np.mean(signs[1:] == signs[:-1]))


def compute_revision_metrics(
    forecast_matrix_df: pd.DataFrame,
    capacity_mw: float,
) -> dict[str, Any]:
    if forecast_matrix_df.empty:
        raise ValueError("Forecast matrix is empty.")

    run_cols = [c for c in forecast_matrix_df.columns if c.startswith("pred_init_")]
    if not run_cols:
        raise ValueError("Forecast matrix has no init prediction columns.")

    # Sort columns by init time to preserve revision order.
    run_cols = sorted(run_cols, key=lambda c: c.replace("pred_init_", ""))
    x = forecast_matrix_df[run_cols].copy()

    per_hour = pd.DataFrame(index=x.index)
    per_hour.index.name = "target_timestamp"
    per_hour["n_available"] = x.notna().sum(axis=1).astype(int)
    per_hour["consensus_median"] = x.median(axis=1, skipna=True)
    q1 = x.quantile(0.25, axis=1, interpolation="linear")
    q3 = x.quantile(0.75, axis=1, interpolation="linear")
    per_hour["consensus_iqr"] = q3 - q1
    per_hour["range"] = x.max(axis=1, skipna=True) - x.min(axis=1, skipna=True)
    per_hour["revision_volatility"] = x.std(axis=1, skipna=True, ddof=0)
    per_hour["directionality_score"] = x.apply(_directionality_score, axis=1)

    # day-1 and day-3 columns from most recent available inits
    sorted_runs = sorted([c.replace("pred_init_", "") for c in run_cols])
    if len(sorted_runs) >= 3:
        day_minus_1 = f"pred_init_{sorted_runs[-1]}"
        day_minus_3 = f"pred_init_{sorted_runs[-3]}"
        per_hour["day_minus_1_minus_day_minus_3"] = x[day_minus_1] - x[day_minus_3]
    else:
        per_hour["day_minus_1_minus_day_minus_3"] = np.nan

    consensus_series = pd.DataFrame(index=x.index)
    consensus_series.index.name = "target_timestamp"
    consensus_series["median"] = per_hour["consensus_median"]
    consensus_series["q1"] = q1
    consensus_series["q3"] = q3
    consensus_series["min"] = x.min(axis=1, skipna=True)
    consensus_series["max"] = x.max(axis=1, skipna=True)

    exceed_mask = x.gt(capacity_mw)
    exceedance_proxy = pd.DataFrame(index=x.index)
    exceedance_proxy.index.name = "target_timestamp"
    exceedance_proxy["exceed_prob_proxy"] = exceed_mask.mean(axis=1, skipna=True)
    exceedance_proxy["median"] = per_hour["consensus_median"]
    exceedance_proxy["capacity_mw"] = float(capacity_mw)

    peak_rows = []
    for col in run_cols:
        s = x[col].dropna()
        if s.empty:
            continue
        peak_ts = s.idxmax()
        peak_rows.append(
            {
                "init_run": col.replace("pred_init_", ""),
                "peak_timestamp": peak_ts,
                "peak_hour_of_day": int(pd.Timestamp(peak_ts).hour),
                "peak_value": float(s.max()),
            }
        )
    peak_table = pd.DataFrame(peak_rows)

    if not peak_table.empty:
        mode_count = int(peak_table["peak_timestamp"].value_counts().max())
        peak_confidence = float(mode_count / len(peak_table))
        peak_time_spread = int(
            (
                peak_table["peak_timestamp"].max() - peak_table["peak_timestamp"].min()
            ).total_seconds()
            / 3600
        )
    else:
        peak_confidence = np.nan
        peak_time_spread = np.nan

    max_range_ts = per_hour["range"].idxmax()
    day_metrics = {
        "n_init_runs": int(len(run_cols)),
        "avg_revision_volatility": float(per_hour["revision_volatility"].mean(skipna=True)),
        "max_range": float(per_hour["range"].max(skipna=True)),
        "max_range_timestamp": pd.Timestamp(max_range_ts).isoformat(),
        "disagreement_index_day": float(per_hour["consensus_iqr"].median(skipna=True)),
        "peak_confidence": float(peak_confidence) if not np.isnan(peak_confidence) else None,
        "peak_time_spread_hours": int(peak_time_spread) if not np.isnan(peak_time_spread) else None,
        "day_exceedance_hours_consensus": int((per_hour["consensus_median"] > capacity_mw).sum()),
        "day_exceedance_hours_majority": int((exceedance_proxy["exceed_prob_proxy"] >= 0.5).sum()),
        "capacity_mw": float(capacity_mw),
    }

    return {
        "per_hour_metrics": per_hour.reset_index(),
        "day_metrics": day_metrics,
        "consensus_series": consensus_series.reset_index(),
        "exceedance_proxy": exceedance_proxy.reset_index(),
        "peak_table": peak_table,
    }


def compute_value_of_information(
    target_date: str,
    init_hour: str,
    forecast_matrix_df: pd.DataFrame,
    history_df: pd.DataFrame,
    realtime_cut_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Compute error vs forecast-age bins using only realized actuals (< realtime_cut_ts)."""
    if init_hour not in ("00", "12"):
        raise ValueError("init_hour must be '00' or '12'.")
    run_cols = [c for c in forecast_matrix_df.columns if c.startswith("pred_init_")]
    if not run_cols:
        return pd.DataFrame(columns=["age_bin", "n_points", "mae_mw", "bias_mw"])

    if "timestamp" not in history_df.columns:
        raise ValueError("history_df must contain 'timestamp'.")
    load_candidates = [c for c in ["load_MW", "Load", "load", "value"] if c in history_df.columns]
    if not load_candidates:
        raise ValueError("history_df does not contain a recognized load column.")
    load_col = load_candidates[0]

    hist = history_df.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist[load_col] = pd.to_numeric(hist[load_col], errors="coerce")
    actuals = hist[["timestamp", load_col]].dropna().rename(columns={load_col: "actual_load"})
    actuals = actuals[actuals["timestamp"] < pd.to_datetime(realtime_cut_ts)]
    actuals = actuals.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    all_eval_rows = []
    for col in run_cols:
        rid = col.replace("pred_init_", "")
        fs = load_forecast_series(rid)
        fs = fs[fs["target_timestamp"] < pd.to_datetime(realtime_cut_ts)]
        if fs.empty:
            continue
        merged = fs.merge(actuals, left_on="target_timestamp", right_on="timestamp", how="inner")
        if merged.empty:
            continue
        merged["forecast_age_hours"] = (
            (merged["target_timestamp"] - merged["init_time"]).dt.total_seconds() / 3600.0
        )
        merged["error"] = merged["predicted_load"] - merged["actual_load"]
        all_eval_rows.append(merged[["forecast_age_hours", "error"]])

    if not all_eval_rows:
        return pd.DataFrame(columns=["age_bin", "n_points", "mae_mw", "bias_mw"])

    eval_df = pd.concat(all_eval_rows, ignore_index=True)
    eval_df = eval_df[eval_df["forecast_age_hours"].between(0, 96, inclusive="left")]
    bins = [0, 24, 48, 72, 96]
    labels = ["0-24", "24-48", "48-72", "72-90"]
    eval_df["age_bin"] = pd.cut(eval_df["forecast_age_hours"], bins=bins, labels=labels, right=False)

    voi = (
        eval_df.groupby("age_bin", observed=False)
        .agg(
            n_points=("error", "count"),
            mae_mw=("error", lambda s: float(np.abs(s).mean()) if len(s) else np.nan),
            bias_mw=("error", lambda s: float(s.mean()) if len(s) else np.nan),
        )
        .reset_index()
    )
    return voi


def _jsonify_value(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _jsonify_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonify_value(x) for x in v]
    return v


def save_phase2_artifacts(
    run_id_for_target_day: str,
    outputs_runs_dir: str,
    target_date: str,
    init_hour: str,
    forecast_matrix_df: pd.DataFrame,
    per_hour_metrics_df: pd.DataFrame,
    day_metrics_json: dict[str, Any],
    voi_df: pd.DataFrame,
    peak_table_df: pd.DataFrame,
    consensus_series_df: pd.DataFrame,
    exceedance_proxy_df: pd.DataFrame,
) -> dict[str, str]:
    base = Path(outputs_runs_dir) / run_id_for_target_day / "phase2"
    base.mkdir(parents=True, exist_ok=True)

    p_forecast_matrix = base / "forecast_matrix.csv"
    p_per_hour = base / "per_hour_metrics.csv"
    p_day = base / "day_metrics.json"
    p_peak = base / "peak_by_init.csv"
    p_consensus = base / "consensus_series.csv"
    p_exceed = base / "exceedance_proxy.csv"
    p_voi = base / "voi.csv"
    p_voi_json = base / "voi.json"
    p_summary = base / "phase2_summary.json"

    forecast_matrix_df.to_csv(p_forecast_matrix)
    per_hour_metrics_df.to_csv(p_per_hour, index=False)
    peak_table_df.to_csv(p_peak, index=False)
    consensus_series_df.to_csv(p_consensus, index=False)
    exceedance_proxy_df.to_csv(p_exceed, index=False)
    if voi_df is not None and not voi_df.empty:
        voi_df.to_csv(p_voi, index=False)
    else:
        pd.DataFrame(columns=["age_bin", "n_points", "mae_mw", "bias_mw"]).to_csv(p_voi, index=False)

    with p_day.open("w", encoding="utf-8") as f:
        json.dump(_jsonify_value(day_metrics_json), f, indent=2)
    voi_metrics = {
        "bins_with_data": int((voi_df["n_points"] > 0).sum()) if voi_df is not None and not voi_df.empty else 0,
        "total_points": int(voi_df["n_points"].sum()) if voi_df is not None and not voi_df.empty else 0,
        "realtime_cut_rule": "actuals used only for timestamps < target_day 00:00",
    }
    with p_voi_json.open("w", encoding="utf-8") as f:
        json.dump(_jsonify_value(voi_metrics), f, indent=2)

    summary = {
        "target_date": target_date,
        "init_hour": init_hour,
        "target_run_id": run_id_for_target_day,
        "day_metrics": day_metrics_json,
        "files": {
            "forecast_matrix_csv": str(p_forecast_matrix),
            "per_hour_metrics_csv": str(p_per_hour),
            "day_metrics_json": str(p_day),
            "peak_by_init_csv": str(p_peak),
            "consensus_series_csv": str(p_consensus),
            "exceedance_proxy_csv": str(p_exceed),
            "voi_csv": str(p_voi),
            "voi_json": str(p_voi_json),
        },
    }
    with p_summary.open("w", encoding="utf-8") as f:
        json.dump(_jsonify_value(summary), f, indent=2)

    return summary["files"] | {"phase2_summary_json": str(p_summary)}
