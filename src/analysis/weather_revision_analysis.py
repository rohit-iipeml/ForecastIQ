from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.analysis.revision_analysis import _parse_run_id
from src.phase1_backend import run_for_date


def _resolve_weather_csv(run_id: str, runs_dir: str = "outputs/runs", regenerate_if_missing: bool = True) -> Path:
    p = Path(runs_dir) / run_id / "weather_window.csv"
    if regenerate_if_missing and not p.exists():
        init_ts = _parse_run_id(run_id)
        init_hh = init_ts.strftime("%H")
        # Cache-first: this will reuse existing forecast/output where available.
        run_for_date(init_ts.strftime("%Y-%m-%d"), init_hh)
    if not p.exists():
        raise FileNotFoundError(f"Missing weather_window.csv for run_id={run_id}: {p}")
    return p


def _detect_time_col(df: pd.DataFrame) -> str:
    for c in ["timestamp", "ForeTime", "time"]:
        if c in df.columns:
            return c
    raise ValueError("No recognized timestamp column in weather data (timestamp/ForeTime/time).")


def _parse_time_col(series: pd.Series, run_id: str) -> pd.Series:
    init_time = _parse_run_id(run_id)
    s = series.astype(str).str.strip()

    # Try direct datetime parse first.
    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().mean() > 0.9:
        return dt

    # Fallback for MM/DD HH format.
    mdh = s.str.extract(r"^(?P<month>\d{1,2})/(?P<day>\d{1,2})\s+(?P<hour>\d{1,2})$")
    if mdh.notna().all().all():
        month = mdh["month"].astype(int).tolist()
        day = mdh["day"].astype(int).tolist()
        hour = mdh["hour"].astype(int).tolist()
        years = []
        year = int(init_time.year)
        prev_key: Optional[tuple[int, int, int]] = None
        for m, d, h in zip(month, day, hour):
            key = (m, d, h)
            if prev_key is not None and key < prev_key:
                year += 1
            years.append(year)
            prev_key = key
        return pd.to_datetime(pd.DataFrame({"year": years, "month": month, "day": day, "hour": hour}), errors="coerce")

    return dt


def load_weather_series(
    run_id: str,
    runs_dir: str = "outputs/runs",
    regenerate_if_missing: bool = True,
) -> pd.DataFrame:
    """Load weather_window for one run. Output has target_timestamp + weather variables."""
    p = _resolve_weather_csv(run_id, runs_dir=runs_dir, regenerate_if_missing=regenerate_if_missing)
    df = pd.read_csv(p)
    tcol = _detect_time_col(df)
    ts = _parse_time_col(df[tcol], run_id)

    out = df.copy()
    out["target_timestamp"] = ts
    out = out.dropna(subset=["target_timestamp"])
    drop_cols = [c for c in [tcol, "init_time", "lead_hour", "load_MW"] if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric weather columns plus target timestamp.
    keep = ["target_timestamp"]
    for c in out.columns:
        if c == "target_timestamp":
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].notna().any():
            keep.append(c)
    out = out[keep].sort_values("target_timestamp").drop_duplicates("target_timestamp", keep="last")
    return out.reset_index(drop=True)


def build_weather_matrix_t2m(
    target_date: str,
    run_ids: list[str],
    runs_dir: str = "outputs/runs",
) -> pd.DataFrame:
    """Minimal Step-1 matrix builder for T2m only."""
    day = pd.to_datetime(target_date).normalize()
    idx = pd.date_range(day, day + pd.Timedelta(hours=23), freq="h")
    m = pd.DataFrame(index=idx)
    m.index.name = "target_timestamp"

    for rid in sorted(run_ids):
        try:
            s = load_weather_series(rid, runs_dir=runs_dir)
        except Exception:
            continue
        if "T2m" not in s.columns:
            continue
        s = s[(s["target_timestamp"] >= day) & (s["target_timestamp"] < day + pd.Timedelta(days=1))]
        if s.empty:
            continue
        m[f"T2m_init_{rid}"] = s.set_index("target_timestamp")["T2m"]

    return m.dropna(axis=0, how="all")


def build_weather_matrix(
    target_date: str,
    init_hour: str,
    run_ids: list[str],
    variables: list[str],
    runs_dir: str = "outputs/runs",
) -> dict[str, pd.DataFrame]:
    day = pd.to_datetime(target_date).normalize()
    idx = pd.date_range(day, day + pd.Timedelta(hours=23), freq="h")
    out: dict[str, pd.DataFrame] = {}

    loaded: dict[str, pd.DataFrame] = {}
    for rid in sorted(run_ids):
        if rid[-2:] != init_hour:
            continue
        try:
            loaded[rid] = load_weather_series(rid, runs_dir=runs_dir, regenerate_if_missing=True)
        except Exception:
            continue

    for var in variables:
        m = pd.DataFrame(index=idx)
        m.index.name = "target_timestamp"
        for rid, s in loaded.items():
            if var not in s.columns:
                continue
            w = s[(s["target_timestamp"] >= day) & (s["target_timestamp"] < day + pd.Timedelta(days=1))]
            if w.empty:
                continue
            m[f"{var}_init_{rid}"] = w.set_index("target_timestamp")[var]
        m = m.dropna(axis=0, how="all")
        if not m.empty:
            out[var] = m
    return out


def compute_weather_revision_metrics(
    weather_matrices_dict: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    rows = []
    day_metrics: dict[str, Any] = {"variables": {}}
    consensus_dict: dict[str, pd.DataFrame] = {}

    for var, matrix in weather_matrices_dict.items():
        cols = sorted(matrix.columns)
        x = matrix[cols].copy()
        per = pd.DataFrame(index=x.index)
        per.index.name = "timestamp"
        per["variable"] = var
        per["n_available"] = x.notna().sum(axis=1).astype(int)
        per["median"] = x.median(axis=1, skipna=True)
        q1 = x.quantile(0.25, axis=1, interpolation="linear")
        q3 = x.quantile(0.75, axis=1, interpolation="linear")
        per["q1"] = q1
        per["q3"] = q3
        per["iqr"] = q3 - q1
        per["min"] = x.min(axis=1, skipna=True)
        per["max"] = x.max(axis=1, skipna=True)
        per["range"] = per["max"] - per["min"]
        per["std"] = x.std(axis=1, skipna=True, ddof=0)
        rows.append(per.reset_index())

        max_range_ts = per["range"].idxmax() if not per.empty else None
        day_metrics["variables"][var] = {
            "n_init_runs": int(len(cols)),
            "variable_disagreement_index_iqr": float(per["iqr"].median(skipna=True)) if not per.empty else None,
            "variable_disagreement_index_range": float(per["range"].median(skipna=True)) if not per.empty else None,
            "avg_std": float(per["std"].mean(skipna=True)) if not per.empty else None,
            "max_range": float(per["range"].max(skipna=True)) if not per.empty else None,
            "max_range_timestamp": pd.Timestamp(max_range_ts).isoformat() if max_range_ts is not None else None,
        }
        consensus_dict[var] = per.reset_index()[["timestamp", "median", "q1", "q3", "min", "max"]]

    if rows:
        per_hour_long = pd.concat(rows, ignore_index=True)
    else:
        per_hour_long = pd.DataFrame(
            columns=["timestamp", "variable", "n_available", "median", "q1", "q3", "iqr", "min", "max", "range", "std"]
        )
    day_metrics["n_variables"] = int(len(day_metrics["variables"]))
    return per_hour_long, day_metrics, consensus_dict


def _choose_load_revision_pair(load_matrix_df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    cols = sorted([c for c in load_matrix_df.columns if c.startswith("pred_init_")], key=lambda c: c.replace("pred_init_", ""))
    if len(cols) < 2:
        return None, None
    closest = cols[-1]
    older = cols[-3] if len(cols) >= 3 else cols[0]
    return closest, older


def compute_weather_load_attribution(
    target_date: str,
    init_hour: str,
    load_matrix_df: pd.DataFrame,
    weather_matrices_dict: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compute ΔLoad and ΔWeather attribution using closest vs older init runs."""
    if init_hour not in ("00", "12"):
        raise ValueError("init_hour must be '00' or '12'.")
    closest_col, older_col = _choose_load_revision_pair(load_matrix_df)
    if not closest_col or not older_col:
        empty_pairs = pd.DataFrame(columns=["timestamp", "init_closest", "init_older", "delta_load"])
        fit = {
            "model": "linear_regression",
            "vars_used": [],
            "intercept": None,
            "coefficients": {},
            "standardized_coefficients": {},
            "r2": None,
            "n_samples": 0,
            "note": "Insufficient load init runs for revision attribution.",
        }
        corr = pd.DataFrame(columns=["variable", "corr_with_delta_load"])
        return {"revision_pairs": empty_pairs, "attribution_fit": fit, "correlation_table": corr}

    closest_run = closest_col.replace("pred_init_", "")
    older_run = older_col.replace("pred_init_", "")

    pairs = pd.DataFrame(index=load_matrix_df.index.copy())
    pairs.index.name = "timestamp"
    pairs["init_closest"] = closest_run
    pairs["init_older"] = older_run
    pairs["delta_load"] = load_matrix_df[closest_col] - load_matrix_df[older_col]

    feature_cols = []
    for var, m in weather_matrices_dict.items():
        c1 = f"{var}_init_{closest_run}"
        c0 = f"{var}_init_{older_run}"
        if c1 in m.columns and c0 in m.columns:
            col = f"delta_{var}"
            aligned = m.reindex(pairs.index)
            pairs[col] = aligned[c1] - aligned[c0]
            feature_cols.append(col)

    pairs = pairs.reset_index()
    model_df = pairs.dropna(subset=["delta_load"] + feature_cols) if feature_cols else pairs.iloc[0:0]

    if feature_cols and len(model_df) >= 3:
        X = model_df[feature_cols].to_numpy(dtype=float)
        y = model_df["delta_load"].to_numpy(dtype=float)
        reg = LinearRegression()
        reg.fit(X, y)
        r2 = float(reg.score(X, y))
        coef = {v: float(c) for v, c in zip(feature_cols, reg.coef_)}
        x_std = model_df[feature_cols].std(ddof=0).replace(0, np.nan)
        y_std = float(model_df["delta_load"].std(ddof=0))
        std_coef = {}
        for v in feature_cols:
            if pd.isna(x_std[v]) or y_std == 0:
                std_coef[v] = None
            else:
                std_coef[v] = float(coef[v] * (x_std[v] / y_std))
        fit = {
            "model": "linear_regression",
            "vars_used": feature_cols,
            "intercept": float(reg.intercept_),
            "coefficients": coef,
            "standardized_coefficients": std_coef,
            "r2": r2,
            "n_samples": int(len(model_df)),
            "note": "Revision attribution / explained variance only (not causality).",
        }
    else:
        fit = {
            "model": "linear_regression",
            "vars_used": feature_cols,
            "intercept": None,
            "coefficients": {},
            "standardized_coefficients": {},
            "r2": None,
            "n_samples": int(len(model_df)),
            "note": "Insufficient samples or variables for fit.",
        }

    corr_rows = []
    for v in feature_cols:
        valid = pairs[["delta_load", v]].dropna()
        corr = float(valid["delta_load"].corr(valid[v])) if len(valid) >= 2 else np.nan
        corr_rows.append({"variable": v.replace("delta_", ""), "corr_with_delta_load": corr})
    corr_df = pd.DataFrame(corr_rows)

    return {"revision_pairs": pairs, "attribution_fit": fit, "correlation_table": corr_df}


def _jsonify(v: Any) -> Any:
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonify(x) for x in v]
    return v


def save_phase25_artifacts(
    run_id_for_target_day: str,
    outputs_runs_dir: str,
    weather_matrices: dict[str, pd.DataFrame],
    weather_per_hour_metrics_df: pd.DataFrame,
    weather_day_metrics_json: dict[str, Any],
    weather_consensus: dict[str, pd.DataFrame],
    revision_pairs_df: pd.DataFrame,
    attribution_fit_json: dict[str, Any],
    correlation_table_df: pd.DataFrame,
    joint_ops_risk_df: pd.DataFrame,
    summary_extras: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    base = Path(outputs_runs_dir) / run_id_for_target_day / "phase25"
    weather_dir = base / "weather_matrices"
    base.mkdir(parents=True, exist_ok=True)
    weather_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}
    for var, m in weather_matrices.items():
        p = weather_dir / f"matrix_{var}.csv"
        m.to_csv(p)
        files[f"weather_matrix_{var}_csv"] = str(p)

    p_weather_per_hour = base / "weather_per_hour_metrics.csv"
    p_weather_day = base / "weather_day_metrics.json"
    p_revision_pairs = base / "revision_pairs.csv"
    p_fit = base / "attribution_fit.json"
    p_corr = base / "correlation_table.csv"
    p_joint = base / "joint_ops_risk.csv"
    p_summary = base / "phase25_summary.json"

    weather_per_hour_metrics_df.to_csv(p_weather_per_hour, index=False)
    with p_weather_day.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(weather_day_metrics_json), f, indent=2)

    for var, cdf in weather_consensus.items():
        p = base / f"weather_consensus_{var}.csv"
        cdf.to_csv(p, index=False)
        files[f"weather_consensus_{var}_csv"] = str(p)

    revision_pairs_df.to_csv(p_revision_pairs, index=False)
    with p_fit.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(attribution_fit_json), f, indent=2)
    correlation_table_df.to_csv(p_corr, index=False)
    joint_ops_risk_df.to_csv(p_joint, index=False)

    files.update(
        {
            "weather_per_hour_metrics_csv": str(p_weather_per_hour),
            "weather_day_metrics_json": str(p_weather_day),
            "revision_pairs_csv": str(p_revision_pairs),
            "attribution_fit_json": str(p_fit),
            "correlation_table_csv": str(p_corr),
            "joint_ops_risk_csv": str(p_joint),
        }
    )
    summary = {"target_run_id": run_id_for_target_day, "files": files, "attribution_fit": attribution_fit_json}
    if summary_extras:
        summary.update(summary_extras)
    with p_summary.open("w", encoding="utf-8") as f:
        json.dump(_jsonify(summary), f, indent=2)
    files["phase25_summary_json"] = str(p_summary)
    return files


def compute_joint_risk_flags(
    target_date: str,
    init_hour: str,
    load_per_hour_metrics_df: pd.DataFrame,
    weather_per_hour_metrics_df: pd.DataFrame,
    capacity_mw: float,
    exceedance_proxy_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build per-hour joint ops risk flags for target day."""
    if init_hour not in ("00", "12"):
        raise ValueError("init_hour must be '00' or '12'.")

    ldf = load_per_hour_metrics_df.copy()
    if ldf.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "median_load",
                "load_iqr",
                "load_range",
                "exceed_prob_proxy",
                "max_weather_iqr",
                "max_weather_range",
                "HighLoadDisagreement",
                "HighWeatherDisagreement",
                "HighOpsRisk",
                "ops_risk_score",
            ]
        )

    ldf["timestamp"] = pd.to_datetime(ldf["target_timestamp"], errors="coerce")
    ldf = ldf.dropna(subset=["timestamp"])
    ldf = ldf.rename(columns={"consensus_median": "median_load", "consensus_iqr": "load_iqr", "range": "load_range"})

    wdf = weather_per_hour_metrics_df.copy()
    wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], errors="coerce")
    wdf = wdf.dropna(subset=["timestamp"])
    w_agg = (
        wdf.groupby("timestamp")
        .agg(max_weather_iqr=("iqr", "max"), max_weather_range=("range", "max"))
        .reset_index()
    )

    out = ldf.merge(w_agg, on="timestamp", how="left")
    if exceedance_proxy_df is not None and not exceedance_proxy_df.empty:
        ex = exceedance_proxy_df.copy()
        ex["timestamp"] = pd.to_datetime(ex["target_timestamp"], errors="coerce")
        out = out.merge(ex[["timestamp", "exceed_prob_proxy"]], on="timestamp", how="left")
    else:
        out["exceed_prob_proxy"] = np.nan

    load_thr = float(out["load_range"].quantile(0.75))
    weather_thr = float(out["max_weather_range"].quantile(0.75)) if out["max_weather_range"].notna().any() else np.nan

    out["HighLoadDisagreement"] = out["load_range"] >= load_thr
    if np.isnan(weather_thr):
        out["HighWeatherDisagreement"] = False
    else:
        out["HighWeatherDisagreement"] = out["max_weather_range"] >= weather_thr

    out["exceed_prob_proxy"] = out["exceed_prob_proxy"].fillna((out["median_load"] > capacity_mw).astype(float))
    out["HighOpsRisk"] = (
        (out["HighLoadDisagreement"] & out["HighWeatherDisagreement"])
        | ((out["exceed_prob_proxy"] >= 0.5) & out["HighLoadDisagreement"])
    )
    out["ops_risk_score"] = (
        out["load_range"].rank(pct=True).fillna(0) * 0.45
        + out["max_weather_range"].rank(pct=True).fillna(0) * 0.35
        + out["exceed_prob_proxy"].fillna(0) * 0.20
    )

    keep_cols = [
        "timestamp",
        "median_load",
        "load_iqr",
        "load_range",
        "exceed_prob_proxy",
        "max_weather_iqr",
        "max_weather_range",
        "HighLoadDisagreement",
        "HighWeatherDisagreement",
        "HighOpsRisk",
        "ops_risk_score",
    ]
    return out[keep_cols].sort_values("timestamp").reset_index(drop=True)
