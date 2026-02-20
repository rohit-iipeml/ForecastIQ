from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from src.gru_universal import ingest_load_csv, ingest_weather_mat, run_forecast


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class RuntimePaths:
    history_csv: Path
    weather_dir: Path
    load_daily_dir: Path
    outputs_dir: Path
    runs_dir: Path


def _resolve_path(path_str: str) -> Path:
    return (REPO_ROOT / path_str).resolve()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid config format: expected a mapping at root.")
    return cfg


def _detect_load_col(df: pd.DataFrame) -> str:
    candidates = ["load_MW", "Load", "load", "value"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find a load column. Expected one of: load_MW, Load, load, value."
    )


def _validate_date(date_str: str, cfg: Dict[str, Any]) -> pd.Timestamp:
    try:
        target_day = pd.to_datetime(date_str).normalize()
    except Exception as exc:
        raise ValueError(f"Invalid date: {date_str}") from exc
    min_day = pd.to_datetime(cfg["allowed_date_min"]).normalize()
    max_day = pd.to_datetime(cfg["allowed_date_max"]).normalize()
    if target_day < min_day or target_day > max_day:
        raise ValueError(
            f"Date {target_day.date()} is out of allowed range "
            f"{min_day.date()} to {max_day.date()}."
        )
    return target_day


def _build_paths(cfg: Dict[str, Any], init_hh: str) -> RuntimePaths:
    if init_hh not in ("00", "12"):
        raise ValueError("init_hh must be '00' or '12'.")

    paths = cfg["paths"]
    history_key = "history_00" if init_hh == "00" else "history_12"
    weather_key = "weather_dir_00" if init_hh == "00" else "weather_dir_12"

    runtime = RuntimePaths(
        history_csv=_resolve_path(paths[history_key]),
        weather_dir=_resolve_path(paths[weather_key]),
        load_daily_dir=_resolve_path(paths["load_daily_dir"]),
        outputs_dir=_resolve_path(paths["outputs_dir"]),
        runs_dir=_resolve_path(paths["runs_dir"]),
    )

    for p in [runtime.history_csv, runtime.weather_dir, runtime.load_daily_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required input path not found: {p}")
    runtime.outputs_dir.mkdir(parents=True, exist_ok=True)
    runtime.runs_dir.mkdir(parents=True, exist_ok=True)
    return runtime


def _forecast_file_path(outputs_dir: Path, init_time: pd.Timestamp) -> Path:
    return outputs_dir / f"forecast_{init_time.strftime('%Y%m%d%H')}.csv"


def _parse_forecast_csv(csv_path: Path, init_time: pd.Timestamp) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if {"ForeTime", "Load"}.issubset(df.columns):
        base_year = init_time.year
        ts = pd.to_datetime(
            [f"{base_year}/{x}" for x in df["ForeTime"].astype(str)],
            format="%Y/%m/%d %H",
            errors="coerce",
        )
        out = pd.DataFrame({"timestamp": ts, "predicted_load": pd.to_numeric(df["Load"], errors="coerce")})
    elif {"timestamp", "predicted_load"}.issubset(df.columns):
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["timestamp"], errors="coerce"),
                "predicted_load": pd.to_numeric(df["predicted_load"], errors="coerce"),
            }
        )
    else:
        raise ValueError(
            f"Unsupported forecast schema in {csv_path}. "
            "Expected either (ForeTime, Load) or (timestamp, predicted_load)."
        )
    out = out.dropna(subset=["timestamp", "predicted_load"]).sort_values("timestamp").reset_index(drop=True)
    return out


def _history_before_init(history_csv: Path, init_time: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(history_csv, parse_dates=["timestamp", "init_time"])
    load_col = _detect_load_col(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df[df["timestamp"] < init_time].copy(), load_col


def _compute_forecast_metrics(
    forecast_df: pd.DataFrame,
    history_df_before_init: pd.DataFrame,
    load_col: str,
) -> Dict[str, Any]:
    if forecast_df.empty:
        raise ValueError("Forecast dataframe is empty.")
    pred = forecast_df["predicted_load"].astype(float)
    idx_max = int(pred.idxmax())
    idx_min = int(pred.idxmin())

    hist_load = pd.to_numeric(history_df_before_init[load_col], errors="coerce").dropna()
    if hist_load.empty:
        raise ValueError("No historical load values available for capacity calculation.")
    capacity = float(np.percentile(hist_load.values, 90))
    exceed = pred - capacity
    exceed_pos = exceed[exceed > 0]

    return {
        "avg_predicted_load_mw": float(pred.mean()),
        "max_predicted_load_mw": float(pred.iloc[idx_max]),
        "max_predicted_load_ts": forecast_df.loc[idx_max, "timestamp"].isoformat(),
        "min_predicted_load_mw": float(pred.iloc[idx_min]),
        "min_predicted_load_ts": forecast_df.loc[idx_min, "timestamp"].isoformat(),
        "total_energy_proxy_mwh": float(pred.sum()),
        "capacity_mw": capacity,
        "hours_above_capacity": int(exceed_pos.shape[0]),
        "max_exceedance_mw": float(exceed_pos.max()) if not exceed_pos.empty else 0.0,
    }


def _extract_weather_window(
    init_time: pd.Timestamp,
    weather_dir: Path,
    history_csv: Path,
    variables_to_plot: list[str],
    horizon_hours: int,
) -> pd.DataFrame:
    mat_path = weather_dir / f"GRU_ECVars_{init_time.strftime('%Y%m%d%H')}.mat"
    use_cols = ["timestamp"] + list(variables_to_plot)

    if mat_path.exists():
        wx_df = ingest_weather_mat(mat_path)
        wx_df["timestamp"] = pd.to_datetime(wx_df["timestamp"], errors="coerce")
        max_ts = init_time + pd.Timedelta(hours=horizon_hours)
        wx_df = wx_df[(wx_df["timestamp"] >= init_time) & (wx_df["timestamp"] <= max_ts)].copy()
        available = [c for c in use_cols if c in wx_df.columns]
        return wx_df[available].sort_values("timestamp").reset_index(drop=True)

    hist = pd.read_csv(history_csv, parse_dates=["timestamp", "init_time"])
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["init_time"] = pd.to_datetime(hist["init_time"], errors="coerce")
    window = hist[hist["init_time"] == init_time].copy()
    available = [c for c in use_cols if c in window.columns]
    if "timestamp" not in available:
        return pd.DataFrame(columns=["timestamp"])
    return window[available].sort_values("timestamp").reset_index(drop=True)


def _find_prev_forecast(init_time: pd.Timestamp, outputs_dir: Path, runs_dir: Path) -> Optional[Path]:
    prev_init = init_time - pd.Timedelta(hours=24)
    run_id = prev_init.strftime("%Y%m%d%H")
    candidates = [
        runs_dir / run_id / "forecast.csv",
        outputs_dir / f"forecast_{run_id}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _compute_reality_check(
    prev_forecast_path: Optional[Path],
    current_init_time: pd.Timestamp,
    history_csv: Path,
    load_daily_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    if prev_forecast_path is None:
        return None, None

    prev_init = current_init_time - pd.Timedelta(hours=24)
    prev_forecast = _parse_forecast_csv(prev_forecast_path, prev_init)

    actuals = _load_actuals_until(
        history_csv=history_csv,
        load_daily_dir=load_daily_dir,
        cutoff_time=current_init_time,
    )

    merged = prev_forecast.merge(actuals, on="timestamp", how="inner")
    if merged.empty:
        return None, None
    merged["error"] = merged["predicted_load"] - merged["actual_load"]

    summary = {
        "realized_points": int(len(merged)),
        "mae_mw": float(np.abs(merged["error"]).mean()),
        "bias_mw": float(merged["error"].mean()),
        "prev_init_time": prev_init.isoformat(),
    }
    return merged, summary


def _load_actuals_until(history_csv: Path, load_daily_dir: Path, cutoff_time: pd.Timestamp) -> pd.DataFrame:
    """Load realized actuals from history + daily load files, deduplicated by timestamp."""
    hist = pd.read_csv(history_csv, parse_dates=["timestamp", "init_time"])
    load_col = _detect_load_col(hist)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist[load_col] = pd.to_numeric(hist[load_col], errors="coerce")
    hist_actuals = hist[["timestamp", load_col]].dropna().rename(columns={load_col: "actual_load"})

    daily_parts: list[pd.DataFrame] = []
    for csv_path in sorted(load_daily_dir.glob("GVL_D_*.csv")):
        try:
            ld = ingest_load_csv(csv_path).rename(columns={"load_MW": "actual_load"})
        except Exception:
            continue
        daily_parts.append(ld[["timestamp", "actual_load"]])

    parts = [hist_actuals]
    if daily_parts:
        parts.append(pd.concat(daily_parts, ignore_index=True))

    actuals = pd.concat(parts, ignore_index=True)
    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"], errors="coerce")
    actuals["actual_load"] = pd.to_numeric(actuals["actual_load"], errors="coerce")
    actuals = actuals.dropna(subset=["timestamp", "actual_load"])
    actuals = actuals[actuals["timestamp"] < cutoff_time]
    actuals = actuals.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return actuals.reset_index(drop=True)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, indent=2)


def run_for_date(date_str: str, init_hh: str = "00", config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config(config_path)
    target_day = _validate_date(date_str, cfg)
    paths = _build_paths(cfg, init_hh)
    horizon_hours = int(cfg.get("horizon_hours", 90))
    variables_to_plot = list(cfg.get("variables_to_plot", ["T2m", "Td2m", "RH2m"]))

    init_time = target_day.replace(hour=int(init_hh))
    run_id = init_time.strftime("%Y%m%d%H")
    forecast_src = _forecast_file_path(paths.outputs_dir, init_time)
    from_cache = forecast_src.exists()

    if not from_cache:
        try:
            run_forecast(
                date_str=target_day.strftime("%Y-%m-%d"),
                init_hh=init_hh,
                history_csv=paths.history_csv,
                history_csv_00=None,
                history_csv_12=None,
                weather_dir=paths.weather_dir,
                weather_dir_00=None,
                weather_dir_12=None,
                load_dir=paths.load_daily_dir,
                out_dir=paths.outputs_dir,
            )
        except Exception as exc:
            LOGGER.exception("Forecast generation failed for %s", run_id)
            raise RuntimeError(
                f"Forecast generation failed for {run_id}. "
                "Check load/weather coverage and input files."
            ) from exc

    if not forecast_src.exists():
        raise FileNotFoundError(
            f"Forecast output missing after run: {forecast_src}. "
            "The pipeline did not produce the expected CSV."
        )

    run_dir = paths.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    forecast_csv = run_dir / "forecast.csv"
    shutil.copy2(forecast_src, forecast_csv)

    forecast_df = _parse_forecast_csv(forecast_csv, init_time)
    history_before_init, load_col = _history_before_init(paths.history_csv, init_time)
    metrics = _compute_forecast_metrics(forecast_df, history_before_init, load_col)

    weather_df = _extract_weather_window(
        init_time=init_time,
        weather_dir=paths.weather_dir,
        history_csv=paths.history_csv,
        variables_to_plot=variables_to_plot,
        horizon_hours=horizon_hours,
    )
    weather_csv = run_dir / "weather_window.csv"
    if not weather_df.empty:
        weather_df.to_csv(weather_csv, index=False)

    backtest_df, backtest_summary = _compute_reality_check(
        prev_forecast_path=_find_prev_forecast(init_time, paths.outputs_dir, paths.runs_dir),
        current_init_time=init_time,
        history_csv=paths.history_csv,
        load_daily_dir=paths.load_daily_dir,
    )
    backtest_csv = run_dir / "backtest_last_available.csv"
    if backtest_df is not None and not backtest_df.empty:
        backtest_df.to_csv(backtest_csv, index=False)
    if backtest_summary is not None:
        metrics["yesterday_performance"] = backtest_summary

    metrics_path = run_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    forecast_payload = {
        "run_id": run_id,
        "cached": from_cache,
        "init_time": init_time.isoformat(),
        "horizon_hours": horizon_hours,
        "forecast": [
            {
                "timestamp": row.timestamp.isoformat(),
                "predicted_load": float(row.predicted_load),
            }
            for row in forecast_df.itertuples(index=False)
        ],
        "metrics": metrics,
        "files": {
            "forecast_csv": str(forecast_csv),
            "forecast_source_csv": str(forecast_src),
            "metrics_json": str(metrics_path),
            "forecast_json": str(run_dir / "forecast.json"),
            "weather_window_csv": str(weather_csv) if weather_df is not None else None,
            "backtest_last_available_csv": str(backtest_csv)
            if backtest_df is not None and not backtest_df.empty
            else None,
        },
    }
    forecast_json_path = run_dir / "forecast.json"
    _write_json(forecast_json_path, forecast_payload)

    return {
        "run_id": run_id,
        "cached": from_cache,
        "init_time": init_time.isoformat(),
        "metrics": metrics,
        "forecast_df": forecast_df,
        "weather_df": weather_df,
        "backtest_df": backtest_df,
        "run_dir": str(run_dir),
        "files": forecast_payload["files"],
    }


def get_or_run_forecast(date_str: str, init_hh: str = "00", config_path: Optional[str] = None) -> Dict[str, Any]:
    return run_for_date(date_str=date_str, init_hh=init_hh, config_path=config_path)


def get_recent_reality_windows(
    date_str: str,
    init_hh: str = "00",
    days_back: int = 4,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Return realized actual-vs-predicted windows for prior init days.

    For a selected init time T, this returns windows for T-24h, T-48h, ... up to
    `days_back`, where actuals are only kept for timestamps strictly before T.
    """
    cfg = load_config(config_path)
    target_day = _validate_date(date_str, cfg)
    paths = _build_paths(cfg, init_hh)
    current_init_time = target_day.replace(hour=int(init_hh))

    actuals = _load_actuals_until(
        history_csv=paths.history_csv,
        load_daily_dir=paths.load_daily_dir,
        cutoff_time=current_init_time,
    )

    rows: list[pd.DataFrame] = []
    min_day = pd.to_datetime(cfg["allowed_date_min"]).normalize()
    for offset in range(1, days_back + 1):
        prior_init = current_init_time - pd.Timedelta(days=offset)
        if prior_init.normalize() < min_day:
            continue
        forecast_path = _forecast_file_path(paths.outputs_dir, prior_init)
        if not forecast_path.exists():
            continue
        prior_df = _parse_forecast_csv(forecast_path, prior_init)
        merged = prior_df.merge(actuals, on="timestamp", how="left")
        merged["error"] = merged["predicted_load"] - merged["actual_load"]
        merged["source_init_time"] = prior_init
        merged["source_init_label"] = prior_init.strftime("%Y-%m-%d %HZ")
        merged["realized"] = merged["actual_load"].notna()
        rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "predicted_load",
                "actual_load",
                "error",
                "source_init_time",
                "source_init_label",
                "realized",
            ]
        )

    return pd.concat(rows, ignore_index=True).sort_values(["source_init_time", "timestamp"]).reset_index(drop=True)
