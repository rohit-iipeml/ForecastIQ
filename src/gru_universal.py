
import argparse
import os
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pytz
from scipy.io import loadmat
from lightgbm import LGBMRegressor


FEATURE_GROUPS = {
    "time_cyc":  ['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','is_weekend'],
    "wx_base":   ['T2m','Td2m','RH2m'],
    "lags_core": ['lag_1','lag_2','lag_6','lag_24'],
    "lags_long": ['lag_48','lag_168']
}
FEATURE_COLS_BEST = (
    FEATURE_GROUPS['time_cyc']
    + FEATURE_GROUPS['wx_base']
    + FEATURE_GROUPS['lags_core']
    + FEATURE_GROUPS['lags_long']
)
BEST_PARAMS = {
    'n_estimators': 667,
    'learning_rate': 0.024053250135099746,
    'num_leaves': 31,
    'min_child_samples': 62,
    'subsample': 0.758528881316744,
    'colsample_bytree': 0.8479881637986328,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
}


def _opt_path(s: Optional[str]) -> Optional[Path]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() == "none":
        return None
    return Path(s)


def ensure_time_features(df: pd.DataFrame, tz_str: str = 'US/Eastern') -> pd.DataFrame:
    need = {'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','is_weekend'}
    if need.issubset(df.columns):
        return df
    df = df.copy()

    eastern = pytz.timezone(tz_str)
    # Treat df['timestamp'] as UTC for feature calc. Keep df columns naive for modeling.
    ts_utc = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['timestamp_et'] = ts_utc.dt.tz_convert(eastern)

    df['hour']      = df['timestamp_et'].dt.hour
    df['dayofweek'] = df['timestamp_et'].dt.dayofweek
    df['month']     = df['timestamp_et'].dt.month

    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24)
    df['dow_sin']   = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos']   = np.cos(2*np.pi*df['dayofweek']/7)
    df['month_sin'] = np.sin(2*np.pi*(df['month']-1)/12)
    df['month_cos'] = np.cos(2*np.pi*(df['month']-1)/12)
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

    return df.drop(columns=['hour','dayofweek','month'])


def add_lag_features(df: pd.DataFrame, load_map: pd.Series, lags=[1,2,6,24,48,168]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df['timestamp'].apply(
            lambda ts: load_map.get(ts - pd.Timedelta(hours=lag), np.nan)
        )
    return df


def ingest_load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H")
    df["timestamp"] = df["timestamp"].dt.round("h")
    df = df.rename(columns={"value": "load_MW"})
    return df[["timestamp", "load_MW"]].copy()


def ingest_weather_mat(mat_path: Path) -> pd.DataFrame:
    """Reads GRU_ECVars_YYYYMMDDHH.mat into long rows with lead hours."""
    raw = loadmat(str(mat_path))
    st = raw["Data"][0, 0]

    leadhrs = np.array(st["leadhrs"]).astype(int).ravel()
    T2m  = np.array(st["T2m"]).reshape(-1)
    Td2m = np.array(st["Td2m"]).reshape(-1)
    RH2m = np.array(st["RH2m"]).reshape(-1)

    # infer init_time from filename
    fname = mat_path.stem
    init_str = fname.split("_")[-1]
    init_time = pd.to_datetime(init_str, format="%Y%m%d%H")

    rows = []
    for j, h in enumerate(leadhrs):
        ts = init_time + pd.Timedelta(hours=int(h))
        rows.append({
            "timestamp": ts.tz_localize(None),
            "init_time": init_time.tz_localize(None),
            "lead_hour": int(h),
            "T2m": float(T2m[j]),
            "Td2m": float(Td2m[j]),
            "RH2m": float(RH2m[j]),
            "load_MW": np.nan,
        })
    return pd.DataFrame(rows)


def update_df_with_new_data(df_base: pd.DataFrame,
                            load_df: Optional[pd.DataFrame] = None,
                            weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df_updated = df_base.copy()

    if load_df is not None and not load_df.empty:
        df_updated = df_updated.merge(load_df, on="timestamp", how="outer", suffixes=("", "_new"))
        df_updated["load_MW"] = df_updated["load_MW"].where(~df_updated["load_MW"].isna(),
                                                            df_updated["load_MW_new"])
        df_updated.drop(columns=["load_MW_new"], inplace=True, errors="ignore")

    if weather_df is not None and not weather_df.empty:
        df_updated = pd.concat([df_updated, weather_df], ignore_index=True)

    # Keep most recent per (init_time, lead_hour)
    if "init_time" in df_updated.columns and "lead_hour" in df_updated.columns:
        df_updated = (
            df_updated
            .sort_values(["init_time", "lead_hour", "timestamp"])
            .drop_duplicates(subset=["init_time", "lead_hour"], keep="last")
            .reset_index(drop=True)
        )
    else:
        df_updated = df_updated.sort_values("timestamp").reset_index(drop=True)

    return df_updated


def select_paths_for_init(init_hh: str,
                          history_csv: Optional[Path],
                          history_csv_00: Optional[Path],
                          history_csv_12: Optional[Path],
                          weather_dir: Optional[Path],
                          weather_dir_00: Optional[Path],
                          weather_dir_12: Optional[Path]) -> Tuple[Path, Optional[Path]]:
    """Pick history CSV and weather dir for chosen init."""
    if init_hh not in ("00", "12"):
        raise ValueError("init must be '00' or '12'")

    # History
    if init_hh == "00":
        hist = history_csv_00 or history_csv
    else:
        hist = history_csv_12 or history_csv
    if hist is None:
        raise ValueError("Provide --history_csv or an init-specific history path.")

    # Weather dir
    if init_hh == "00":
        wdir = weather_dir_00 or weather_dir
    else:
        wdir = weather_dir_12 or weather_dir
    # wdir may be None if relying entirely on prebuilt history; we allow None.

    return Path(hist), (_opt_path(str(wdir)) if wdir is not None else None)


def latest_loaded_ts(df_hist: pd.DataFrame) -> Optional[pd.Timestamp]:
    x = df_hist.loc[df_hist['load_MW'].notna(), 'timestamp']
    return None if x.empty else pd.to_datetime(x).max().tz_localize(None)


def latest_weather_date_for_hour(df_hist: pd.DataFrame, hour_str: str) -> Optional[pd.Timestamp]:
    hh = int(hour_str)
    if 'init_time' not in df_hist.columns:
        return None
    tmp = df_hist[pd.to_datetime(df_hist['init_time']).dt.hour == hh]
    if tmp.empty:
        return None
    return pd.to_datetime(tmp['init_time']).max().tz_localize(None)


def ingest_missing_weather(df_hist: pd.DataFrame,
                           weather_dir: Optional[Path],
                           init_hh: str,
                           target_day: pd.Timestamp) -> pd.DataFrame:
    """Add only missing daily MATs from last covered date+1 to target_day."""
    if weather_dir is None:
        # If history does not already contain the target init_time, we will fail later.
        return df_hist

    start_floor = pd.Timestamp("2025-01-01")
    last_have = latest_weather_date_for_hour(df_hist, init_hh)
    start_day = start_floor if last_have is None else max(start_floor, last_have.normalize() + pd.Timedelta(days=1))

    if start_day > target_day.normalize():
        # up to date
        return df_hist

    hh = init_hh
    df_upd = df_hist.copy()
    for d in pd.date_range(start_day, target_day, freq="D"):
        mat_path = weather_dir / f"GRU_ECVars_{d.strftime('%Y%m%d')}{hh}.mat"
        if mat_path.exists():
            wx_df = ingest_weather_mat(mat_path)
            df_upd = update_df_with_new_data(df_upd, weather_df=wx_df)
        else:
            print(f"[WARN] Missing weather file for {d.date()} {hh}Z at {mat_path}")
    return df_upd


def ingest_missing_loads(df_hist: pd.DataFrame,
                         load_dir: Path,
                         needed_latest_ts: pd.Timestamp) -> pd.DataFrame:
    """Append only missing daily CSVs and, if needed, partial same-day hours up to needed_latest_ts."""
    df_upd = df_hist.copy()
    have_last = latest_loaded_ts(df_upd)

    # Determine first day to start ingesting
    floor_day = pd.Timestamp("2025-01-01")
    if have_last is None:
        start_day = floor_day
    else:
        start_day = max(floor_day, (have_last + pd.Timedelta(hours=1)).normalize())

    # Ingest all full days strictly before needed day
    needed_day = needed_latest_ts.normalize()
    for d in pd.date_range(start_day, needed_day - pd.Timedelta(days=1), freq="D"):
        csv_path = load_dir / f"GVL_D_{d.strftime('%Y%m%d')}.csv"
        if csv_path.exists():
            ld_df = ingest_load_csv(csv_path)
            df_upd = update_df_with_new_data(df_upd, load_df=ld_df)
        else:
            print(f"[WARN] Missing load file for {d.date()} at {csv_path}")

    # Handle the needed day itself: require hours up to needed_latest_ts.hour
    # If file exists, append and then verify. If not, we will error below.
    day_csv = load_dir / f"GVL_D_{needed_day.strftime('%Y%m%d')}.csv"
    if day_csv.exists():
        ld_df = ingest_load_csv(day_csv)
        # Keep only rows up to needed hour
        ld_df = ld_df[ld_df['timestamp'] <= needed_latest_ts]
        df_upd = update_df_with_new_data(df_upd, load_df=ld_df)
    else:
        print(f"[WARN] Missing load file for {needed_day.date()} at {day_csv}")

    # Verify coverage
    after_last = latest_loaded_ts(df_upd)
    if (after_last is None) or (after_last < needed_latest_ts):
        have_str = "none" if after_last is None else after_last.strftime("%Y-%m-%d %H:%M")
        need_str = needed_latest_ts.strftime("%Y-%m-%d %H:%M")
        raise ValueError(
            f"Insufficient load coverage. Have through {have_str}, need through {need_str}. "
            f"Ensure daily CSV exists with hours up to {needed_latest_ts.hour:02d}."
        )

    return df_upd


def forecast_one_init_time_best(
    init_time,
    df: pd.DataFrame,
    feature_cols=FEATURE_COLS_BEST,
    lags=[1,2,6,24,48,168],
    model_params=BEST_PARAMS,
) -> pd.DataFrame:
    init_time = pd.to_datetime(init_time).tz_localize(None)
    df = df.copy()

    # normalize types
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df['init_time'] = pd.to_datetime(df['init_time']).dt.tz_localize(None)

    df = ensure_time_features(df)

    # training data strictly prior to init_time with observed load
    train_df = df[(df['timestamp'] < init_time) & (df['load_MW'].notna())].copy()
    if train_df.empty:
        raise ValueError("No training data before the given init_time.")

    # build actuals map and lags
    load_map_actuals = train_df.drop_duplicates('timestamp').set_index('timestamp')['load_MW']
    train_df = add_lag_features(train_df, load_map_actuals, lags=lags)

    safe_feats = [c for c in feature_cols if c in train_df.columns]
    if not safe_feats:
        raise ValueError("Requested feature_cols are missing from the DataFrame.")
    monotone = [0]*len(safe_feats)

    train_df = train_df.dropna(subset=['load_MW'])
    train_df[safe_feats] = train_df[safe_feats].astype(float)

    model = LGBMRegressor(**model_params, monotone_constraints=monotone)
    model.fit(train_df[safe_feats], train_df['load_MW'])

    window_df = df[df['init_time'] == init_time].copy().sort_values('lead_hour').reset_index(drop=True)
    if window_df.empty:
        raise ValueError("No rows found for the given init_time in df.")

    preds, predicted_map = [], {}
    hist_series = load_map_actuals[load_map_actuals.index < init_time]

    for _, row in window_df.iterrows():
        ts = row['timestamp']
        combined_map = pd.concat([hist_series, pd.Series(predicted_map)])
        combined_map = combined_map[~combined_map.index.duplicated(keep='last')]

        row_df = row.to_frame().T
        row_df = add_lag_features(row_df, combined_map, lags=lags)

        for c in safe_feats:
            if c not in row_df.columns:
                row_df[c] = np.nan
        row_df = row_df[safe_feats].astype(float)

        y_hat = model.predict(row_df)[0]
        preds.append(y_hat)
        predicted_map[ts] = y_hat

    window_df['predicted_load'] = preds
    return window_df[['timestamp','init_time','lead_hour','predicted_load']]


def run_forecast(date_str: str,
                 init_hh: str,
                 history_csv: Optional[Path],
                 history_csv_00: Optional[Path],
                 history_csv_12: Optional[Path],
                 weather_dir: Optional[Path],
                 weather_dir_00: Optional[Path],
                 weather_dir_12: Optional[Path],
                 load_dir: Path,
                 out_dir: Path):
    # pick paths for chosen init
    hist_csv, wx_dir = select_paths_for_init(
        init_hh, history_csv, history_csv_00, history_csv_12, weather_dir, weather_dir_00, weather_dir_12
    )

    if not hist_csv.exists():
        raise FileNotFoundError(f"History CSV not found: {hist_csv}")
    if load_dir is None or not Path(load_dir).exists():
        raise FileNotFoundError(f"Load directory not found: {load_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # read history
    df_hist = pd.read_csv(hist_csv, parse_dates=['timestamp', 'init_time'])

    # set target init_time
    target_day = pd.to_datetime(date_str)
    init_time = target_day.replace(hour=int(init_hh))

    # 1) ensure weather rows present for this init hour up to target_day
    df_hist = ingest_missing_weather(df_hist, wx_dir, init_hh, target_day)

    # If still no rows for this init_time, fail early with a clear message
    if df_hist[df_hist['init_time'] == init_time].empty:
        needed_tag = f"{target_day.strftime('%Y%m%d')}{init_hh}"
        where = "the configured weather_dir"
        raise ValueError(
            f"No rows for init_time {init_time}. Provide {where} containing GRU_ECVars_{needed_tag}.mat "
            f"or ensure history already includes that window."
        )

    # 2) ensure load coverage: strictly before init_time
    needed_latest = init_time - pd.Timedelta(hours=1)
    df_hist = ingest_missing_loads(df_hist, Path(load_dir), needed_latest)

    # 3) forecast
    fcst_df = forecast_one_init_time_best(init_time, df_hist)

    # 4) write two-column output with naming convention
    out_view = fcst_df[['timestamp','predicted_load']].copy()
    out_view['ForeTime'] = pd.to_datetime(out_view['timestamp']).dt.strftime('%m/%d %H')
    out_view['Load'] = out_view['predicted_load'].round(2)
    out_view = out_view[['ForeTime','Load']]

    init_ts = pd.to_datetime(fcst_df['init_time'].iloc[0])
    out_csv = out_dir / f"forecast_{init_ts.strftime('%Y%m%d%H')}.csv"
    out_view.to_csv(out_csv, index=False)

    print(f"[OK] Forecast saved to: {out_csv}")
    with pd.option_context("display.width", 120, "display.max_rows", 10):
        print("\n=== Forecast Preview ===")
        print(out_view.head(10))


def main():
    p = argparse.ArgumentParser(description="Universal GRU forecast (00Z or 12Z) with history self-check and safe ingest.")
    p.add_argument("date", help="Target date in YYYY-MM-DD (e.g., 2025-01-24)")
    p.add_argument("init", choices=["00","12"], help="Initialization hour: 00 or 12")

    # Generic paths (used for the chosen init unless overridden)
    p.add_argument("--history_csv", type=str, default=None, help="Path to merged history CSV for chosen init")
    p.add_argument("--weather_dir", type=str, default=None, help="Dir with GRU_ECVars_YYYYMMDDHH.mat for chosen init")

    # Optional init-specific overrides
    p.add_argument("--history_csv_00", type=str, default=None, help="History CSV for 00Z")
    p.add_argument("--history_csv_12", type=str, default=None, help="History CSV for 12Z")
    p.add_argument("--weather_dir_00", type=str, default=None, help="Weather dir for 00Z .mat files")
    p.add_argument("--weather_dir_12", type=str, default=None, help="Weather dir for 12Z .mat files")

    # Common paths
    p.add_argument("--load_dir", type=str, required=True, help="Directory with GVL_D_YYYYMMDD.csv files")
    p.add_argument("--out_dir", type=str, required=True, help="Directory to save outputs")
    args = p.parse_args()

    run_forecast(
        date_str=args.date,
        init_hh=args.init,
        history_csv=_opt_path(args.history_csv),
        history_csv_00=_opt_path(args.history_csv_00),
        history_csv_12=_opt_path(args.history_csv_12),
        weather_dir=_opt_path(args.weather_dir),
        weather_dir_00=_opt_path(args.weather_dir_00),
        weather_dir_12=_opt_path(args.weather_dir_12),
        load_dir=Path(args.load_dir),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()