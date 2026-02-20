from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RUNS_ROOT = Path("outputs/runs")


@dataclass
class ToolResult:
    tool_name: str
    run_id: str
    created_paths: dict[str, str]
    summary: dict[str, Any]
    preview_markdown: str
    errors: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts


def _first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _preview_markdown(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "No rows returned."
    head = df.head(n).copy()
    cols = list(head.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in head.iterrows():
        vals = []
        for c in cols:
            v = row.get(c)
            if isinstance(v, float):
                vals.append(str(v))
            elif pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _args_hash(args: dict[str, Any]) -> str:
    text = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tool_paths(run_id: str, tool_name: str, args: dict[str, Any]) -> tuple[Path, Path, Path]:
    tools_dir = RUNS_ROOT / run_id / "phase4" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    args_h = _args_hash(args)
    meta = tools_dir / f"{tool_name}__{args_h}.meta.json"
    run_meta = tools_dir / f"{tool_name}__latest.meta.json"
    return tools_dir, meta, run_meta


def _maybe_cached(run_id: str, tool_name: str, args: dict[str, Any]) -> ToolResult | None:
    _, meta_path, _ = _tool_paths(run_id, tool_name, args)
    if not meta_path.exists():
        return None
    try:
        meta = _read_json(meta_path)
        created = meta.get("created_paths", {}) or {}
        if not created:
            return None
        for p in created.values():
            if p and not Path(p).exists():
                return None
        return ToolResult(
            tool_name=tool_name,
            run_id=run_id,
            created_paths=created,
            summary=meta.get("summary", {}),
            preview_markdown=meta.get("preview_markdown", ""),
            errors=meta.get("errors", []),
        )
    except Exception:
        return None


def _save_meta(result: ToolResult, args: dict[str, Any]) -> None:
    _, meta_path, run_meta = _tool_paths(result.run_id, result.tool_name, args)
    payload = {
        "tool_name": result.tool_name,
        "run_id": result.run_id,
        "args": args,
        "args_hash": _args_hash(args),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **asdict(result),
    }
    _write_json(meta_path, payload)
    _write_json(run_meta, payload)


def _load_load_series_for_exceedance(run_id: str) -> tuple[pd.DataFrame, str]:
    run_dir = RUNS_ROOT / run_id
    p_cons = run_dir / "phase2" / "consensus_series.csv"
    if p_cons.exists():
        df = pd.read_csv(p_cons)
        ts_col = _first_existing(list(df.columns), ["target_timestamp", "timestamp", "ForeTime", "time"])
        load_col = _first_existing(list(df.columns), ["median", "consensus_median", "load_mw", "Load"])
        if ts_col and load_col:
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
                "load_mw": pd.to_numeric(df[load_col], errors="coerce"),
            }).dropna(subset=["timestamp", "load_mw"])
            return out, str(p_cons)

    p_fc = run_dir / "forecast.csv"
    if p_fc.exists():
        df = pd.read_csv(p_fc)
        year = int(run_id[:4])
        if "ForeTime" in df.columns and "Load" in df.columns:
            ts = pd.to_datetime(df["ForeTime"].astype(str).map(lambda x: f"{year}/{x}"), errors="coerce")
            out = pd.DataFrame({
                "timestamp": ts,
                "load_mw": pd.to_numeric(df["Load"], errors="coerce"),
            }).dropna(subset=["timestamp", "load_mw"])
            return out, str(p_fc)
        ts_col = _first_existing(list(df.columns), ["timestamp", "target_timestamp", "ForeTime", "time"])
        load_col = _first_existing(list(df.columns), ["Load", "predicted_load", "load_mw"])
        if ts_col and load_col:
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
                "load_mw": pd.to_numeric(df[load_col], errors="coerce"),
            }).dropna(subset=["timestamp", "load_mw"])
            return out, str(p_fc)

    raise FileNotFoundError("No load series available from phase2 consensus_series.csv or forecast.csv")


def tool_exceedance_hours(run_id: str) -> ToolResult:
    tool_name = "tool_exceedance_hours"
    args: dict[str, Any] = {}
    cached = _maybe_cached(run_id, tool_name, args)
    if cached is not None:
        return cached

    run_dir = RUNS_ROOT / run_id
    tools_dir, _, _ = _tool_paths(run_id, tool_name, args)
    out_csv = tools_dir / "exceedance_hours.csv"
    out_json = tools_dir / "exceedance_hours_summary.json"
    errors: list[str] = []

    briefing = _read_json(run_dir / "phase3" / "briefing.json")
    capacity_mw = float((briefing.get("capacity") or {}).get("capacity_mw"))

    load_df, load_source = _load_load_series_for_exceedance(run_id)
    df = load_df.copy()
    df["capacity_mw"] = capacity_mw
    df["exceedance_mw"] = df["load_mw"] - capacity_mw
    df = df[df["load_mw"] >= capacity_mw].copy()
    df = df.sort_values("timestamp")

    df.to_csv(out_csv, index=False)

    summary = {
        "exceedance_hours_count": int(len(df)),
        "max_exceedance_mw": float(df["exceedance_mw"].max()) if not df.empty else 0.0,
        "first_exceedance_time": df["timestamp"].iloc[0].isoformat() if not df.empty else None,
        "last_exceedance_time": df["timestamp"].iloc[-1].isoformat() if not df.empty else None,
        "capacity_mw": capacity_mw,
        "load_source": load_source,
    }
    _write_json(out_json, summary)

    result = ToolResult(
        tool_name=tool_name,
        run_id=run_id,
        created_paths={
            "exceedance_hours_csv": str(out_csv),
            "summary_json": str(out_json),
        },
        summary=summary,
        preview_markdown=_preview_markdown(df),
        errors=errors,
    )
    _save_meta(result, args)
    return result


def _resolve_source_csv(run_id: str, path_or_name: str) -> Path:
    p = Path(path_or_name)
    if p.exists():
        return p
    p2 = RUNS_ROOT / run_id / "phase4" / "tools" / path_or_name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"timestamps_source_csv not found: {path_or_name}")


def _load_weather_series(run_id: str, var: str) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    run_dir = RUNS_ROOT / run_id
    p_cons = run_dir / "phase25" / f"weather_consensus_{var}.csv"
    if p_cons.exists():
        df = pd.read_csv(p_cons)
        ts_col = _first_existing(list(df.columns), ["timestamp", "target_timestamp", "ForeTime", "time"])
        val_col = _first_existing(list(df.columns), ["median", var])
        if ts_col and val_col:
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
                var: pd.to_numeric(df[val_col], errors="coerce"),
            }).dropna(subset=["timestamp"])
            return out, str(p_cons)

    p_w = run_dir / "weather_window.csv"
    if p_w.exists():
        df = pd.read_csv(p_w)
        ts_col = _first_existing(list(df.columns), ["timestamp", "target_timestamp", "ForeTime", "time"])
        if ts_col and var in df.columns:
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
                var: pd.to_numeric(df[var], errors="coerce"),
            }).dropna(subset=["timestamp"])
            return out, str(p_w)

    return None, None


def tool_weather_at_times(run_id: str, timestamps_source_csv: str, vars: list[str] | None = None) -> ToolResult:
    tool_name = "tool_weather_at_times"
    args = {
        "timestamps_source_csv": timestamps_source_csv,
        "vars": vars or ["T2m"],
    }
    cached = _maybe_cached(run_id, tool_name, args)
    if cached is not None:
        return cached

    tools_dir, _, _ = _tool_paths(run_id, tool_name, args)
    out_csv = tools_dir / "exceedance_weather.csv"
    out_json = tools_dir / "exceedance_weather_summary.json"
    errors: list[str] = []

    src = _resolve_source_csv(run_id, timestamps_source_csv)
    src_df = pd.read_csv(src)
    ts_col = _first_existing(list(src_df.columns), ["timestamp", "target_timestamp", "ForeTime", "time"])
    if ts_col is None:
        raise ValueError("timestamps_source_csv missing timestamp column")

    out = pd.DataFrame({"timestamp": pd.to_datetime(src_df[ts_col], errors="coerce")})
    if "load_mw" in src_df.columns:
        out["load_mw"] = pd.to_numeric(src_df["load_mw"], errors="coerce")
    if "exceedance_mw" in src_df.columns:
        out["exceedance_mw"] = pd.to_numeric(src_df["exceedance_mw"], errors="coerce")

    weather_sources: dict[str, str] = {}
    for v in (vars or ["T2m"]):
        wdf, src_path = _load_weather_series(run_id, v)
        if wdf is None:
            errors.append(f"Weather variable unavailable: {v}")
            out[v] = pd.NA
            continue
        weather_sources[v] = src_path
        wdf = wdf.sort_values("timestamp")
        merged = pd.merge(out[["timestamp"]], wdf, on="timestamp", how="left")
        out[v] = merged[v]

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out.to_csv(out_csv, index=False)

    summary = {
        "row_count": int(len(out)),
        "requested_vars": vars or ["T2m"],
        "weather_sources": weather_sources,
        "timestamp_source": str(src),
        "missing_var_count": int(len(errors)),
    }
    _write_json(out_json, summary)

    result = ToolResult(
        tool_name=tool_name,
        run_id=run_id,
        created_paths={
            "exceedance_weather_csv": str(out_csv),
            "summary_json": str(out_json),
        },
        summary=summary,
        preview_markdown=_preview_markdown(out),
        errors=errors,
    )
    _save_meta(result, args)
    return result


def tool_top_risk_hours(run_id: str) -> ToolResult:
    tool_name = "tool_top_risk_hours"
    args: dict[str, Any] = {}
    cached = _maybe_cached(run_id, tool_name, args)
    if cached is not None:
        return cached

    tools_dir, _, _ = _tool_paths(run_id, tool_name, args)
    out_csv = tools_dir / "top_risk_hours.csv"
    out_json = tools_dir / "top_risk_hours_summary.json"

    briefing = _read_json(RUNS_ROOT / run_id / "phase3" / "briefing.json")
    rows: list[dict[str, Any]] = []

    for r in (briefing.get("capacity_watchlist_hours", []) or []):
        rows.append(
            {
                "kind": "capacity",
                "timestamp": r.get("time"),
                "expected_load_mw": r.get("expected_load_mw"),
                "exceedance_mw": r.get("exceedance_mw"),
                "volatility_mw": None,
                "range_mw": None,
            }
        )

    for r in (briefing.get("stability_watchlist_hours", []) or []):
        rows.append(
            {
                "kind": "stability",
                "timestamp": r.get("time"),
                "expected_load_mw": r.get("expected_load_mw", r.get("median_load_mw")),
                "exceedance_mw": None,
                "volatility_mw": r.get("volatility_mw"),
                "range_mw": r.get("range_mw"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["kind", "timestamp"])
    df.to_csv(out_csv, index=False)

    summary = {
        "capacity_rows": int((df["kind"] == "capacity").sum()) if not df.empty else 0,
        "stability_rows": int((df["kind"] == "stability").sum()) if not df.empty else 0,
        "total_rows": int(len(df)),
    }
    _write_json(out_json, summary)

    result = ToolResult(
        tool_name=tool_name,
        run_id=run_id,
        created_paths={
            "top_risk_hours_csv": str(out_csv),
            "summary_json": str(out_json),
        },
        summary=summary,
        preview_markdown=_preview_markdown(df),
        errors=[],
    )
    _save_meta(result, args)
    return result


def tool_compare_revisions(run_id: str) -> ToolResult:
    tool_name = "tool_compare_revisions"
    args: dict[str, Any] = {}
    cached = _maybe_cached(run_id, tool_name, args)
    if cached is not None:
        return cached

    tools_dir, _, _ = _tool_paths(run_id, tool_name, args)
    out_csv = tools_dir / "revision_extremes.csv"
    out_json = tools_dir / "revision_extremes_summary.json"

    src = RUNS_ROOT / run_id / "phase2" / "per_hour_metrics.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing file: {src}")
    df = pd.read_csv(src)

    ts_col = _first_existing(list(df.columns), ["target_timestamp", "timestamp", "ForeTime", "time"])
    vol_col = _first_existing(list(df.columns), ["revision_volatility", "std"])
    range_col = _first_existing(list(df.columns), ["range", "consensus_range"])
    med_col = _first_existing(list(df.columns), ["consensus_median", "median"])
    if ts_col is None or (vol_col is None and range_col is None):
        raise ValueError("per_hour_metrics.csv missing required columns")

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
        "revision_volatility": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else pd.NA,
        "range": pd.to_numeric(df[range_col], errors="coerce") if range_col else pd.NA,
    })
    if med_col:
        out["consensus_median"] = pd.to_numeric(df[med_col], errors="coerce")

    sort_col = "revision_volatility" if vol_col else "range"
    out = out.sort_values(sort_col, ascending=False).head(10)
    out.to_csv(out_csv, index=False)

    full_range = pd.to_numeric(df[range_col], errors="coerce") if range_col else pd.Series(dtype=float)
    full_vol = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else pd.Series(dtype=float)
    max_idx = full_range.idxmax() if not full_range.empty and full_range.notna().any() else None
    max_time = None
    if max_idx is not None and ts_col in df.columns:
        ts = _parse_ts(df.loc[max_idx, ts_col])
        max_time = ts.isoformat() if ts is not None else str(df.loc[max_idx, ts_col])

    summary = {
        "max_range": float(full_range.max()) if not full_range.empty and full_range.notna().any() else None,
        "max_range_time": max_time,
        "avg_revision_volatility": float(full_vol.mean()) if not full_vol.empty and full_vol.notna().any() else None,
        "source": str(src),
    }
    _write_json(out_json, summary)

    result = ToolResult(
        tool_name=tool_name,
        run_id=run_id,
        created_paths={
            "revision_extremes_csv": str(out_csv),
            "summary_json": str(out_json),
        },
        summary=summary,
        preview_markdown=_preview_markdown(out),
        errors=[],
    )
    _save_meta(result, args)
    return result


ALLOWED_TOOLS = {
    "tool_exceedance_hours": tool_exceedance_hours,
    "tool_weather_at_times": tool_weather_at_times,
    "tool_top_risk_hours": tool_top_risk_hours,
    "tool_compare_revisions": tool_compare_revisions,
}


def execute_tool(run_id: str, tool_name: str, args: dict[str, Any]) -> ToolResult:
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError(f"Unsupported tool: {tool_name}")
    fn = ALLOWED_TOOLS[tool_name]
    if tool_name in {"tool_exceedance_hours", "tool_top_risk_hours", "tool_compare_revisions"}:
        return fn(run_id)
    if tool_name == "tool_weather_at_times":
        return fn(
            run_id,
            timestamps_source_csv=str(args.get("timestamps_source_csv", "exceedance_hours.csv")),
            vars=args.get("vars", ["T2m"]),
        )
    raise ValueError(f"Unsupported tool dispatch: {tool_name}")
