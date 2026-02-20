from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import timedelta
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.phase1_backend import get_or_run_forecast, get_recent_reality_windows, load_config
from src.phase2_backend import get_or_build_phase2
from src.phase25_backend import get_or_build_phase25
from src.phase3.generator import generate_phase3_outputs
from src.phase4.generator import get_or_build_phase4
from src.phase4.chat_handler import answer_question
from src.phase4.planner import plan_tools


PLOT_TEMPLATE = "plotly_dark"


def _fmt_mw(value: Any) -> str:
    try:
        return f"{float(value):,.1f} MW"
    except Exception:
        return "Not available"


def _fmt_mwh(value: Any) -> str:
    try:
        return f"{float(value):,.0f} MWh"
    except Exception:
        return "Not available"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "Not available"


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "Not available"


def _friendly_level(value: Any) -> str:
    if value is None:
        return "Not available"
    text = str(value).strip().replace("_", " ").lower()
    return text.title()


def _fmt_ts_short(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "Not available"
    return ts.strftime("%b %d, %H:%M")


def _safe_metrics(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value) if value is not None else default


def _to_celsius(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") - 273.15


def _weather_display_df(weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df.empty:
        return weather_df
    out = weather_df.copy()
    for v in ["T2m", "Td2m"]:
        if v in out.columns:
            out[v] = _to_celsius(out[v])
    return out


def _chart_key(run_id: str, tab: str, chart: str, section: str = "", extra: str = "") -> str:
    raw = f"{tab}::{section}::{chart}::{run_id}::{extra}"
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", ":"} else "_" for ch in raw.lower())
    return safe


def _load_forecast_plot(forecast_df: pd.DataFrame, capacity_mw: float) -> go.Figure:
    fig = go.Figure()
    peak_idx = forecast_df["predicted_load"].idxmax() if not forecast_df.empty else None
    peak_ts = forecast_df.loc[peak_idx, "timestamp"] if peak_idx is not None else None
    peak_val = forecast_df.loc[peak_idx, "predicted_load"] if peak_idx is not None else None
    fig.add_trace(
        go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_load"],
            mode="lines+markers",
            name="Predicted Load",
            line=dict(width=2.5),
            marker=dict(size=6),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Load=%{y:.1f} MW<extra></extra>",
        )
    )
    fig.add_hline(
        y=capacity_mw,
        line_width=2,
        line_dash="dash",
        annotation_text=f"Capacity (P90): {capacity_mw:.1f} MW",
    )

    exceed = forecast_df[forecast_df["predicted_load"] > capacity_mw]
    if not exceed.empty:
        fig.add_trace(
            go.Scatter(
                x=exceed["timestamp"],
                y=exceed["predicted_load"],
                mode="markers",
                name="Exceeds Capacity",
                marker=dict(size=10, symbol="diamond", color="#ff6b6b"),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Load=%{y:.1f} MW<extra></extra>",
            )
        )
    if peak_ts is not None:
        fig.add_vline(x=peak_ts, line_dash="dot", line_width=1.5, line_color="#ffd166")
        fig.add_trace(
            go.Scatter(
                x=[peak_ts],
                y=[peak_val],
                mode="markers+text",
                text=["Peak"],
                textposition="top center",
                name="Peak Hour",
                marker=dict(size=11, color="#ffd166"),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Peak=%{y:.1f} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title="90-Hour Load Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", x=0, y=1.18, xanchor="left", yanchor="top"),
        margin=dict(l=20, r=20, t=95, b=20),
        font=dict(size=13),
    )
    return fig


def _weather_plot(weather_df: pd.DataFrame, variable: str) -> go.Figure:
    is_temp = variable in {"T2m", "Td2m"}
    unit = "°C" if is_temp else ""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weather_df["timestamp"],
            y=weather_df[variable],
            mode="lines+markers",
            name=variable,
            hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{variable}=%{{y:.1f}} {unit}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{variable} Forecast Window",
        xaxis_title="Timestamp",
        yaxis_title=f"{variable} ({unit})" if unit else variable,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def _reality_plot(backtest_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["timestamp"],
            y=backtest_df["predicted_load"],
            mode="lines+markers",
            name="Predicted",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Pred=%{y:.2f} MW<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["timestamp"],
            y=backtest_df["actual_load"],
            mode="lines+markers",
            name="Actual",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual=%{y:.2f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        title="Last Available Reality Check",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def _multi_day_reality_plot(df: pd.DataFrame, source_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted_load"],
            mode="lines+markers",
            name="Predicted (90h)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Pred=%{y:.2f} MW<extra></extra>",
        )
    )
    realized = df[df["actual_load"].notna()]
    if not realized.empty:
        fig.add_trace(
            go.Scatter(
                x=realized["timestamp"],
                y=realized["actual_load"],
                mode="lines+markers",
                name="Actual (realized only)",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Actual=%{y:.2f} MW<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Prior Init {source_label}: Predicted vs Actual",
        xaxis_title="Timestamp",
        yaxis_title="Load (MW)",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", x=0, y=1.14, xanchor="left", yanchor="top"),
        margin=dict(l=20, r=20, t=90, b=20),
        font=dict(size=13),
    )
    return fig


def _consensus_vs_day1_plot(consensus_df: pd.DataFrame, matrix_df: pd.DataFrame, capacity_mw: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=consensus_df["target_timestamp"],
            y=consensus_df["q3"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=consensus_df["target_timestamp"],
            y=consensus_df["q1"],
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.15)",
            line=dict(width=0),
            name="Consensus IQR",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Q1=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=consensus_df["target_timestamp"],
            y=consensus_df["median"],
            mode="lines+markers",
            name="Consensus Median",
            line=dict(width=3),
            marker=dict(size=5),
        )
    )

    run_cols = sorted([c for c in matrix_df.columns if c.startswith("pred_init_")])
    if run_cols:
        day1_col = run_cols[-1]
        fig.add_trace(
            go.Scatter(
                x=matrix_df.index,
                y=matrix_df[day1_col],
                mode="lines+markers",
                name=f"Closest Init ({day1_col.replace('pred_init_', '')})",
                line=dict(width=2, dash="dot"),
                marker=dict(size=4),
            )
        )

    fig.add_hline(y=capacity_mw, line_dash="dash", annotation_text=f"Capacity {capacity_mw:.1f} MW")
    exceed = consensus_df[consensus_df["median"] > capacity_mw]
    if not exceed.empty:
        fig.add_trace(
            go.Scatter(
                x=exceed["target_timestamp"],
                y=exceed["median"],
                mode="markers",
                name="Median > Capacity",
                marker=dict(size=8, symbol="diamond"),
            )
        )
    fig.update_layout(
        title="Consensus vs Closest Init Forecast",
        xaxis_title="Target Timestamp",
        yaxis_title="Load (MW)",
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", x=0, y=1.16, xanchor="left", yanchor="top"),
        margin=dict(l=20, r=20, t=95, b=20),
        font=dict(size=13),
    )
    return fig


def _disagreement_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    z = matrix_df.T.values
    y = [c.replace("pred_init_", "") for c in matrix_df.columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=matrix_df.index,
            y=y,
            colorscale="Viridis",
            colorbar=dict(title="Pred MW"),
            hovertemplate="Init=%{y}<br>Time=%{x|%Y-%m-%d %H:%M}<br>Load=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Disagreement Heatmap (Predicted Load)",
        xaxis_title="Target Timestamp",
        yaxis_title="Init Run",
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def _volatility_bar(per_hour_df: pd.DataFrame) -> go.Figure:
    top5_idx = set(per_hour_df.nlargest(5, "revision_volatility").index.tolist())
    colors = ["#d62728" if i in top5_idx else "#1f77b4" for i in per_hour_df.index]
    fig = go.Figure(
        data=go.Bar(
            x=per_hour_df["target_timestamp"],
            y=per_hour_df["revision_volatility"],
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Std=%{y:.2f} MW<extra></extra>",
        )
    )
    fig.update_layout(
        title="Revision Volatility by Hour (Top 5 highlighted)",
        xaxis_title="Target Timestamp",
        yaxis_title="Std Dev Across Inits (MW)",
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def _peak_scatter(peak_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Scatter(
            x=pd.to_datetime(peak_df["init_run"], format="%Y%m%d%H", errors="coerce"),
            y=pd.to_datetime(peak_df["peak_timestamp"]),
            mode="markers+lines",
            marker=dict(
                size=peak_df["peak_value"].clip(lower=1) / peak_df["peak_value"].max() * 28,
                color=peak_df["peak_value"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Peak MW"),
            ),
            text=peak_df["init_run"],
            hovertemplate="Init=%{text}<br>Peak time=%{y|%Y-%m-%d %H:%M}<br>Peak MW=%{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Peak Timing Across Inits",
        xaxis_title="Init Time",
        yaxis_title="Predicted Peak Timestamp",
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def _joint_consensus_ops_plot(
    load_consensus_df: pd.DataFrame,
    t2m_consensus_df: pd.DataFrame,
    joint_risk_df: pd.DataFrame,
    capacity_mw: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=load_consensus_df["target_timestamp"], y=load_consensus_df["q3"], line=dict(width=0), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=load_consensus_df["target_timestamp"],
            y=load_consensus_df["q1"],
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.16)",
            line=dict(width=0),
            name="Load IQR",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=load_consensus_df["target_timestamp"],
            y=load_consensus_df["median"],
            mode="lines+markers",
            name="Load Median",
            line=dict(width=3),
            yaxis="y1",
        )
    )
    if not t2m_consensus_df.empty:
        fig.add_trace(
            go.Scatter(
                x=t2m_consensus_df["timestamp"],
                y=t2m_consensus_df["median"],
                mode="lines",
                name="T2m Median",
                line=dict(width=2, dash="dot"),
                yaxis="y2",
            )
        )
    fig.add_hline(y=capacity_mw, line_dash="dash", annotation_text=f"Capacity {capacity_mw:.1f} MW")
    risk_hours = joint_risk_df[joint_risk_df["HighOpsRisk"] == True] if not joint_risk_df.empty else pd.DataFrame()
    if not risk_hours.empty:
        fig.add_trace(
            go.Scatter(
                x=risk_hours["timestamp"],
                y=risk_hours["median_load"],
                mode="markers",
                name="High Ops Risk",
                marker=dict(symbol="x", size=10),
                yaxis="y1",
            )
        )
    fig.update_layout(
        title="Joint Consensus View (Load + T2m)",
        template=PLOT_TEMPLATE,
        xaxis=dict(title="Target Timestamp"),
        yaxis=dict(title="Load (MW)"),
        yaxis2=dict(title="T2m (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h", x=0, y=1.14),
        margin=dict(l=20, r=20, t=80, b=20),
        font=dict(size=13),
    )
    return fig


def _weather_var_disagreement_bar(weather_day_metrics: Dict[str, Any]) -> go.Figure:
    rows = []
    for var, d in weather_day_metrics.get("variables", {}).items():
        rows.append({"variable": var, "disagreement_index": d.get("variable_disagreement_index_iqr")})
    df = pd.DataFrame(rows).dropna()
    if not df.empty:
        df = df.sort_values("disagreement_index", ascending=False)
    fig = go.Figure(
        data=go.Bar(
            x=df["variable"] if not df.empty else [],
            y=df["disagreement_index"] if not df.empty else [],
            marker_color="#1f77b4",
        )
    )
    fig.update_layout(
        title="Variable Disagreement Index (Median IQR)",
        xaxis_title="Weather Variable",
        yaxis_title="Disagreement Index",
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def _attribution_scatter_plot(revision_pairs_df: pd.DataFrame, var_col: str, corr: Optional[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=revision_pairs_df[var_col],
            y=revision_pairs_df["delta_load"],
            mode="markers",
            name="Revision Pairs",
            hovertemplate="ΔVar=%{x:.3f}<br>ΔLoad=%{y:.3f}<extra></extra>",
        )
    )
    if revision_pairs_df[var_col].notna().sum() >= 2:
        d = revision_pairs_df[["delta_load", var_col]].dropna().sort_values(var_col)
        x = d[var_col].to_numpy()
        y = d["delta_load"].to_numpy()
        if len(x) >= 2:
            import numpy as _np

            coef = _np.polyfit(x, y, 1)
            yhat = coef[0] * x + coef[1]
            fig.add_trace(go.Scatter(x=x, y=yhat, mode="lines", name="Best-fit line"))

    title_corr = f" (corr={corr:.3f})" if corr is not None and not pd.isna(corr) else ""
    fig.update_layout(
        title=f"Revision Attribution: ΔLoad vs {var_col}{title_corr}",
        xaxis_title=var_col,
        yaxis_title="ΔLoad",
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=13),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="ForecastIQ Dashboard", layout="wide")
    cfg = load_config()
    min_day = pd.to_datetime(cfg["allowed_date_min"]).date()
    max_day = pd.to_datetime(cfg["allowed_date_max"]).date()
    default_day = pd.to_datetime("2025-01-25").date()

    if "control_date" not in st.session_state:
        st.session_state["control_date"] = default_day
    if "control_init" not in st.session_state:
        st.session_state["control_init"] = "00"

    st.title("ForecastIQ")
    st.caption("Load Forecast Dashboard - cache-first viewer for Jan 2025 runs.")
    st.markdown(
        """
<style>
.run-context {
  position: sticky;
  top: 0.5rem;
  z-index: 99;
  background: rgba(30, 41, 59, 0.86);
  border: 1px solid rgba(148, 163, 184, 0.35);
  padding: 0.6rem 0.8rem;
  border-radius: 0.6rem;
  margin-bottom: 0.75rem;
}
.muted-note { color: #94a3b8; font-size: 0.92rem; }
.status-chip {
  display: inline-block;
  border: 1px solid rgba(148, 163, 184, 0.45);
  background: rgba(15, 23, 42, 0.7);
  border-radius: 999px;
  padding: 0.25rem 0.7rem;
  margin-right: 0.45rem;
  margin-bottom: 0.45rem;
  font-size: 0.86rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Run Controls")
        st.date_input(
            "Forecast date",
            value=st.session_state["control_date"],
            min_value=min_day,
            max_value=max_day,
            key="control_date",
        )
        st.selectbox("Init hour", ["00", "12"], key="control_init")
        c_prev, c_next = st.columns(2)
        prev_day_clicked = c_prev.button("Prev Day", use_container_width=True)
        next_day_clicked = c_next.button("Next Day", use_container_width=True)
        run_clicked = st.button("Run / Load Forecast", use_container_width=True, type="primary")

    if prev_day_clicked:
        st.session_state["control_date"] = max(min_day, st.session_state["control_date"] - timedelta(days=1))
    if next_day_clicked:
        st.session_state["control_date"] = min(max_day, st.session_state["control_date"] + timedelta(days=1))

    should_run = (
        run_clicked
        or prev_day_clicked
        or next_day_clicked
        or ("phase1_result" not in st.session_state)
    )

    if should_run:
        with st.spinner("Preparing forecast package..."):
            try:
                result = get_or_run_forecast(
                    str(st.session_state["control_date"]),
                    st.session_state["control_init"],
                )
                st.session_state["phase1_result"] = result
                source_tag = "cache" if result["cached"] else "fresh run"
                st.success(f"Run {result['run_id']} loaded from {source_tag}.")
            except Exception as exc:
                st.error(f"Unable to prepare forecast: {exc}")
                st.exception(exc)
                return

    result = st.session_state.get("phase1_result")
    if not result:
        st.info("Choose date/init in the sidebar and click Run / Load Forecast.")
        return

    metrics = result["metrics"]
    forecast_df: pd.DataFrame = result["forecast_df"].copy()
    weather_df: pd.DataFrame = result["weather_df"].copy()
    backtest_df = result["backtest_df"]
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    if not weather_df.empty:
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    if backtest_df is not None and not backtest_df.empty:
        backtest_df = backtest_df.copy()
        backtest_df["timestamp"] = pd.to_datetime(backtest_df["timestamp"])

    loaded_init = pd.to_datetime(result["init_time"])
    run_id = loaded_init.strftime("%Y%m%d%H")
    st.markdown(
        f"<div class='run-context'><strong>Run Context:</strong> {run_id} | Init {loaded_init.strftime('%Y-%m-%d %H:%M')} | Horizon {int(cfg.get('horizon_hours', 90))}h</div>",
        unsafe_allow_html=True,
    )

    selected_init = pd.to_datetime(f"{st.session_state['control_date']} {st.session_state['control_init']}:00")
    if selected_init != loaded_init:
        st.warning(
            f"Loaded run is {loaded_init.strftime('%Y-%m-%d %HZ')}. "
            f"Current selection is {selected_init.strftime('%Y-%m-%d %HZ')}. "
            "Click Run / Load Forecast to refresh."
        )

    tabs = st.tabs(
        [
            "Overview",
            "Forecast",
            "Stability",
            "Ops Risk",
            "Briefing (Phase 3)",
            "Ask",
            "Artifacts",
        ]
    )

    with tabs[0]:
        st.subheader("Overview")
        st.markdown(
            f"<div class='muted-note'>This forecast covers the next {int(cfg.get('horizon_hours', 90))} hours starting at {loaded_init.strftime('%Y-%m-%d %H:%M')}.</div>",
            unsafe_allow_html=True,
        )
        p3_overview = generate_phase3_outputs(run_id, force=False, use_llm_writer=False)
        p3_overview_json_path = Path(p3_overview["files"]["briefing_json"])
        p3_overview_payload = json.loads(p3_overview_json_path.read_text(encoding="utf-8")) if p3_overview_json_path.exists() else {}

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Average Load", _fmt_mw(_safe_metrics(metrics, "avg_predicted_load_mw")))
        k2.metric("Peak Load", _fmt_mw(_safe_metrics(metrics, "max_predicted_load_mw")))
        k3.metric("Capacity (P90)", _fmt_mw(_safe_metrics(metrics, "capacity_mw")))
        k4.metric("Hours Above Capacity", str(int(_safe_metrics(metrics, "hours_above_capacity"))))

        risk_txt = _friendly_level(p3_overview_payload.get("risk_level", "Not available"))
        stability_txt = _friendly_level(p3_overview_payload.get("forecast_stability_level", "Not available"))
        agreement_txt = _friendly_level(p3_overview_payload.get("peak_timing_agreement", "Not available"))
        st.markdown(
            "".join(
                [
                    f"<span class='status-chip'><strong>Risk:</strong> {risk_txt}</span>",
                    f"<span class='status-chip'><strong>Forecast Stability:</strong> {stability_txt}</span>",
                    f"<span class='status-chip'><strong>Peak Timing:</strong> {agreement_txt}</span>",
                ]
            ),
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.45, 1.0], gap="large")
        capacity_mw = _safe_metrics(metrics, "capacity_mw")
        with left:
            st.plotly_chart(
                _load_forecast_plot(forecast_df, capacity_mw),
                use_container_width=True,
                key=_chart_key(run_id, "overview", "load_forecast_main", "story"),
            )
            st.markdown("**Plain-Language Summary**")
            peak_ts = _fmt_ts_short(metrics.get("max_predicted_load_ts"))
            peak_mw = _fmt_mw(metrics.get("max_predicted_load_mw"))
            risk_level = risk_txt
            stability_level = stability_txt
            peak_agree = agreement_txt
            st.write(
                "\n".join(
                    [
                        f"- Peak is expected around {peak_ts} at {peak_mw}.",
                        f"- Capacity threshold is {_fmt_mw(capacity_mw)} with {int(_safe_metrics(metrics, 'hours_above_capacity'))} forecast hour(s) above it.",
                        f"- Risk level is **{risk_level}** and forecast stability is **{stability_level}**.",
                        f"- Peak timing agreement across recent updates is **{peak_agree}**.",
                        "- Focus operations on top watch hours below before commitment decisions.",
                    ]
                )
            )
        with right:
            st.markdown("### Today's Callouts")
            c1, c2, c3 = st.columns(3)
            c1.metric("Peak Hour", _fmt_ts_short(metrics.get("max_predicted_load_ts")))
            c2.metric("Risk", risk_txt)
            c3.metric("Peak Agreement", agreement_txt)
            c4, c5 = st.columns(2)
            c4.metric("Hours > Capacity", str(int(_safe_metrics(metrics, "hours_above_capacity"))))
            c5.metric("Max Exceedance", _fmt_mw(metrics.get("max_exceedance_mw")))

            cap_watch = pd.DataFrame(p3_overview_payload.get("capacity_watchlist_hours", [])[:5])
            stab_watch = pd.DataFrame(p3_overview_payload.get("stability_watchlist_hours", [])[:5])
            st.markdown("### Top Watch Hours")
            if not cap_watch.empty:
                st.caption("Capacity Watchlist (top 5)")
                cap_show = cap_watch.rename(
                    columns={
                        "time": "Time",
                        "expected_load_mw": "Expected Load (MW)",
                        "exceedance_mw": "Exceedance (MW)",
                        "reason": "Reason",
                    }
                )
                if "Time" in cap_show.columns:
                    cap_show["Time"] = cap_show["Time"].map(_fmt_ts_short)
                st.dataframe(cap_show, use_container_width=True, hide_index=True)
            else:
                st.info("Capacity watchlist not available.")
            if not stab_watch.empty:
                st.caption("Stability Watchlist (top 5)")
                stab_show = stab_watch.rename(
                    columns={
                        "time": "Time",
                        "expected_load_display": "Expected Load",
                        "volatility_mw": "Volatility (MW)",
                        "range_mw": "Range (MW)",
                        "reason": "Reason",
                    }
                )
                if "Time" in stab_show.columns:
                    stab_show["Time"] = stab_show["Time"].map(_fmt_ts_short)
                st.dataframe(stab_show, use_container_width=True, hide_index=True)
            else:
                st.info("Stability watchlist not available.")
            with st.expander("Show full watchlists", expanded=False):
                st.write("Capacity")
                st.dataframe(pd.DataFrame(p3_overview_payload.get("capacity_watchlist_hours", [])), use_container_width=True, hide_index=True)
                st.write("Stability")
                st.dataframe(pd.DataFrame(p3_overview_payload.get("stability_watchlist_hours", [])), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Forecast")
        st.markdown("<div class='muted-note'>Load and weather outlook with quick reality checks.</div>", unsafe_allow_html=True)
        st.markdown("### Load Forecast")
        capacity_mw = _safe_metrics(metrics, "capacity_mw")
        st.plotly_chart(
            _load_forecast_plot(forecast_df, capacity_mw),
            use_container_width=True,
            key=_chart_key(run_id, "forecast", "load_forecast", "load"),
        )
        top5 = forecast_df.sort_values("predicted_load", ascending=False).head(5).copy()
        top5["timestamp"] = top5["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        top5["predicted_load"] = top5["predicted_load"].round(1)
        with st.expander("Top 5 Predicted Hours", expanded=False):
            st.dataframe(top5, use_container_width=True, hide_index=True)

        st.markdown("### Weather Forecast")
        weather_display_df = _weather_display_df(weather_df)
        if weather_display_df.empty:
            st.warning("Weather data is not available for this run.")
        else:
            st.caption("Display note: T2m and Td2m are shown in °C (converted from Kelvin for readability).")
            available_vars = [c for c in weather_display_df.columns if c != "timestamp"]
            if not available_vars:
                st.warning("Weather file loaded, but no configured variables were found.")
            for var in available_vars:
                st.plotly_chart(
                    _weather_plot(weather_display_df, var),
                    use_container_width=True,
                    key=_chart_key(run_id, "forecast", "weather_forecast", "weather", var),
                )

        st.markdown("### Reality Checks")
        perf = metrics.get("yesterday_performance")
        if backtest_df is None or backtest_df.empty:
            st.info("No realized overlap available for yesterday performance.")
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("Realized Points", str(perf.get("realized_points", 0) if perf else "Not available"), help="Count of timestamps with realized actual load available.")
            p2.metric("MAE", _fmt_mw(float(perf.get("mae_mw", 0.0)) if perf else None), help="Mean absolute error over realized portion.")
            p3.metric("Bias", _fmt_mw(float(perf.get("bias_mw", 0.0)) if perf else None), help="Average signed forecast error over realized portion.")
            with st.expander("Yesterday Actual vs Predicted", expanded=False):
                st.plotly_chart(
                    _reality_plot(backtest_df),
                    use_container_width=True,
                    key=_chart_key(run_id, "forecast", "reality_yesterday", "reality"),
                )

        recent = get_recent_reality_windows(loaded_init.strftime("%Y-%m-%d"), loaded_init.strftime("%H"), days_back=4)
        with st.expander("Prior 4 Days Reality Window", expanded=False):
            if recent.empty:
                st.info("No prior-day forecast files found for multi-day reality view.")
            else:
                st.caption(
                    "Shows prior init forecasts (up to 4 days back). Actuals appear only where realized before selected init time."
                )
                summary_rows = []
                for source_label, grp in recent.groupby("source_init_label", sort=False):
                    grp = grp.copy()
                    grp["timestamp"] = pd.to_datetime(grp["timestamp"])
                    st.plotly_chart(
                        _multi_day_reality_plot(grp, source_label),
                        use_container_width=True,
                        key=_chart_key(run_id, "forecast", "reality_multi_day", "reality", source_label),
                    )
                    realized = grp[grp["actual_load"].notna()].copy()
                    mae = float((realized["predicted_load"] - realized["actual_load"]).abs().mean()) if not realized.empty else None
                    bias = float((realized["predicted_load"] - realized["actual_load"]).mean()) if not realized.empty else None
                    summary_rows.append(
                        {
                            "init": source_label,
                            "forecast_points": int(len(grp)),
                            "realized_points": int(len(realized)),
                            "mae_mw": round(mae, 1) if mae is not None else None,
                            "bias_mw": round(bias, 1) if bias is not None else None,
                        }
                    )
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Stability")
        st.markdown("<div class='muted-note'>How much the forecast changes across update cycles.</div>", unsafe_allow_html=True)
        loaded_date_str = loaded_init.strftime("%Y-%m-%d")
        loaded_hour = loaded_init.strftime("%H")
        phase2_key = f"phase2::{loaded_date_str}::{loaded_hour}"
        if phase2_key not in st.session_state:
            with st.spinner("Building revision/stability analytics..."):
                st.session_state[phase2_key] = get_or_build_phase2(loaded_date_str, loaded_hour)

        p2 = st.session_state[phase2_key]
        dm = p2["day_metrics"]
        matrix_df = p2["forecast_matrix_df"].copy()
        if not matrix_df.empty:
            matrix_df.index = pd.to_datetime(matrix_df.index)
        per_hour_df = p2["per_hour_metrics_df"].copy()
        consensus_df = p2["consensus_series_df"].copy()
        exceed_df = p2["exceedance_proxy_df"].copy()
        peak_df = p2["peak_table_df"].copy()

        if dm.get("n_init_runs", 0) < 2 or matrix_df.empty:
            st.warning("Insufficient overlapping inits for advanced revision analytics.")
            with st.expander("Day metrics", expanded=False):
                st.json(dm)
        else:
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Disagreement Index", _fmt_mw(float(dm.get("disagreement_index_day", 0.0))))
            m2.metric("Peak Confidence", _fmt_pct(float(dm.get("peak_confidence", 0.0))))
            m3.metric("Peak Time Spread (h)", str(dm.get("peak_time_spread_hours", "Not available")))
            m4.metric("Avg Revision Volatility", _fmt_mw(float(dm.get("avg_revision_volatility", 0.0))))
            m5.metric("Max Range", _fmt_mw(float(dm.get("max_range", 0.0))))
            m6.metric("Majority Exceed Hours", str(int(dm.get("day_exceedance_hours_majority", 0))))
            st.caption(f"Max range timestamp: {dm.get('max_range_timestamp', 'Not available')}")

            if not consensus_df.empty:
                consensus_df["target_timestamp"] = pd.to_datetime(consensus_df["target_timestamp"])
                st.plotly_chart(
                    _consensus_vs_day1_plot(consensus_df, matrix_df, float(dm.get("capacity_mw", 0.0))),
                    use_container_width=True,
                    key=_chart_key(run_id, "stability", "consensus_vs_closest", "summary"),
                )
            if not per_hour_df.empty:
                per_hour_df["target_timestamp"] = pd.to_datetime(per_hour_df["target_timestamp"])
                st.plotly_chart(
                    _volatility_bar(per_hour_df),
                    use_container_width=True,
                    key=_chart_key(run_id, "stability", "volatility_bar", "summary"),
                )
            if not p2["voi_df"].empty:
                with st.expander("Value of Information (error vs forecast age)", expanded=False):
                    st.dataframe(p2["voi_df"], use_container_width=True, hide_index=True)
            with st.expander("Detailed Stability Views", expanded=False):
                if not matrix_df.empty:
                    st.plotly_chart(
                        _disagreement_heatmap(matrix_df),
                        use_container_width=True,
                        key=_chart_key(run_id, "stability", "disagreement_heatmap", "details"),
                    )
                if not peak_df.empty:
                    peak_df["peak_timestamp"] = pd.to_datetime(peak_df["peak_timestamp"])
                    st.plotly_chart(
                        _peak_scatter(peak_df),
                        use_container_width=True,
                        key=_chart_key(run_id, "stability", "peak_scatter", "details"),
                    )
                    st.dataframe(peak_df, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("Ops Risk")
        st.markdown("<div class='muted-note'>Weather-load coupled risk indicators and revision attribution.</div>", unsafe_allow_html=True)
        loaded_date_str = loaded_init.strftime("%Y-%m-%d")
        loaded_hour = loaded_init.strftime("%H")
        p25_key = f"phase25::{loaded_date_str}::{loaded_hour}"

        if p25_key not in st.session_state:
            st.info("Compute Phase 2.5 weather-load operational analytics for this run.")
            if st.button("Compute Phase 2.5", key=f"compute_phase25_{p25_key}", type="primary"):
                with st.spinner("Computing Weather–Load Ops analytics..."):
                    st.session_state[p25_key] = get_or_build_phase25(loaded_date_str, loaded_hour)
                st.success("Phase 2.5 ready.")

        p25 = st.session_state.get(p25_key)
        if p25:
            p2_key = f"phase2::{loaded_date_str}::{loaded_hour}"
            p2 = st.session_state.get(p2_key)
            if not p2:
                p2 = get_or_build_phase2(loaded_date_str, loaded_hour)
                st.session_state[p2_key] = p2

            load_dis = float(p2["day_metrics"].get("disagreement_index_day", 0.0))
            t2m_dis = p25["weather_day_metrics"].get("variables", {}).get("T2m", {}).get("variable_disagreement_index_iqr")
            r2 = p25["attribution_fit"].get("r2")
            peak_conf = p2["day_metrics"].get("peak_confidence")
            high_ops = int(p25["joint_ops_risk_df"]["HighOpsRisk"].sum()) if not p25["joint_ops_risk_df"].empty else 0

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Load Disagreement Index", _fmt_mw(load_dis))
            k2.metric("T2m Disagreement Index", _fmt_mw(float(t2m_dis)) if t2m_dis is not None else "Not available")
            k3.metric("Attribution R²", _fmt_num(float(r2), 2) if r2 is not None else "Not available")
            k4.metric("Peak Confidence", _fmt_pct(float(peak_conf)) if peak_conf is not None else "Not available")
            k5.metric("# High Ops Risk Hours", str(high_ops))

            load_cons = p2["consensus_series_df"].copy()
            load_cons["target_timestamp"] = pd.to_datetime(load_cons["target_timestamp"])
            t2m_cons = p25["weather_consensus"].get("T2m", pd.DataFrame()).copy()
            if not t2m_cons.empty:
                t2m_cons["timestamp"] = pd.to_datetime(t2m_cons["timestamp"])
                for c in ["median", "q1", "q3", "min", "max"]:
                    if c in t2m_cons.columns:
                        t2m_cons[c] = _to_celsius(t2m_cons[c])
            joint = p25["joint_ops_risk_df"].copy()
            if not joint.empty:
                joint["timestamp"] = pd.to_datetime(joint["timestamp"])
            st.plotly_chart(
                _joint_consensus_ops_plot(load_cons, t2m_cons, joint, float(p2["day_metrics"].get("capacity_mw", 0.0))),
                use_container_width=True,
                key=_chart_key(run_id, "ops_risk", "joint_consensus", "summary"),
            )

            st.plotly_chart(
                _weather_var_disagreement_bar(p25["weather_day_metrics"]),
                use_container_width=True,
                key=_chart_key(run_id, "ops_risk", "weather_disagreement_bar", "summary"),
            )

            w_per = p25["weather_per_hour_metrics_df"]
            if not w_per.empty:
                for var in sorted(w_per["variable"].unique().tolist()):
                    d = w_per[w_per["variable"] == var].copy()
                    d["timestamp"] = pd.to_datetime(d["timestamp"])
                    if var in {"T2m", "Td2m"}:
                        for c in ["median", "q1", "q3", "min", "max"]:
                            if c in d.columns:
                                d[c] = _to_celsius(d[c])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["q3"], line=dict(width=0), showlegend=False))
                    fig.add_trace(
                        go.Scatter(
                            x=d["timestamp"],
                            y=d["q1"],
                            fill="tonexty",
                            fillcolor="rgba(44,160,44,0.14)",
                            line=dict(width=0),
                            name=f"{var} IQR",
                        )
                    )
                    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["median"], mode="lines+markers", name=f"{var} Median"))
                    fig.update_layout(
                        title=f"{var} Consensus + Disagreement",
                        template=PLOT_TEMPLATE,
                        xaxis_title="Target Timestamp",
                        yaxis_title=f"{var} (°C)" if var in {"T2m", "Td2m"} else var,
                        margin=dict(l=20, r=20, t=60, b=20),
                    )
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=_chart_key(run_id, "ops_risk", "weather_consensus_iqr", "weather", var),
                    )

            rev = p25["revision_pairs_df"].copy()
            corr = p25["correlation_table_df"].copy()
            if not rev.empty:
                available_delta_vars = [c for c in rev.columns if c.startswith("delta_") and c != "delta_load"]
                if available_delta_vars:
                    primary_var = "delta_T2m" if "delta_T2m" in available_delta_vars else available_delta_vars[0]
                    corr_val = None
                    if not corr.empty:
                        cvar = primary_var.replace("delta_", "")
                        sub = corr[corr["variable"] == cvar]
                        if not sub.empty:
                            corr_val = float(sub.iloc[0]["corr_with_delta_load"])
                    st.plotly_chart(
                        _attribution_scatter_plot(rev, primary_var, corr_val),
                        use_container_width=True,
                        key=_chart_key(run_id, "ops_risk", "attribution_scatter", "attribution", primary_var),
                    )
                st.subheader("Attribution Fit")
                fit = p25["attribution_fit"]
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Model", str(fit.get("model", "Not available")))
                a2.metric("Samples", str(fit.get("n_samples", "Not available")))
                a3.metric("R²", _fmt_num(float(fit["r2"]), 2) if fit.get("r2") is not None else "Not available")
                a4.metric("Vars Used", str(len(fit.get("vars_used", []))))

                coefs = fit.get("coefficients", {}) or {}
                std_coefs = fit.get("standardized_coefficients", {}) or {}
                if coefs:
                    coef_rows = []
                    for k, v in coefs.items():
                        coef_rows.append(
                            {
                                "variable": k.replace("delta_", ""),
                                "coefficient": v,
                                "std_coefficient": std_coefs.get(k),
                            }
                        )
                    coef_df = pd.DataFrame(coef_rows).sort_values("variable")
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)
                st.caption("Interpretation: revision attribution / explained variance, not causality.")
                if not corr.empty:
                    st.subheader("Correlation Table")
                    with st.expander("Show correlation details", expanded=False):
                        st.dataframe(corr, use_container_width=True, hide_index=True)
                with st.expander("Raw attribution JSON", expanded=False):
                    st.json(fit)

            risk_top = p25["joint_ops_risk_df"].copy()
            if not risk_top.empty:
                risk_top = risk_top.sort_values(["HighOpsRisk", "ops_risk_score"], ascending=[False, False]).head(10)
                with st.expander("Top 10 Ops Risk Hours", expanded=False):
                    st.dataframe(risk_top, use_container_width=True, hide_index=True)

    with tabs[4]:
        loaded_date_str = loaded_init.strftime("%Y-%m-%d")
        loaded_hour = loaded_init.strftime("%H")
        run_id = loaded_init.strftime("%Y%m%d%H")
        p3_key = f"phase3::{loaded_date_str}::{loaded_hour}"
        p3_llm_toggle_key = f"phase3_llm_toggle::{loaded_date_str}::{loaded_hour}"
        p3_last_mode_key = f"phase3_last_mode::{loaded_date_str}::{loaded_hour}"

        use_llm = st.toggle("Polish briefing with LLM (Groq)", value=False, key=p3_llm_toggle_key)

        if p3_key not in st.session_state or st.session_state.get(p3_last_mode_key) != use_llm:
            with st.spinner("Preparing Phase 3 briefing..."):
                st.session_state[p3_key] = generate_phase3_outputs(run_id, force=False, use_llm_writer=use_llm)
                st.session_state[p3_last_mode_key] = use_llm

        p3 = st.session_state[p3_key]
        p3_files = p3.get("files", {})
        briefing_json_raw = p3_files.get("briefing_json")
        actions_json_raw = p3_files.get("action_items_json")
        phase3_input_raw = p3_files.get("phase3_input_json")
        briefing_md_raw = p3_files.get("briefing_md")
        briefing_llm_md_raw = p3_files.get("briefing_llm_md")

        briefing_json_path = Path(briefing_json_raw) if briefing_json_raw else None
        actions_json_path = Path(actions_json_raw) if actions_json_raw else None
        phase3_input_path = Path(phase3_input_raw) if phase3_input_raw else None
        briefing_md_path = Path(briefing_md_raw) if briefing_md_raw else None
        briefing_llm_md_path = Path(briefing_llm_md_raw) if briefing_llm_md_raw else None

        if briefing_json_path and briefing_json_path.is_file():
            briefing_payload = json.loads(briefing_json_path.read_text(encoding="utf-8"))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk Level", str(briefing_payload.get("risk_level", "Not available")))
            c2.metric("Forecast Stability", str(briefing_payload.get("forecast_stability_level", "Not available")))
            c3.metric("Peak Timing Agreement", str(briefing_payload.get("peak_timing_agreement", "Not available")))
            c4.metric("Actions", str(len(briefing_payload.get("recommended_actions", []))))

            st.subheader("Capacity Watchlist Hours")
            cap_rows = briefing_payload.get("capacity_watchlist_hours", []) or []
            if cap_rows:
                st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No capacity watchlist hours available.")

            st.subheader("Stability Watchlist Hours")
            stab_rows = briefing_payload.get("stability_watchlist_hours", []) or []
            if stab_rows:
                stab_df = pd.DataFrame(stab_rows)
                if "expected_load_display" not in stab_df.columns:
                    stab_df["expected_load_display"] = "—"
                stab_df["expected_load_display"] = stab_df["expected_load_display"].fillna("—")
                cols = [c for c in ["time", "expected_load_display", "volatility_mw", "range_mw", "reason"] if c in stab_df.columns]
                st.dataframe(stab_df[cols] if cols else stab_df, use_container_width=True, hide_index=True)
            else:
                st.info("No stability watchlist hours available.")

            st.subheader("Recommended Actions")
            for a in briefing_payload.get("recommended_actions", []):
                st.write(f"- `{a.get('priority')}` {a.get('title')}: {a.get('rationale')}")

        show_md_path = briefing_llm_md_path if (use_llm and briefing_llm_md_path and briefing_llm_md_path.is_file()) else briefing_md_path
        if use_llm:
            llm_info = p3.get("llm", {}) or {}
            llm_status = llm_info.get("status")
            meta_raw = llm_info.get("meta_path")
            meta_path = Path(meta_raw) if meta_raw else None
            if llm_status is None and meta_path and meta_path.is_file():
                try:
                    llm_status = json.loads(meta_path.read_text(encoding="utf-8")).get("status")
                except Exception:
                    llm_status = None
            if llm_status == "ok":
                st.markdown(
                    "<span style='background:#e8f5e9;color:#1b5e20;padding:4px 8px;border-radius:6px;font-size:0.9em;'>LLM-polished</span>",
                    unsafe_allow_html=True,
                )
            elif llm_status == "fallback":
                st.markdown(
                    "<span style='background:#f1f3f4;color:#37474f;padding:4px 8px;border-radius:6px;font-size:0.9em;'>LLM polish unavailable; deterministic briefing shown</span>",
                    unsafe_allow_html=True,
                )
        if show_md_path and show_md_path.is_file():
            st.subheader("Briefing")
            st.markdown(show_md_path.read_text(encoding="utf-8"))

        st.subheader("Downloads")
        if phase3_input_path and phase3_input_path.is_file():
            st.download_button(
                "Download phase3_input.json",
                data=phase3_input_path.read_bytes(),
                file_name=phase3_input_path.name,
                mime="application/json",
            )
        if briefing_json_path and briefing_json_path.is_file():
            st.download_button(
                "Download briefing.json",
                data=briefing_json_path.read_bytes(),
                file_name=briefing_json_path.name,
                mime="application/json",
            )
        if actions_json_path and actions_json_path.is_file():
            st.download_button(
                "Download action_items.json",
                data=actions_json_path.read_bytes(),
                file_name=actions_json_path.name,
                mime="application/json",
            )
        if briefing_md_path and briefing_md_path.is_file():
            st.download_button(
                "Download briefing.md",
                data=briefing_md_path.read_bytes(),
                file_name=briefing_md_path.name,
                mime="text/markdown",
            )
        if briefing_llm_md_path and briefing_llm_md_path.is_file():
            st.download_button(
                "Download briefing_llm.md",
                data=briefing_llm_md_path.read_bytes(),
                file_name=briefing_llm_md_path.name,
                mime="text/markdown",
            )

    with tabs[5]:
        loaded_date_str = loaded_init.strftime("%Y-%m-%d")
        loaded_hour = loaded_init.strftime("%H")
        run_id = loaded_init.strftime("%Y%m%d%H")
        p4_key = f"phase4::{loaded_date_str}::{loaded_hour}"
        chat_key = f"phase4_chat::{loaded_date_str}::{loaded_hour}"
        ask_input_key = f"phase4_ask_input::{loaded_date_str}::{loaded_hour}"
        clear_next_key = f"{ask_input_key}__clear_next"

        if p4_key not in st.session_state:
            with st.spinner("Preparing Phase 4 facts and summary..."):
                st.session_state[p4_key] = get_or_build_phase4(run_id, force=False, use_llm_summary=True)

        p4 = st.session_state[p4_key]
        facts_path = Path(p4["files"]["facts_pack_json"])
        summary_path = Path(p4["files"]["simple_summary_md"])
        st.markdown(
            "<span style='background:#e3f2fd;color:#0d47a1;padding:4px 8px;border-radius:6px;font-size:0.9em;'>Answers are based on saved forecast data.</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Context: You are asking about run `{run_id}`.")
        if summary_path.exists():
            with st.expander("Simple Summary", expanded=False):
                st.markdown(summary_path.read_text(encoding="utf-8"))

        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        history: list[dict[str, Any]] = st.session_state[chat_key]
        user_count = sum(1 for m in history if m.get("role") == "user")
        st.caption(f"Messages used: {user_count}/5")

        st.markdown("### Forecast Assistant")
        st.markdown("<div class='muted-note'>Ask plain-language questions and get answers backed by saved artifacts.</div>", unsafe_allow_html=True)
        chat_box = st.container()
        with chat_box:
            for msg in history:
                role = msg.get("role", "assistant")
                with st.chat_message("user" if role == "user" else "assistant"):
                    st.markdown(msg.get("text", ""))
                    if role == "assistant" and msg.get("sources"):
                        with st.expander("Sources", expanded=False):
                            src_df = pd.DataFrame(msg.get("sources", []))
                            if not src_df.empty:
                                st.dataframe(src_df, use_container_width=True, hide_index=True)
                    previews = msg.get("tool_previews", []) if role == "assistant" else []
                    for i, tp in enumerate(previews):
                        with st.expander(f"Tool Preview: {tp.get('tool_name', f'tool_{i}')}", expanded=False):
                            if tp.get("errors"):
                                st.warning("; ".join(tp.get("errors", [])))
                            st.markdown(tp.get("preview_markdown", ""))

        disabled = user_count >= 5
        if disabled:
            st.info("Message limit reached for this run (5). Clear chat to ask more.")

        if st.session_state.get(clear_next_key, False):
            st.session_state[ask_input_key] = ""
            st.session_state[clear_next_key] = False

        q = st.text_input(
            "Your Question",
            key=ask_input_key,
            placeholder="Example: Why is risk severe? Is tomorrow stable? Which hours are most risky?",
            disabled=disabled,
        )
        if q.strip():
            with st.expander("Planned tools", expanded=False):
                planned = plan_tools(q.strip(), allow_llm_fallback=False)
                if planned:
                    st.json(planned)
                else:
                    st.write("No deterministic tool route matched; planner fallback may be used.")
        col_ask, col_clear = st.columns([1, 1])
        with col_ask:
            if st.button("Get Answer", key=f"send_{ask_input_key}", disabled=disabled or not q.strip()):
                question = q.strip()
                history.append({"role": "user", "text": question})
                ans = answer_question(run_id, question)
                history.append(
                    {
                        "role": "assistant",
                        "text": ans.get("final_markdown", ""),
                        "sources": ans.get("sources", []),
                        "status": ans.get("status"),
                        "tool_previews": ans.get("tool_results", []),
                        "plan": ans.get("plan", []),
                    }
                )
                st.session_state[chat_key] = history
                st.session_state[clear_next_key] = True
                st.rerun()
        with col_clear:
            if st.button("Clear Chat", key=f"clear_{ask_input_key}"):
                st.session_state[chat_key] = []
                st.session_state[clear_next_key] = True
                st.rerun()

        st.subheader("Phase 4 Files")
        if facts_path.exists():
            st.download_button(
                "Download facts_pack.json",
                data=facts_path.read_bytes(),
                file_name=facts_path.name,
                mime="application/json",
            )
        if summary_path.exists():
            st.download_button(
                "Download simple_summary.md",
                data=summary_path.read_bytes(),
                file_name=summary_path.name,
                mime="text/markdown",
            )

    with tabs[6]:
        st.subheader("Saved Artifacts")
        st.markdown("<div class='muted-note'>Downloads grouped by phase for quick access.</div>", unsafe_allow_html=True)

        forecast_csv_path = Path(result["files"]["forecast_csv"])
        forecast_json_path = Path(result["files"]["forecast_json"])
        st.markdown("### Phase 1")
        if forecast_csv_path.exists():
            st.download_button(
                "Download forecast.csv",
                data=forecast_csv_path.read_bytes(),
                file_name=forecast_csv_path.name,
                mime="text/csv",
            )
        if forecast_json_path.exists():
            st.download_button(
                "Download forecast.json",
                data=forecast_json_path.read_bytes(),
                file_name=forecast_json_path.name,
                mime="application/json",
            )
        loaded_date_str = loaded_init.strftime("%Y-%m-%d")
        loaded_hour = loaded_init.strftime("%H")
        phase2_key = f"phase2::{loaded_date_str}::{loaded_hour}"
        p2 = st.session_state.get(phase2_key)
        st.markdown("### Phase 2")
        if p2 and p2.get("files"):
            phase2_summary_path = Path(p2["files"].get("phase2_summary_json", ""))
            phase2_matrix_path = Path(p2["files"].get("forecast_matrix_csv", ""))
            if phase2_summary_path.exists():
                st.download_button(
                    "Download phase2_summary.json",
                    data=phase2_summary_path.read_bytes(),
                    file_name=phase2_summary_path.name,
                    mime="application/json",
                )
            if phase2_matrix_path.exists():
                st.download_button(
                    "Download phase2 forecast_matrix.csv",
                    data=phase2_matrix_path.read_bytes(),
                    file_name=phase2_matrix_path.name,
                    mime="text/csv",
                )
        loaded_date_str = loaded_init.strftime("%Y-%m-%d")
        loaded_hour = loaded_init.strftime("%H")
        p25_key = f"phase25::{loaded_date_str}::{loaded_hour}"
        p25 = st.session_state.get(p25_key)
        st.markdown("### Phase 2.5")
        if p25 and p25.get("files"):
            p25_summary = Path(p25["files"].get("phase25_summary_json", ""))
            p25_fit = Path(p25["files"].get("attribution_fit_json", ""))
            p25_pairs = Path(p25["files"].get("revision_pairs_csv", ""))
            p25_risk = Path(p25["files"].get("joint_ops_risk_csv", ""))
            if p25_summary.exists():
                st.download_button(
                    "Download phase25_summary.json",
                    data=p25_summary.read_bytes(),
                    file_name=p25_summary.name,
                    mime="application/json",
                )
            if p25_fit.exists():
                st.download_button(
                    "Download attribution_fit.json",
                    data=p25_fit.read_bytes(),
                    file_name=p25_fit.name,
                    mime="application/json",
                )
            if p25_pairs.exists():
                st.download_button(
                    "Download revision_pairs.csv",
                    data=p25_pairs.read_bytes(),
                    file_name=p25_pairs.name,
                    mime="text/csv",
                )
            if p25_risk.exists():
                st.download_button(
                    "Download joint_ops_risk.csv",
                    data=p25_risk.read_bytes(),
                    file_name=p25_risk.name,
                    mime="text/csv",
                )
        p4_key = f"phase4::{loaded_date_str}::{loaded_hour}"
        p4 = st.session_state.get(p4_key)
        st.markdown("### Phase 4")
        if p4 and p4.get("files"):
            p4_facts = Path(p4["files"].get("facts_pack_json", ""))
            p4_summary = Path(p4["files"].get("simple_summary_md", ""))
            if p4_facts.exists():
                st.download_button(
                    "Download phase4 facts_pack.json",
                    data=p4_facts.read_bytes(),
                    file_name=p4_facts.name,
                    mime="application/json",
                )
            if p4_summary.exists():
                st.download_button(
                    "Download phase4 simple_summary.md",
                    data=p4_summary.read_bytes(),
                    file_name=p4_summary.name,
                    mime="text/markdown",
                )
        st.markdown("### Paths")
        with st.expander("Show artifact paths", expanded=False):
            for key, path_str in result["files"].items():
                st.write(f"`{key}`: `{path_str}`")


if __name__ == "__main__":
    main()
