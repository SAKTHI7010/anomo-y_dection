"""Page 3 - Historical replay (recorded data playback - NOT live hardware telemetry)."""
from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import theme
from src import config as C
from src.health_score import states_series
from src.historical_replay import downsample, replay_summary, window
from views.common import fmt, sensor_title, smooth

ALERT_COLOR = "#fb923c"
LABEL_COLOR = "#fb7185"
STATE_COLORS = {"Healthy": "#22c55e", "Attention": "#eab308", "Warning": "#f97316", "Critical": "#ef4444", "No data": "#64748b"}


def build_figure(w: pd.DataFrame, smooth_min: int, alert_col: str, alert_name: str) -> go.Figure:
    d = downsample(w)
    ds = len(d) != len(w)
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.26, 0.15, 0.15, 0.15, 0.15],
                        subplot_titles=("Health score (prototype)", *[sensor_title(c) for c in C.CHANNELS]))
    for lo, hi, col in ((70, 100, "rgba(34,197,94,0.07)"), (40, 70, "rgba(234,179,8,0.07)"), (20, 40, "rgba(249,115,22,0.08)"), (0, 20, "rgba(239,68,68,0.09)")):
        fig.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=smooth(d["health"], 1 if ds else smooth_min), mode="lines", name="Health score", line=dict(color="#34d399", width=1.8)), row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=1, col=1)
    for i, ch in enumerate(C.CHANNELS):
        r = i + 2
        fig.add_trace(go.Scatter(x=d.index, y=smooth(d[ch], 1 if ds else smooth_min), mode="lines", name="Observed reading", legendgroup="obs",
                                 showlegend=(i == 0), line=dict(color=theme.SENSOR_COLORS[ch], width=1.3)), row=r, col=1)
        fig.add_hline(y=C.THRESHOLDS[ch], line=dict(color="#f87171", dash="dash", width=1), row=r, col=1)
        ex = d[d[f"exceed_{ch}"]]
        if not ex.empty:
            fig.add_trace(go.Scatter(x=ex.index, y=ex[ch], mode="markers", name="Threshold-derived candidate anomaly", legendgroup="lab",
                                     showlegend=not any(t.legendgroup == "lab" for t in fig.data), marker=dict(color=LABEL_COLOR, symbol="x", size=6)), row=r, col=1)
        al = d[d[alert_col] & d[ch].notna()]
        if not al.empty:
            fig.add_trace(go.Scatter(x=al.index, y=al[ch], mode="markers", name=f"Model detection ({alert_name})", legendgroup="mod",
                                     showlegend=not any(t.legendgroup == "mod" for t in fig.data), marker=dict(color=ALERT_COLOR, symbol="diamond-open", size=6, line=dict(width=1.3))), row=r, col=1)
    theme.style_fig(fig, height=860)
    fig.update_annotations(font=dict(size=11, color="#cbd5e1"))
    fig.update_layout(legend=dict(y=1.03))
    return fig


def render(scored: pd.DataFrame, summary: dict) -> None:
    theme.hero("Historical Replay", "Select a recorded time range, replay the sensor history and review health variation and detected events.",
               ["HISTORICAL PLAYBACK", "NOT LIVE HARDWARE TELEMETRY"])
    theme.note("This page plays back <b>recorded</b> historical data. It is not connected to any live plant source.")
    idx = scored.index
    lo, hi = idx.min().to_pydatetime(), idx.max().to_pydatetime()
    default_start = max(lo, pd.Timestamp("2023-07-28").to_pydatetime())
    default_end = min(hi, default_start + timedelta(days=3))

    a, b, c = st.columns([3, 1.3, 1.5])
    with a:
        rng = st.slider("Time range", min_value=lo, max_value=hi, value=(default_start, default_end), step=timedelta(hours=1),
                        format="YYYY-MM-DD HH:mm", key="hr_range")
    with b:
        smooth_min = st.slider("Display smoothing (min)", 1, 60, 5, key="hr_smooth", help="Trailing mean on plotted traces only (applied when the range is short enough to draw every minute).")
    with c:
        model = st.selectbox("Detections shown", ["Isolation Forest", "XGBoost", "Autoencoder"], key="hr_model")
    alert_col = {"Isolation Forest": "alert_if", "XGBoost": "alert_xgb", "Autoencoder": "alert_ae"}[model]

    start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    w = window(scored, start, end)
    if w.empty or not w["row_valid"].any():
        theme.note("The selected range contains no valid sensor readings (data gap / sensor outage). Choose another range.", "warn")
        return
    if not w[alert_col].any() and model != "Isolation Forest" and not scored[alert_col].any():
        theme.note(f"{model} outputs are not available in the scored dataset (optional package not installed or route skipped).", "warn")

    # Replay cursor -------------------------------------------------------------------------------
    total_min = max(1, int((end - start) / pd.Timedelta(minutes=1)))
    st.session_state.setdefault("hr_cursor", 100)
    cc1, cc2 = st.columns([5, 1])
    with cc2:
        play = st.button("▶ Play replay", key="hr_play", use_container_width=True)
    with cc1:
        pct_pos = st.slider("Replay position (%)", 1, 100, key="hr_cursor", help="Reveal the recorded history up to this point in the selected range.")
    cursor_ts = start + pd.Timedelta(minutes=int(total_min * pct_pos / 100))
    slot = st.empty()
    ph_sum = st.empty()

    def draw(upto: pd.Timestamp, key: str) -> None:
        ww = w.loc[:upto]
        slot.plotly_chart(build_figure(ww, smooth_min, alert_col, model), use_container_width=True, key=key)
        s = replay_summary(ww, alert_col)
        with ph_sum.container():
            m = st.columns(5)
            m[0].metric("Replay time", str(upto)[:16])
            m[1].metric("Mean health", fmt(s.get("health_mean"), 1))
            m[2].metric("Min health", fmt(s.get("health_min"), 1))
            m[3].metric("Detected events", s.get("alert_events", 0))
            m[4].metric("Valid minutes", f"{s.get('valid_rows', 0):,}")

    if play:
        frames = 24
        for i in range(1, frames + 1):
            upto = start + pd.Timedelta(minutes=int(total_min * i / frames))
            draw(upto, f"hr_play_{i}")
            time.sleep(0.12)
        cursor_ts = end
    draw(cursor_ts if not play else end, "hr_final")

    # State distribution + event table ----------------------------------------------------------------
    ww = w.loc[:cursor_ts] if not play else w
    st_counts = states_series(ww.loc[ww["row_valid"], "health"]).value_counts()
    t1, t2 = st.tabs(["Health-state time share", "Detected events in range"])
    with t1:
        if st_counts.empty:
            st.info("No valid health values in this range.")
        else:
            order = [s for s in ("Healthy", "Attention", "Warning", "Critical") if s in st_counts.index]
            fig = go.Figure(go.Bar(x=order, y=[st_counts[s] for s in order], marker_color=[STATE_COLORS[s] for s in order],
                                   text=[f"{st_counts[s]:,} min" for s in order], textposition="outside"))
            theme.style_fig(fig, height=300, title="Minutes in each prototype health state")
            fig.update_yaxes(title="Minutes")
            st.plotly_chart(fig, use_container_width=True, key="hr_states")
    with t2:
        s = replay_summary(ww, alert_col)
        ev = s.get("events")
        if ev is None or len(ev) == 0:
            st.info(f"No {model} alert events in this range.")
        else:
            show = ev.copy()
            show["start"], show["end"] = show["start"].astype(str), show["end"].astype(str)
            st.dataframe(show, use_container_width=True, hide_index=True)
    theme.note(C.DISCLAIMER, "warn")
