"""Page 1 - Real-time-style health monitor (replays recorded data as of a chosen time)."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import theme
from src import config as C
from src.health_score import forecast_linear, health_state
from views.common import fmt, latest_valid, period_of, sensor_title, smooth

ALERT_COLOR = "#fb923c"
LABEL_COLOR = "#fb7185"


def gauge(value: float | None, state: str, colour: str) -> go.Figure:
    v = 0 if value is None or not np.isfinite(value) else float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={"font": {"size": 54, "color": colour}, "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickwidth": 1},
            "bar": {"color": colour, "thickness": 0.28},
            "bgcolor": "rgba(15,23,42,0.4)", "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "rgba(239,68,68,0.28)"},
                {"range": [20, 40], "color": "rgba(249,115,22,0.26)"},
                {"range": [40, 70], "color": "rgba(234,179,8,0.22)"},
                {"range": [70, 100], "color": "rgba(34,197,94,0.22)"},
            ],
        },
        title={"text": "Equipment health score (prototype)", "font": {"size": 14, "color": "#94a3b8"}},
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"))
    return fig


def sensor_grid(w: pd.DataFrame, smooth_min: int, show_alerts: bool) -> go.Figure:
    titles = [sensor_title(c) for c in C.CHANNELS]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, shared_xaxes=True, vertical_spacing=0.14, horizontal_spacing=0.07)
    for i, ch in enumerate(C.CHANNELS):
        r, c = divmod(i, 2)
        r, c = r + 1, c + 1
        y = smooth(w[ch], smooth_min)
        fig.add_trace(go.Scatter(x=w.index, y=y, mode="lines", name="Observed reading", legendgroup="obs",
                                 line=dict(color=theme.SENSOR_COLORS[ch], width=1.8), showlegend=(i == 0),
                                 hovertemplate="%{x|%b %d %H:%M}<br>%{y:.2f}<extra></extra>"), row=r, col=c)
        thr = C.THRESHOLDS[ch]
        fig.add_hline(y=thr, line=dict(color="#f87171", dash="dash", width=1), row=r, col=c,
                      annotation_text=f"DCS ≥ {thr:g}", annotation_font=dict(size=10, color="#f87171"), annotation_position="top left")
        if show_alerts:
            ex = w[w[f"exceed_{ch}"]]
            if not ex.empty:
                fig.add_trace(go.Scatter(x=ex.index, y=ex[ch], mode="markers", name="Threshold-derived candidate anomaly",
                                         legendgroup="lab", marker=dict(color=LABEL_COLOR, symbol="x", size=7),
                                         showlegend=not any(t.legendgroup == "lab" for t in fig.data)), row=r, col=c)
            al = w[w["alert_if"] & w[ch].notna()]
            if not al.empty:
                fig.add_trace(go.Scatter(x=al.index, y=al[ch], mode="markers", name="Isolation Forest detection",
                                         legendgroup="mod", marker=dict(color=ALERT_COLOR, symbol="diamond-open", size=7, line=dict(width=1.5)),
                                         showlegend=not any(t.legendgroup == "mod" for t in fig.data)), row=r, col=c)
    theme.style_fig(fig, height=560)
    fig.update_annotations(font=dict(size=12, color="#cbd5e1"))
    fig.update_layout(legend=dict(y=1.06))
    return fig


def render(scored: pd.DataFrame, summary: dict) -> None:
    theme.hero("Health Monitor", "Real-time-style view of the Sulzer CFG-42 centrifuge - replaying recorded sensor history.",
               ["HISTORICAL PLAYBACK", "NOT LIVE TELEMETRY", "PROTOTYPE HEALTH SCORE"])
    cutoff = summary.get("split", {}).get("cutoff")

    valid_idx = scored.index[scored["row_valid"]]
    if len(valid_idx) == 0:
        st.error("No valid sensor rows found in the scored dataset.")
        return
    lo, hi = valid_idx.min().to_pydatetime(), valid_idx.max().to_pydatetime()

    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.4])
    with c1:
        asof = st.slider("As-of time (recorded data)", min_value=lo, max_value=hi, value=hi, step=timedelta(hours=1),
                         format="YYYY-MM-DD HH:mm", key="hm_asof")
    with c2:
        smooth_min = st.slider("Display smoothing (min)", 1, 30, 5, help="Causal trailing mean applied to plotted traces only. Detection and scores are never smoothed.", key="hm_smooth")
    with c3:
        show_alerts = st.checkbox("Show anomaly markers", True, key="hm_alerts")
    with c4:
        show_fc = st.checkbox("24 h trend estimate", False, key="hm_fc", help="Simple linear extrapolation of the last 24 h of hourly health. Estimate only.")

    end = pd.Timestamp(asof)
    w = scored.loc[end - pd.Timedelta(hours=24): end]
    cur = latest_valid(w)

    left, right = st.columns([1.05, 1.6])
    with left:
        if cur is None:
            state, colour, interp = "No data", "#64748b", "No valid sensor data in the last 24 h (data gap / sensor outage)."
            h = None
        else:
            h = float(cur["health"])
            state, colour, interp = health_state(h)
        st.plotly_chart(gauge(h, state, colour), use_container_width=True, key="hm_gauge")
        st.markdown(
            f'<div style="text-align:center"><span class="state-pill" style="color:{colour};border-color:{colour};'
            f'background:{colour}1f">{state.upper()}</span><div style="color:#94a3b8;margin-top:.6rem;font-size:.9rem">{interp}</div>'
            f'<div style="color:#64748b;margin-top:.3rem;font-size:.78rem">Reading time: {cur.name if cur is not None else "–"} · {period_of(end, cutoff)}</div></div>',
            unsafe_allow_html=True)
    with right:
        cols = st.columns(4)
        for i, ch in enumerate(C.CHANNELS):
            info = C.CHANNEL_INFO[ch]
            v = None if cur is None else float(cur[ch])
            over = v is not None and v >= info["threshold"]
            sub = f"{info['unit']} · DCS ≥ {info['threshold']:g}" + (" · ABOVE" if over else "")
            cols[i].markdown(theme.card(info["label"], fmt(v, 1), sub, "#f87171" if over else None), unsafe_allow_html=True)
        n_alert = int(w["alert_if"].sum())
        n_lab = int(w["label_strong_any"].sum())
        n_vib = int(w["label_weak_vibration"].sum())
        c = st.columns(3)
        c[0].markdown(theme.card("Model detections (24 h)", f"{n_alert:,}", "Isolation Forest minutes flagged", ALERT_COLOR if n_alert else None), unsafe_allow_html=True)
        c[1].markdown(theme.card("Threshold exceedances", f"{n_lab:,}", "any channel ≥ DCS value (minutes)", LABEL_COLOR if n_lab else None), unsafe_allow_html=True)
        c[2].markdown(theme.card("Vibration ≥ 12 mm/s", f"{n_vib:,}", "primary weak label (minutes)", LABEL_COLOR if n_vib else None), unsafe_allow_html=True)
        theme.legend([("Observed reading", "#22d3ee"), ("Threshold-derived candidate anomaly (weak label)", LABEL_COLOR),
                      ("Model detection (Isolation Forest)", ALERT_COLOR), ("Forecast estimate", "#a78bfa")])

    if w[C.CHANNELS].notna().any().any():
        st.plotly_chart(sensor_grid(w, smooth_min, show_alerts), use_container_width=True, key="hm_sensors")
    else:
        theme.note("No valid sensor readings in this 24 h window - the equipment was offline or the channel data is missing here.", "warn")

    # Health trend + optional forecast
    hs = smooth(w["health"], smooth_min)
    fig = go.Figure()
    for lo_b, hi_b, col in ((70, 100, "rgba(34,197,94,0.07)"), (40, 70, "rgba(234,179,8,0.07)"), (20, 40, "rgba(249,115,22,0.08)"), (0, 20, "rgba(239,68,68,0.09)")):
        fig.add_hrect(y0=lo_b, y1=hi_b, fillcolor=col, line_width=0)
    fig.add_trace(go.Scatter(x=w.index, y=hs, mode="lines", name="Health score (observed)", line=dict(color="#34d399", width=2)))
    if show_fc:
        fc = forecast_linear(w["health"].where(w["row_valid"]))
        if fc is None:
            theme.note("Not enough valid hourly data in the last 24 h to build a trend estimate.", "warn")
        else:
            fig.add_trace(go.Scatter(x=list(fc["ts"]) + list(fc["ts"])[::-1], y=list(fc["hi"]) + list(fc["lo"])[::-1], fill="toself",
                                     fillcolor="rgba(167,139,250,0.18)", line=dict(width=0), name="~95 % band (estimate)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc["ts"], y=fc["mean"], mode="lines", name="Linear trend estimate", line=dict(color="#a78bfa", dash="dot", width=2)))
            st.caption(f"Trend slope {fc['slope_per_h']:+.2f} health points/hour. This is a simple extrapolation, **not** a validated failure prediction.")
    fig.update_yaxes(range=[0, 100], title="Health (0–100)")
    theme.style_fig(fig, height=300, title="Health score - last 24 h")
    st.plotly_chart(fig, use_container_width=True, key="hm_health")
    theme.note(C.DISCLAIMER, "warn")
