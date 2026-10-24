"""Page 4 - Feature importance / model explanation (+ optional torque-temperature residual twin)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_access as D
import theme
from src import config as C
from views.common import fmt, pct, sensor_title, smooth


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def shap_section(summary: dict) -> None:
    xd = summary.get("xgb_default")
    sh = D.load_shap()
    st.markdown("### Supervised route - XGBoost SHAP")
    if not xd:
        theme.note("The supervised XGBoost route was not run (xgboost / imbalanced-learn missing or skipped), so there are **no SHAP results**. "
                   "Nothing is fabricated here.", "warn")
        return
    if sh.empty:
        theme.note("SHAP values are unavailable (the `shap` package was not installed when the pipeline ran). Install it and re-run "
                   "`python backend/run_pipeline.py` to enable this section.", "warn")
        return
    theme.note(f"Model: XGBoost · target: <b>{xd['label'].replace('_', ' ')}</b> alarm within <b>{xd['horizon_min']} min</b> · imbalance handling: <b>{xd['variant']}</b>. "
               f"SHAP computed on {len(sh):,} held-out test rows (all positives + random negatives; <b>sample-based - not the full test set</b>). "
               "Because labels are threshold-derived, the model explains *which signals precede the plant threshold*, not true mechanical failure causes.")
    sc = [c for c in sh.columns if c.startswith("shap__")]
    vc = [c.replace("shap__", "val__") for c in sc]
    S = sh[sc].to_numpy(float)
    names = [c.replace("shap__", "") for c in sc]
    imp = np.abs(S).mean(axis=0)
    order = np.argsort(imp)[::-1][:15]

    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Bar(x=imp[order][::-1], y=[_pretty(names[i]) for i in order][::-1], orientation="h",
                               marker=dict(color=imp[order][::-1], colorscale=[[0, "#164e63"], [1, "#22d3ee"]])))
        fig.update_xaxes(title="mean |SHAP value|")
        theme.style_fig(fig, height=470, title="Global feature importance (top 15)")
        st.plotly_chart(fig, use_container_width=True, key="ex_imp")
    with b:
        fig = go.Figure()
        rng = np.random.default_rng(1)
        for rank, i in enumerate(order[:12][::-1]):
            v = sh[vc[i]].to_numpy(float)
            lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
            norm = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
            fig.add_trace(go.Scattergl(x=S[:, i], y=rank + rng.uniform(-0.28, 0.28, len(v)), mode="markers", showlegend=False,
                                       marker=dict(size=4, color=norm, colorscale=[[0, "#38bdf8"], [0.5, "#a78bfa"], [1, "#fb7185"]], opacity=0.75),
                                       hovertemplate=f"{_pretty(names[i])}<br>SHAP %{{x:.3f}}<extra></extra>"))
        fig.update_yaxes(tickmode="array", tickvals=list(range(12)), ticktext=[_pretty(names[i]) for i in order[:12][::-1]])
        fig.update_xaxes(title="SHAP value (impact on predicted risk)")
        theme.style_fig(fig, height=470, title="SHAP summary (colour = feature value, blue low → red high)")
        st.plotly_chart(fig, use_container_width=True, key="ex_summary")

    st.markdown("##### Local explanation")
    pos = sh.index[sh["target"] == 1]
    choices = {"Highest-risk positive examples": pos, "Highest predicted-risk rows": sh.sort_values("prob", ascending=False).index}
    kind = st.radio("Pick from", list(choices), horizontal=True, key="ex_kind")
    pool = list(choices[kind][:200]) if len(choices[kind]) else list(sh.index[:50])
    ts = st.selectbox("Sample timestamp", pool, format_func=lambda t: f"{t}  ·  risk {sh.loc[t, 'prob']:.3f}", key="ex_ts")
    row = sh.loc[ts]
    loc = pd.Series(row[sc].to_numpy(float), index=names)
    top = loc.reindex(loc.abs().sort_values(ascending=False).index)[:12]
    fig = go.Figure(go.Bar(x=top.values[::-1], y=[f"{_pretty(n)} = {row['val__' + n]:.2f}" for n in top.index][::-1], orientation="h",
                           marker_color=["#fb7185" if v > 0 else "#38bdf8" for v in top.values[::-1]]))
    fig.update_xaxes(title="SHAP contribution (red raises risk, blue lowers)")
    theme.style_fig(fig, height=430, title=f"Why this minute scored {row['prob']:.3f} · {ts}")
    st.plotly_chart(fig, use_container_width=True, key="ex_local")


def isolation_diagnostics(scored: pd.DataFrame, summary: dict) -> None:
    st.markdown("### Unsupervised route - Isolation Forest diagnostics")
    theme.note("Isolation Forest has no native additive explanations, so <b>no SHAP is claimed</b>. Instead, this view shows each sensor's deviation from its "
               "<b>training-period normal operating level</b> (robust z-score) at the chosen minute - a supported, model-independent diagnostic.")
    tr = scored[(scored["split"] == "train") & scored["row_valid"] & scored["running"]]
    med = tr[C.CHANNELS].median()
    mad = (tr[C.CHANNELS].sub(med).abs()).median() * 1.4826
    te = scored[(scored["split"] == "test") & scored["alert_if"]]
    if te.empty:
        st.info("Isolation Forest raised no alerts in the test period.")
        return
    pool = te["if_score"].sort_values(ascending=False).index[:300]
    ts = st.selectbox("Alerted minute (highest anomaly scores first)", list(pool), format_func=lambda t: f"{t}  ·  score {scored.loc[t, 'if_score']:.3f}", key="ex_if_ts")
    r = scored.loc[ts]
    z = ((r[C.CHANNELS].astype(float) - med) / mad.replace(0, np.nan)).fillna(0)
    fig = go.Figure(go.Bar(x=[sensor_title(c) for c in C.CHANNELS], y=z.values, marker_color=["#fb7185" if v > 0 else "#38bdf8" for v in z.values],
                           text=[f"{r[c]:.1f}" for c in C.CHANNELS], textposition="outside"))
    fig.update_yaxes(title="Robust z-score vs training normal")
    theme.style_fig(fig, height=340, title=f"Sensor deviation at {ts} (labels show raw readings)")
    st.plotly_chart(fig, use_container_width=True, key="ex_if_dev")


def residual_section(scored: pd.DataFrame, summary: dict) -> None:
    st.markdown("### Optional digital-twin upgrade - torque → bearing-temperature residual")
    if "resid_flag_any" not in scored.columns:
        st.info("Residual model outputs are not available.")
        return
    theme.note("Expected bearing temperature is learned from torque (no ambient temperature exists in this dataset) on training-period data. "
               "residual = actual − expected. A <b>persistent positive residual</b> is only a condition <i>worth investigating</i> - it does not prove bearing degradation or poor lubrication.")
    ch = st.radio("Bearing", ["temp_solid", "temp_liquid"], format_func=sensor_title, horizontal=True, key="rs_ch")
    col, flag = f"resid_{ch}_persist", f"resid_{ch}_flag"
    thr = summary.get("residual_thresholds", {}).get(ch)
    d = scored[scored["row_valid"]]
    if col not in d or d[col].dropna().empty:
        st.info("No residual values for this bearing.")
        return
    daily = d[[col, flag]].resample("1D").agg({col: "median", flag: "mean"}).dropna(subset=[col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily[col], mode="lines", name="Daily median persistent residual (°C)", line=dict(color="#fbbf24", width=2)))
    if thr is not None:
        fig.add_hline(y=thr, line=dict(color="#fb923c", dash="dash"), annotation_text=f"flag threshold {thr:.1f} °C (training 99th pct)")
    cut = summary.get("split", {}).get("cutoff")
    if cut:
        fig.add_vline(x=pd.Timestamp(cut).timestamp() * 1000, line=dict(color="#94a3b8", dash="dot"), annotation_text="train | test")
    fig.update_yaxes(title="Residual (°C)")
    theme.style_fig(fig, height=330, title="Persistent temperature residual (60-min rolling median, daily summary)")
    st.plotly_chart(fig, use_container_width=True, key="rs_chart")

    rv = summary.get("residual_vs_labels", {})
    if rv:
        rows = []
        names_map = {"weak_vibration": "Vibration ≥ 12 mm/s", "strong_any": "Any channel ≥ threshold"}
        for k, v in rv.items():
            rows.append({"reference label": names_map.get(k, k), "label events": v["label_events"], "residual alert events": v["alert_events"],
                         "events hit": v["events_detected"], "false-alarm events": v["false_alarm_events"],
                         "false alarms / week": fmt(v["false_alarms_per_week"], 2), "median lead (min)": fmt(v["median_lead_min"], 1)})
        st.markdown("**Does it add value?** Comparison on the held-out test period:")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render(scored: pd.DataFrame, summary: dict) -> None:
    theme.hero("Model Explanation", "Which signals drive detections - SHAP for the supervised route, honest diagnostics for the unsupervised route.",
               ["SHAP ONLY WHERE SUPPORTED", "NO FABRICATED EXPLANATIONS"])
    t1, t2, t3 = st.tabs(["XGBoost · SHAP", "Isolation Forest · diagnostics", "Torque–temperature residual twin"])
    with t1:
        shap_section(summary)
    with t2:
        isolation_diagnostics(scored, summary)
    with t3:
        residual_section(scored, summary)
