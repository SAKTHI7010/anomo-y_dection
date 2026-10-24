"""Page 2 - Anomaly detection results (held-out test period)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data_access as D
import theme
from src import config as C
from src.event_evaluation import curve_metrics, event_evaluation, pointwise_metrics, pr_curve
from views.common import fmt, friendly_label_names, pct, sensor_title, table_fmt

MODELS = {
    "Isolation Forest (baseline, unsupervised)": ("if_score", "alert_if"),
    "XGBoost weak-label (alarm within 30 min)": ("xgb_prob", "alert_xgb"),
    "Autoencoder (optional, healthy-trained)": ("ae_error", "alert_ae"),
}
ALERT_COLOR = "#fb923c"
LABEL_COLOR = "#fb7185"


@st.cache_data(show_spinner=False)
def compute_eval(_scored: pd.DataFrame, token: str, score_col: str, alert_col: str, label: str, merge_gap: int, lookback: int):
    """Test-split point-wise and event-level evaluation of one model against one threshold-derived label."""
    te = (_scored["split"] == "test") & _scored[score_col].notna()
    y = _scored[f"label_{label}"]
    pw = pointwise_metrics(y[te].to_numpy(), _scored.loc[te, alert_col].to_numpy(), _scored.loc[te, score_col].to_numpy())
    ev = event_evaluation(y[te].reindex(_scored.index, fill_value=False), _scored[alert_col], te, merge_gap, lookback)
    crv = pr_curve(y[te].to_numpy(), _scored.loc[te, score_col].to_numpy())
    return pw, ev, crv


def render(scored: pd.DataFrame, summary: dict) -> None:
    theme.hero("Anomaly Detection Results", "Model alerts versus threshold-derived candidate anomalies on the chronologically held-out test period.",
               ["TIME-BASED SPLIT", "WEAK LABELS ≠ CONFIRMED FAULTS", "NO ACCURACY HEADLINES"])
    token = str(summary.get("finished", ""))
    cutoff = summary.get("split", {}).get("cutoff", "")
    names = friendly_label_names()

    avail = {k: v for k, v in MODELS.items() if scored[v[0]].notna().any()}
    c1, c2, c3, c4 = st.columns([2.2, 2.2, 1, 1])
    model_name = c1.selectbox("Model", list(avail), key="ar_model")
    label = c2.selectbox("Reference label", list(names), format_func=lambda k: names[k], key="ar_label")
    merge_gap = c3.number_input("Event merge gap (min)", 5, 240, 30, 5, key="ar_gap")
    lookback = c4.number_input("Early-warning window (min)", 15, 480, 120, 15, key="ar_look")
    score_col, alert_col = avail[model_name]

    pw, ev, crv = compute_eval(scored, token, score_col, alert_col, label, int(merge_gap), int(lookback))

    theme.note(f"Test period starts <b>{cutoff[:10]}</b>. Evaluated on <b>{pw['n']:,}</b> valid minutes with <b>{pw['positives']:,}</b> labelled minutes "
               f"({pw['positive_rate_pct']:.2f} %) in <b>{ev['label_events']}</b> events. "
               + ("<b>Very few positive events - treat all rates as indicative only.</b>" if ev["label_events"] < 30 else ""),
               "warn" if ev["label_events"] < 30 else "note")

    cards = st.columns(6)
    items = [
        ("Precision", pct(pw["precision"]), f"{pw['tp']} TP / {pw['fp']} FP"),
        ("Recall", pct(pw["recall"]), f"{pw['tp']} TP / {pw['fn']} FN"),
        ("F1", pct(pw["f1"]), "point-wise"),
        ("PR-AUC", fmt(pw.get("pr_auc"), 3), f"baseline {fmt(pw.get('pr_baseline'), 4)}"),
        ("Event recall", pct(ev["event_recall"]), f"{ev['events_detected']}/{ev['label_events']} events"),
        ("False alarms / week", fmt(ev["false_alarms_per_week"], 2), f"{ev['false_alarm_events']} alert events"),
    ]
    for col, (k, v, s) in zip(cards, items):
        col.markdown(theme.card(k, v, s), unsafe_allow_html=True)
    st.write("")

    t1, t2, t3, t4 = st.tabs(["Scores & thresholds", "Precision–Recall & confusion", "Events & lead time", "Model comparison & imbalance (SMOTE)"])

    with t1:
        ch = st.selectbox("Overlay sensor with DCS threshold", C.CHANNELS, format_func=sensor_title, key="ar_ch")
        te = scored[(scored["split"] == "test") & scored[score_col].notna()]
        if te.empty:
            st.info("No scored rows in the test period.")
        else:
            step = max(1, int(np.ceil(len(te) / 5000)))
            sc = te[score_col].groupby(np.arange(len(te)) // step).max()
            idx = te.index[::step][: len(sc)]
            val = te[ch].groupby(np.arange(len(te)) // step).max()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.07,
                                subplot_titles=(f"Anomaly score - {model_name.split(' (')[0]}", f"{sensor_title(ch)} with DCS threshold"))
            fig.add_trace(go.Scatter(x=idx, y=sc.to_numpy(), mode="lines", name="Anomaly score", line=dict(color="#22d3ee", width=1.2)), row=1, col=1)
            thr_val = float(te.loc[te[alert_col], score_col].min()) if te[alert_col].any() else None
            if thr_val is not None:
                fig.add_hline(y=thr_val, line=dict(color=ALERT_COLOR, dash="dash"), row=1, col=1, annotation_text="alert threshold", annotation_font_color=ALERT_COLOR)
            lab = te[te[f"label_{label}"]]
            if not lab.empty:
                fig.add_trace(go.Scatter(x=lab.index, y=lab[score_col], mode="markers", name="Threshold-derived candidate anomaly",
                                         marker=dict(color=LABEL_COLOR, size=6, symbol="x")), row=1, col=1)
            fig.add_trace(go.Scatter(x=idx, y=val.to_numpy(), mode="lines", name=sensor_title(ch), line=dict(color=theme.SENSOR_COLORS[ch], width=1.1)), row=2, col=1)
            fig.add_hline(y=C.THRESHOLDS[ch], line=dict(color="#f87171", dash="dash"), row=2, col=1,
                          annotation_text=f"DCS ≥ {C.THRESHOLDS[ch]:g}", annotation_font_color="#f87171")
            al = te[te[alert_col]]
            if not al.empty:
                fig.add_trace(go.Scatter(x=al.index, y=al[ch], mode="markers", name="Model detection",
                                         marker=dict(color=ALERT_COLOR, size=5, symbol="diamond-open")), row=2, col=1)
            theme.style_fig(fig, height=620)
            st.plotly_chart(fig, use_container_width=True, key="ar_scores")
            st.caption("Plot aggregates to the max per bin so short spikes are never hidden. Observed readings, weak labels and model detections use distinct markers.")

    with t2:
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            if crv is None:
                st.info("Precision–Recall curve is undefined: the held-out period has no positive reference labels for this label.")
            else:
                p, r, _ = crv
                fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"PR-AUC {fmt(pw.get('pr_auc'), 3)}", line=dict(color="#22d3ee", width=2.4), fill="tozeroy", fillcolor="rgba(34,211,238,0.08)"))
                fig.add_hline(y=pw.get("pr_baseline") or 0, line=dict(color="#64748b", dash="dot"), annotation_text="no-skill baseline")
                fig.update_xaxes(title="Recall", range=[0, 1.01])
                fig.update_yaxes(title="Precision", range=[0, 1.02])
                theme.style_fig(fig, height=380, title="Precision–Recall curve (test period)")
                st.plotly_chart(fig, use_container_width=True, key="ar_pr")
        with b:
            z = [[pw["tn"], pw["fp"]], [pw["fn"], pw["tp"]]]
            fig = go.Figure(go.Heatmap(z=np.log10(np.array(z) + 1), x=["Pred normal", "Pred alert"], y=["Actual normal", "Actual candidate"],
                                       text=[[f"{v:,}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=18),
                                       colorscale=[[0, "#0f172a"], [1, "#0891b2"]], showscale=False, hoverinfo="skip"))
            fig.update_yaxes(autorange="reversed")
            theme.style_fig(fig, height=380, title="Confusion matrix at the chosen alert threshold")
            st.plotly_chart(fig, use_container_width=True, key="ar_cm")
        st.caption("Accuracy is intentionally not shown - with rare positives it is dominated by true negatives.")

    with t3:
        pe = ev["_per_event"]
        a, b = st.columns([1.3, 1])
        with a:
            fig = go.Figure()
            if not pe.empty:
                fig.add_trace(go.Bar(base=pe["start"], x=(pe["duration_min"].clip(lower=5) * 60_000), y=["Threshold-derived events"] * len(pe), orientation="h",
                                     marker_color=LABEL_COLOR, name="Candidate anomaly events", hovertext=pe["start"].astype(str)))
            al_ev = ev.get("_false_alarms", pd.DataFrame())
            from src.event_evaluation import merge_events
            all_al = merge_events(scored.loc[scored["split"] == "test", alert_col], int(merge_gap))
            if not all_al.empty:
                fig.add_trace(go.Bar(base=all_al["start"], x=(all_al["duration_min"].clip(lower=5) * 60_000), y=["Model alert events"] * len(all_al), orientation="h",
                                     marker_color=ALERT_COLOR, name="Model alert events"))
            fig.update_layout(barmode="overlay", bargap=0.45)
            theme.style_fig(fig, height=250, title="Event timeline - model alerts vs candidate anomalies")
            st.plotly_chart(fig, use_container_width=True, key="ar_timeline")
        with b:
            lead = pe["lead_min"].dropna() if not pe.empty else pd.Series(dtype=float)
            if lead.empty:
                st.info("Lead-time histogram is not available: no label event was detected, so no lead time can be measured.")
            else:
                fig = go.Figure(go.Histogram(x=lead, nbinsx=20, marker_color="#a78bfa"))
                fig.add_vline(x=0, line=dict(color="#e2e8f0", dash="dash"))
                fig.update_xaxes(title="Lead time (min) - positive = alert BEFORE threshold alarm")
                fig.update_yaxes(title="Events")
                theme.style_fig(fig, height=250, title="Early-warning lead time per detected event")
                st.plotly_chart(fig, use_container_width=True, key="ar_lead")
        m = st.columns(4)
        m[0].metric("Events detected", f"{ev['events_detected']}/{ev['label_events']}")
        m[1].metric("Early warnings (lead > 0)", ev["events_early_warning"])
        m[2].metric("Median lead (min)", fmt(ev["median_lead_min"], 1))
        m[3].metric("Alert-event precision", pct(ev["alert_event_precision"]))
        st.caption(f"Definitions: events merge when gaps ≤ {merge_gap} min · detected if an alert occurs within [start − {lookback} min, end] · lead = event start − first alert · "
                   f"false alarm = alert event with no label event within [start − {lookback} min, end + 30 min] · rate over {ev['observed_weeks']:.1f} valid weeks.")
        if not pe.empty:
            show = pe.copy()
            show["start"] = show["start"].astype(str)
            show["end"] = show["end"].astype(str)
            show["first_alert"] = show["first_alert"].astype(str)
            st.dataframe(show, use_container_width=True, hide_index=True, height=240)

    with t4:
        st.markdown("##### Isolation Forest - configurations compared on the test period")
        ifr = D.load_metrics_csv("isolation_forest_results.csv")
        if ifr.empty:
            st.info("Run the backend pipeline to generate this table.")
        else:
            d = ifr[ifr["label"] == label][["features", "train_rows", "contamination", "pw_predicted_positives", "pw_precision", "pw_recall", "pw_pr_auc",
                                            "ev_events_detected", "ev_label_events", "ev_median_lead_min", "ev_false_alarms_per_week"]].copy()
            d.columns = ["features", "training rows", "contamination", "alerts (min)", "precision", "recall", "PR-AUC", "events hit", "events", "median lead (min)", "false alarms/wk"]
            st.dataframe(table_fmt(d, pct_cols=("precision", "recall"), float_cols=("PR-AUC", "median lead (min)", "false alarms/wk")), use_container_width=True, hide_index=True)
            st.caption("Default (fixed before looking at test results): 'selected' features · all valid training rows · contamination 1 %.")

        st.markdown("##### XGBoost - class-imbalance handling (none vs class weight vs **SMOTE**)")
        xr = D.load_metrics_csv("xgboost_results.csv")
        if xr.empty:
            st.info("Supervised results unavailable (xgboost/imbalanced-learn missing or pipeline not run).")
        else:
            x = xr[["label", "horizon_min", "variant", "synthetic_samples", "val_pr_auc", "pw_precision", "pw_recall", "pw_f1", "pw_pr_auc", "ev_events_detected", "ev_label_events", "ev_false_alarms_per_week"]].copy()
            x.columns = ["label", "horizon (min)", "imbalance handling", "synthetic samples", "val PR-AUC", "precision", "recall", "F1", "test PR-AUC", "events hit", "events", "false alarms/wk"]
            x["label"] = x["label"].map(lambda k: names.get(k, k))
            st.dataframe(table_fmt(x, pct_cols=("precision", "recall", "F1"), float_cols=("val PR-AUC", "test PR-AUC", "false alarms/wk")), use_container_width=True, hide_index=True)
            theme.note("SMOTE is applied <b>only to the training portion after the chronological split</b> (never to validation/test, never across the time boundary). "
                       "It adds synthetic minority rows to the training set; evaluation always uses real, unaltered data. 'Test PR-AUC' is the fair comparison metric.")
        ae = D.load_metrics_csv("autoencoder_results.csv")
        if not ae.empty:
            st.markdown("##### Autoencoder (99th-percentile reconstruction-error threshold)")
            a = ae[["label", "pw_predicted_positives", "pw_precision", "pw_recall", "pw_pr_auc", "ev_events_detected", "ev_label_events", "ev_false_alarms_per_week"]].copy()
            a.columns = ["label", "alerts (min)", "precision", "recall", "PR-AUC", "events hit", "events", "false alarms/wk"]
            a["label"] = a["label"].map(lambda k: names.get(k, k))
            st.dataframe(table_fmt(a, pct_cols=("precision", "recall"), float_cols=("PR-AUC", "false alarms/wk")), use_container_width=True, hide_index=True)
