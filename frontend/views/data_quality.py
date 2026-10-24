"""Page 5 - Data quality, methodology and limitations."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_access as D
import theme
from src import config as C
from views.common import fmt, friendly_label_names


def render(scored: pd.DataFrame, summary: dict) -> None:
    theme.hero("Data Quality & Method", "What the real Sulzer CFG-42 data looks like, what was cleaned, and how the numbers were produced.",
               ["REAL DATA ONLY", "EVERY DECISION LOGGED"])
    q = D.load_quality()
    if not q:
        st.info("Run the backend pipeline to generate the data-quality report.")
        return

    cols = st.columns(5)
    cards = [
        ("Records", f"{q['rows']:,}", f"{q['timestamp_min'][:10]} → {q['timestamp_max'][:10]}"),
        ("Timestamp gaps", f"{q['gap_count']}", f"{q['pct_intervals_exactly_1min']:.2f} % intervals = 1 min"),
        ("Duplicates", f"{q['duplicate_timestamps']}", "duplicate timestamps"),
        ("Rows w/ any missing", f"{max(q['missing_values'].values()):,}", "largest per-channel NaN count"),
        ("Fully valid rows", f"{int(scored['row_valid'].sum()):,}", f"{scored['row_valid'].mean()*100:.1f} % of grid"),
    ]
    for c, (k, v, s) in zip(cols, cards):
        c.markdown(theme.card(k, v, s), unsafe_allow_html=True)
    st.write("")

    t1, t2, t3, t4 = st.tabs(["Data quality", "Cleaning decisions & split", "Labels & imbalance", "Method, licence & limitations"])
    with t1:
        for w in q.get("warnings", []):
            theme.note(w, "warn")
        a, b = st.columns(2)
        with a:
            st.markdown("**Missing values per channel**")
            st.dataframe(pd.DataFrame({"channel": list(q["missing_values"]), "missing": list(q["missing_values"].values())}), hide_index=True, use_container_width=True)
            st.markdown("**Readings at/above the brief's plant-specific thresholds**")
            rows = [{"channel": C.CHANNEL_INFO[c]["label"], "threshold": f'{v["threshold"]:g} {C.CHANNEL_INFO[c]["unit"]}', "minutes": f'{v["count"]:,}', "share": f'{v["pct"]:.3f} %'}
                    for c, v in q["threshold_exceedances"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with b:
            st.markdown("**Summary statistics (raw values)**")
            ss = pd.DataFrame(q["summary_stats"]).T[["count", "mean", "std", "min", "1%", "50%", "99%", "max"]]
            st.dataframe(ss.round(2), use_container_width=True)
        daily = scored["row_valid"].resample("1D").mean() * 100
        fig = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color="#22d3ee"))
        fig.update_yaxes(title="% of minutes fully valid", range=[0, 100])
        theme.style_fig(fig, height=260, title="Daily share of fully valid minutes (all four channels)")
        st.plotly_chart(fig, use_container_width=True, key="dq_daily")

    with t2:
        st.markdown("**Cleaning / filtering decisions (raw file is never modified; nothing is interpolated)**")
        for d in summary.get("prep_decisions", []):
            st.markdown(f"- {d}")
        sp = summary.get("split", {})
        st.markdown("**Chronological split**")
        st.markdown(f"- Cutoff: `{sp.get('cutoff')}` ({sp.get('note')})  \n- Valid training rows: **{sp.get('train_valid_rows', 0):,}** · valid test rows: **{sp.get('test_valid_rows', 0):,}**  \n"
                    "- Scalers, health-score percentiles, Isolation-Forest thresholds, autoencoder scaling and SMOTE are all fitted on the **training** portion only.")
        hs = D.load_scorer()
        if hs:
            st.markdown("**Health score** = 100 × (1 − w₁·norm(vibration) − w₂·norm(torque) − w₃·norm(max bearing temp)), clipped to [0,100]")
            b = pd.DataFrame({"signal": list(hs["bounds"]), "norm = 0 at (training median)": [round(v[0], 2) for v in hs["bounds"].values()],
                              "norm = 1 at (training 99.9th pct)": [round(v[1], 2) for v in hs["bounds"].values()],
                              "weight": [hs["weights"].get(k) for k in hs["bounds"]]})
            st.dataframe(b, hide_index=True, use_container_width=True)

    with t3:
        ls = summary.get("label_stats", {})
        names = friendly_label_names()
        rows = []
        for lab in ("weak_vibration", "weak_combined", "strong_any"):
            for sp_ in ("train", "test"):
                v = ls.get(f"{lab}__{sp_}")
                if v:
                    rows.append({"label": names[lab], "split": sp_, "valid minutes": f"{v['valid_rows']:,}", "positive minutes": f"{v['positives']:,}", "positive rate": f"{v['rate_pct']:.3f} %"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        sa = ls.get("strong_any__train", {})
        theme.note(f"<b>Finding that differs from the brief:</b> the brief expects anomalies to be rare (&lt; 0.5 %). In the real data the strong 'any channel ≥ threshold' rule flags "
                   f"<b>{sa.get('rate_pct', 0):.1f} %</b> of valid training minutes because torque ≥ 85 % is routine operation for this centrifuge. "
                   "Only the vibration ≥ 12 mm/s label is genuinely rare, which is where class-imbalance handling (class weights, SMOTE) matters.", "warn")

    with t4:
        st.markdown(f"""
**Dataset:** {C.DATASET_NAME} · Arab Potash Company (Jordan) · Sulzer CFG-42 · [{C.DATASET_DOI}]({C.DATASET_DOI})
Licence: described as open access in the brief - please verify the current licence on the Zenodo record before redistribution.

**Methods:** Isolation Forest baseline (unsupervised) · XGBoost weak-label classifier with none / class-weight / **SMOTE (training-only)** imbalance handling · optional healthy-trained autoencoder · torque→temperature residual twin.

**Limitations**
- Labels are threshold-derived weak labels, not confirmed faults; DCS thresholds are plant-specific, not universal.
- Vibration channel is flat (< 1 mm/s) until early Nov 2022 and the liquid-end bearing sensor reads 0 °C for long stretches - those rows are excluded, leaving a shorter clean training window.
- A ~27-day gap (June 2023) removes data from the start of the test period.
- Very few held-out positive events - metrics are indicative, not conclusive.
- Health score and state bands are prototype rules requiring engineering validation.
- This is historical playback, not a live digital twin, and not a safety system.
""")
        theme.note(C.DISCLAIMER, "warn")
