"""Centrifuge Anomaly Detection & Equipment Health Digital Twin - Streamlit dashboard.

Run from the project root:   streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Centrifuge Health Digital Twin", page_icon="⚙️", layout="wide", initial_sidebar_state="expanded")

import data_access as D  # noqa: E402  (also puts backend/ on sys.path)
import theme  # noqa: E402
from src import config as C  # noqa: E402

PAGES = {
    "🩺  Health Monitor": "health_monitor",
    "🎯  Anomaly Detection Results": "anomaly_results",
    "⏪  Historical Replay": "historical_replay",
    "🔍  Model Explanation": "explanation",
    "📋  Data Quality & Method": "data_quality",
}


def run_pipeline_ui() -> None:
    """Run the backend pipeline in-process (a few minutes). Shows progress and any error."""
    with st.spinner("Running backend pipeline on the real dataset - this takes a few minutes ..."):
        try:
            import run_pipeline  # backend/run_pipeline.py
            run_pipeline.main(fast=False, skip_ae=False)
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Pipeline finished.")
            st.rerun()
        except Exception as e:  # show a readable message instead of a stack trace
            st.error(f"Pipeline failed: {e}")


def setup_screen() -> None:
    theme.hero("Centrifuge Health Digital Twin", "Anomaly detection and equipment-health monitoring for the Sulzer CFG-42 industrial centrifuge.",
               ["PROJECT 2", "STREAMLIT + PLOTLY", "REAL DATA ONLY"])
    from src.data_loader import DatasetNotFoundError, find_dataset
    try:
        p = find_dataset()
        theme.note(f"Dataset found: <code>{p.name}</code>. Model artefacts have not been generated yet.")
        if st.button("▶ Run backend pipeline now", type="primary", key="run_pipe"):
            run_pipeline_ui()
        st.code("python backend/run_pipeline.py", language="bash")
    except DatasetNotFoundError as e:
        theme.note("The Sulzer CFG-42 dataset is missing. No synthetic data is substituted.", "warn")
        st.code(str(e))


def main() -> None:
    theme.inject_css()
    with st.sidebar:
        st.markdown("### ⚙️ CFG-42 Digital Twin")
        st.caption("Equipment health · anomaly detection")
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")
        st.divider()
        st.caption("**Historical playback** of recorded data. Prototype for decision support - not a safety system.")
        st.caption(f"Dataset DOI: {C.DATASET_DOI}")

    if not D.artefacts_ready():
        setup_screen()
        return

    scored, summary = D.load_scored(), D.load_summary()
    with st.sidebar:
        sp = summary.get("split", {})
        st.caption(f"Train/test cutoff: `{str(sp.get('cutoff', ''))[:10]}`")
        if st.button("↻ Re-run pipeline", key="rerun_pipe"):
            run_pipeline_ui()

    try:
        import importlib
        mod = importlib.import_module(f"views.{PAGES[page]}")
        mod.render(scored, summary)
    except Exception as e:  # keep the app alive and explain, rather than crash
        st.error(f"This page could not be rendered: {type(e).__name__}: {e}")
        st.exception(e)


main()
