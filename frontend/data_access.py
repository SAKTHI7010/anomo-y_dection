"""Cached access to backend artefacts. The frontend never recomputes models - it only reads files
written by ``backend/run_pipeline.py`` (real-data results; nothing is simulated)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src import config as C  # noqa: E402

SCORED_PATH = C.PROCESSED_DIR / "scored.parquet"
SUMMARY_PATH = C.METRICS_DIR / "summary.json"
SHAP_PATH = C.PROCESSED_DIR / "shap_sample.parquet"


def artefacts_ready() -> bool:
    return SCORED_PATH.exists() and SUMMARY_PATH.exists()


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading scored sensor data ...")
def _load_scored(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(SCORED_PATH)


def load_scored() -> pd.DataFrame:
    return _load_scored(_mtime(SCORED_PATH))


@st.cache_data(show_spinner=False)
def _load_json(path: str, mtime: float) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def load_summary() -> dict:
    return _load_json(str(SUMMARY_PATH), _mtime(SUMMARY_PATH))


def load_quality() -> dict:
    p = C.METRICS_DIR / "data_quality_report.json"
    return _load_json(str(p), _mtime(p))


def load_scorer() -> dict:
    p = C.MODELS_DIR / "health_scorer.json"
    return _load_json(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _load_csv(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_metrics_csv(name: str) -> pd.DataFrame:
    p = C.METRICS_DIR / name
    return _load_csv(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def _load_parquet(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_shap() -> pd.DataFrame:
    return _load_parquet(str(SHAP_PATH), _mtime(SHAP_PATH))


def pr_curve_files() -> list[str]:
    return sorted(p.name for p in C.METRICS_DIR.glob("pr_curve__*.csv"))
