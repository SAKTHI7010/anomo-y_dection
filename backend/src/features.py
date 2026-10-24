"""Causal time-series feature engineering.

Every rolling statistic uses a *trailing* window (past + current observation only), so a
feature at time t never depends on data after t. Warm-up rows (and rows whose window holds a
NaN) are NaN and are dropped consistently via ``feature_valid``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

EPS_VIB = 0.1  # mm/s floor used to guard the torque/vibration ratio against division by ~0


def build_features(df: pd.DataFrame, windows: tuple[int, ...] = C.ROLL_WINDOWS) -> pd.DataFrame:
    """Return a feature frame aligned to ``df.index``. Does not modify ``df``."""
    feats: dict[str, pd.Series] = {}
    base = df[C.CHANNELS].astype(float).copy()
    base["temp_max"] = base[["temp_solid", "temp_liquid"]].max(axis=1, skipna=False)
    for c in C.CHANNELS + ["temp_max"]:
        s = base[c]
        feats[c] = s
        for w in windows:
            r = s.rolling(w, min_periods=w)
            feats[f"{c}_mean_{w}"] = r.mean()
            feats[f"{c}_std_{w}"] = r.std()
        for w in (15, 60):
            r = s.rolling(w, min_periods=w)
            feats[f"{c}_min_{w}"] = r.min()
            feats[f"{c}_max_{w}"] = r.max()
        feats[f"{c}_diff_1"] = s.diff(1)
        feats[f"{c}_diff_5"] = s.diff(5)
    # cross-channel features (units documented in FEATURE_DOC)
    feats["vib_x_temp"] = base["vibration"] * base["temp_max"]  # mm/s * °C
    feats["torque_per_vib"] = base["torque"] / base["vibration"].clip(lower=EPS_VIB)  # % per (mm/s)
    out = pd.DataFrame(feats, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


FEATURE_DOC = {
    "<ch>_mean_W / _std_W": "trailing W-minute mean / standard deviation (W = 5, 15, 60)",
    "<ch>_min_W / _max_W": "trailing W-minute min / max (W = 15, 60)",
    "<ch>_diff_1 / _diff_5": "first difference over 1 / 5 minutes (unit per minute step)",
    "temp_max": "max(temp_solid, temp_liquid) in °C",
    "vib_x_temp": "vibration [mm/s] × temp_max [°C]",
    "torque_per_vib": f"torque [%] / max(vibration, {EPS_VIB}) [mm/s]  (division-by-zero guard)",
}


def isolation_forest_features() -> list[str]:
    """Four raw channels plus selected rolling features (brief, Route A)."""
    cols = list(C.CHANNELS)
    for c in C.CHANNELS:
        cols += [f"{c}_mean_15", f"{c}_std_15", f"{c}_mean_60"]
    return cols


def raw_features() -> list[str]:
    return list(C.CHANNELS)


def feature_valid(feats: pd.DataFrame, row_valid: pd.Series, cols: list[str]) -> pd.Series:
    """Rows usable for modelling: raw channels valid AND every requested feature finite."""
    return row_valid & feats[cols].notna().all(axis=1)
