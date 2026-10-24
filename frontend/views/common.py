"""Shared helpers for dashboard pages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def smooth(s: pd.Series, minutes: int) -> pd.Series:
    """Causal display smoothing (trailing mean). minutes<=1 returns the series unchanged.

    NaN gaps stay NaN (min_periods=1 so a window with any valid value is drawn, but fully
    invalid stretches are never bridged).
    """
    if minutes <= 1:
        return s
    return s.rolling(minutes, min_periods=1).mean().where(s.notna())


def fmt(v, nd: int = 1, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:,.{nd}f}{unit}"


def pct(v, nd: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v*100:.{nd}f}%"


def latest_valid(w: pd.DataFrame) -> pd.Series | None:
    """Most recent row with all channels valid (or None)."""
    v = w[w["row_valid"]]
    return None if v.empty else v.iloc[-1]


def period_of(ts: pd.Timestamp, cutoff: str | None) -> str:
    if not cutoff:
        return "n/a"
    return "training period (in-sample)" if ts < pd.Timestamp(cutoff) else "held-out test period"


def sensor_title(ch: str) -> str:
    i = C.CHANNEL_INFO[ch]
    return f"{i['label']} ({i['unit']})"


def friendly_label_names() -> dict[str, str]:
    return {
        "weak_vibration": "Vibration ≥ 12 mm/s (primary weak label)",
        "weak_combined": "Vibration ≥ 12 mm/s OR bearing temp ≥ 85 °C",
        "strong_any": "Any channel ≥ threshold (strong rule)",
    }


def table_fmt(df: pd.DataFrame, pct_cols=(), float_cols=(), nd: int = 3):
    """Return a display copy with consistent formatting (None -> '–')."""
    d = df.copy()
    for c in pct_cols:
        if c in d:
            d[c] = d[c].map(lambda x: "–" if pd.isna(x) else f"{x*100:.1f}%")
    for c in float_cols:
        if c in d:
            d[c] = d[c].map(lambda x: "–" if pd.isna(x) else f"{x:.{nd}f}")
    return d.fillna("–")
