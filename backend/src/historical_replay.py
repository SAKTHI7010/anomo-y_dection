"""Historical replay helpers. Playback of *recorded* data - not live hardware telemetry."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .event_evaluation import merge_events


def window(scored: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Slice [start, end] inclusive (empty frame if outside range)."""
    if start > end:
        start, end = end, start
    return scored.loc[start:end]


def downsample(w: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    """Reduce plotting load: mean for signals, max for flags/scores (so short alerts are never hidden)."""
    if len(w) <= max_points:
        return w
    rule = f"{int(np.ceil(len(w) / max_points))}min"
    bool_cols = [c for c in w.columns if w[c].dtype == bool]
    max_cols = [c for c in ("if_score", "xgb_prob", "ae_error") if c in w.columns]
    float_cols = [c for c in w.columns if w[c].dtype.kind == "f" and c not in max_cols]
    parts = [
        w[float_cols].resample(rule).mean(),
        w[max_cols].resample(rule).max(),
        w[bool_cols].astype(int).resample(rule).max().astype(bool),
    ]
    out = pd.concat(parts, axis=1)
    if "split" in w.columns:
        out["split"] = w["split"].resample(rule).first()
    return out


def replay_summary(w: pd.DataFrame, alert_col: str = "alert_if", merge_gap_min: int = 30) -> dict[str, Any]:
    """Headline numbers for a replay window."""
    if w.empty:
        return {"rows": 0}
    ev = merge_events(w[alert_col], merge_gap_min) if alert_col in w else pd.DataFrame()
    return {
        "rows": int(len(w)),
        "valid_rows": int(w["row_valid"].sum()) if "row_valid" in w else None,
        "health_min": float(w["health"].min()) if w["health"].notna().any() else None,
        "health_mean": float(w["health"].mean()) if w["health"].notna().any() else None,
        "alert_events": int(len(ev)),
        "events": ev,
    }
