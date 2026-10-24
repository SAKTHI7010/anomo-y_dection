"""Time-aware preprocessing: grid alignment, sensor-validity flags, labels and chronological split.

Design rules
------------
* The raw file is never modified. Values are never interpolated here - missing/invalid
  readings stay ``NaN`` and the affected rows are excluded from modelling via ``row_valid``.
* Labels are *threshold-derived candidate anomalies* (weak labels), not confirmed faults.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

LABEL_NAMES = {
    "strong_any": "Any channel ≥ threshold (strong rule)",
    "weak_vibration": "Vibration ≥ 12 mm/s (primary weak label)",
    "weak_combined": "Vibration ≥ 12 mm/s OR bearing temp ≥ 85 °C",
}


@dataclass
class PrepInfo:
    """Records every cleaning decision so it can be reported in the app and README."""

    decisions: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.decisions.append(msg)


def prepare(df_raw: pd.DataFrame, info: PrepInfo | None = None) -> pd.DataFrame:
    """Sort, de-duplicate, reindex to a strict 1-minute grid and flag invalid sensor readings.

    Returns a DataFrame indexed by ``timestamp`` with:
      channel columns (invalid readings set to NaN), ``valid_<ch>`` booleans and ``row_valid``.
    """
    info = info or PrepInfo()
    df = df_raw.dropna(subset=["timestamp"]).copy()
    n0 = len(df)
    df = df.sort_values("timestamp")
    dups = int(df["timestamp"].duplicated().sum())
    if dups:
        df = df.drop_duplicates("timestamp", keep="first")
        info.add(f"Dropped {dups} duplicate timestamps (kept first).")
    df = df.set_index("timestamp")
    grid = pd.date_range(df.index.min(), df.index.max(), freq=C.EXPECTED_FREQ)
    added = len(grid) - len(df)
    df = df.reindex(grid)
    df.index.name = "timestamp"
    info.add(f"Reindexed {n0:,} rows to a strict 1-min grid ({len(grid):,} rows; {added:,} missing minutes inserted as NaN, no interpolation).")

    valid = pd.DataFrame(index=df.index)
    for c in C.CHANNELS:
        valid[c] = df[c].notna() & np.isfinite(df[c].astype(float))

    # (1) Liquid-end bearing temperature reading exactly 0 °C is physically implausible next to a
    #     solid-end bearing at tens of °C -> treated as a sensor outage, not a measurement.
    zero_liq = (df["temp_liquid"] == 0).fillna(False)
    if zero_liq.any():
        valid.loc[zero_liq, "temp_liquid"] = False
        info.add(f"temp_liquid == 0.0 °C on {int(zero_liq.sum()):,} rows treated as sensor outage (excluded from modelling, kept in raw file).")

    # (2) Vibration sensor regime: while the machine is running (torque >= 30 %) a daily median
    #     vibration < 1 mm/s means the channel was effectively flat / not yet commissioned.
    running = df["torque"] >= 30
    day = df.index.normalize()
    med = df["vibration"].where(running).groupby(day).median()
    suspect_days = med[med < 1.0].index
    vib_suspect = pd.Series(day.isin(suspect_days), index=df.index)
    if vib_suspect.any():
        valid.loc[vib_suspect, "vibration"] = False
        info.add(
            f"Vibration flat (<1 mm/s daily median while torque ≥ 30 %) from {suspect_days.min().date()} to "
            f"{suspect_days.max().date()} ({int(vib_suspect.sum()):,} rows) - treated as unreliable vibration regime and excluded."
        )

    out = df.copy()
    for c in C.CHANNELS:
        out[f"valid_{c}"] = valid[c]
        out.loc[~valid[c], c] = np.nan
    out["row_valid"] = valid.all(axis=1)
    out["running"] = (out["torque"] >= 30).fillna(False)
    info.add(f"{int(out['row_valid'].sum()):,} of {len(out):,} rows ({out['row_valid'].mean()*100:.1f} %) have all four channels valid.")
    out.attrs["prep_decisions"] = list(info.decisions)
    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-channel exceedance flags and the three named weak/strong labels (valid rows only)."""
    out = df.copy()
    for c in C.CHANNELS:
        out[f"exceed_{c}"] = (out[c] >= C.THRESHOLDS[c]).fillna(False)
    ex = out[[f"exceed_{c}" for c in C.CHANNELS]]
    out["label_strong_any"] = ex.any(axis=1)
    out["label_weak_vibration"] = out["exceed_vibration"]
    out["label_weak_combined"] = out["exceed_vibration"] | out["exceed_temp_solid"] | out["exceed_temp_liquid"]
    return out


def chronological_split(df: pd.DataFrame, train_months: int = C.TRAIN_FRACTION_MONTHS) -> tuple[pd.Timestamp, str]:
    """Return (cutoff timestamp, note). First ``train_months`` months train, remainder test.

    Falls back to a 75/25 split if the data span is too short for the requested months or
    the test side would contain fewer than 1,000 valid rows.
    """
    start, end = df.index.min(), df.index.max()
    cutoff = start + pd.DateOffset(months=train_months)
    note = f"first {train_months} months train / remainder test"
    test_valid = int(df.loc[df.index >= cutoff, "row_valid"].sum())
    if cutoff >= end or test_valid < 1000:
        cutoff = start + (end - start) * 0.75
        note = "fallback 75 % / 25 % chronological split (data span too short for 9/3-month split)"
    return pd.Timestamp(cutoff), note


def assign_split(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    return pd.Series(np.where(df.index < cutoff, "train", "test"), index=df.index, name="split")


def healthy_mask(df: pd.DataFrame) -> pd.Series:
    """Conservative healthy-period selection (brief): vib<8, torque<70, temps<75 on valid rows.

    These are training-selection rules only - not universal limits.
    """
    m = df["row_valid"].copy()
    for c, lim in C.HEALTHY_RULES.items():
        m &= (df[c] < lim).fillna(False)
    return m


def future_label(label: pd.Series, valid: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Early-warning target: does ``label`` occur at any of the next ``horizon`` minutes?

    Returns (target, usable). Rows are usable only if no alarm is active *now* and all of the
    next ``horizon`` minutes are valid, so unknown/missing futures are never treated as negatives.
    """
    lab = label.astype(float)
    # forward-looking window over t+1 .. t+h  (reverse-roll trick)
    fut = lab[::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    fut_valid = valid.astype(float)[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    usable = (fut_valid == 1.0) & valid & ~label
    return (fut == 1.0) & usable, usable
