"""Data-quality validation: dimensions, range, duplicates, gaps, missing values, thresholds."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import config as C


def validate(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serialisable data-quality report. Does not modify ``df``."""
    rep: dict[str, Any] = {}
    rep["rows"] = int(len(df))
    rep["columns"] = list(df.columns)
    ts = df["timestamp"]
    rep["unparseable_timestamps"] = int(ts.isna().sum())
    ts_ok = ts.dropna()
    rep["timestamp_min"] = str(ts_ok.min()) if len(ts_ok) else None
    rep["timestamp_max"] = str(ts_ok.max()) if len(ts_ok) else None
    rep["duplicate_timestamps"] = int(ts_ok.duplicated().sum())
    rep["is_monotonic_increasing"] = bool(ts_ok.is_monotonic_increasing)

    uniq = ts_ok.drop_duplicates().sort_values()
    diffs = uniq.diff().dropna()
    one_min = pd.Timedelta(minutes=1)
    rep["pct_intervals_exactly_1min"] = float((diffs == one_min).mean() * 100) if len(diffs) else None
    gaps = diffs[diffs > one_min]
    rep["gap_count"] = int(len(gaps))
    rep["largest_gap_minutes"] = float(gaps.max() / one_min) if len(gaps) else 0.0
    rep["missing_minutes_total"] = float(((gaps / one_min) - 1).sum()) if len(gaps) else 0.0
    rep["top_gaps"] = [
        {"start": str(uniq.loc[i - 1] if (i - 1) in uniq.index else ""), "gap_minutes": float(g / one_min)}
        for i, g in gaps.sort_values(ascending=False).head(5).items()
    ]
    if len(uniq):
        expected = int((uniq.max() - uniq.min()) / one_min) + 1
        rep["expected_records_at_1min"] = expected
        rep["coverage_pct"] = float(len(uniq) / expected * 100)
        rep["span_days"] = float((uniq.max() - uniq.min()).total_seconds() / 86400)

    rep["missing_values"] = {c: int(df[c].isna().sum()) for c in C.CHANNELS}
    rep["numeric_dtypes_ok"] = bool(all(pd.api.types.is_numeric_dtype(df[c]) for c in C.CHANNELS))
    rep["non_finite_values"] = {c: int((~np.isfinite(df[c].dropna())).sum()) for c in C.CHANNELS}
    rep["summary_stats"] = {
        c: {k: float(v) for k, v in df[c].describe(percentiles=[0.01, 0.5, 0.99]).items()} for c in C.CHANNELS
    }
    rep["negative_values"] = {c: int((df[c] < 0).sum()) for c in C.CHANNELS}
    rep["threshold_exceedances"] = {
        c: {"threshold": C.THRESHOLDS[c], "count": int((df[c] >= C.THRESHOLDS[c]).sum()),
            "pct": float((df[c] >= C.THRESHOLDS[c]).mean() * 100)}
        for c in C.CHANNELS
    }
    any_exc = np.zeros(len(df), dtype=bool)
    for c in C.CHANNELS:
        any_exc |= (df[c] >= C.THRESHOLDS[c]).to_numpy()
    rep["any_channel_exceedance_count"] = int(any_exc.sum())
    rep["any_channel_exceedance_pct"] = float(any_exc.mean() * 100)

    # Flat-line (stuck sensor) check: longest run of identical consecutive values
    stuck = {}
    for c in C.CHANNELS:
        s = df[c]
        grp = (s != s.shift()).cumsum()
        stuck[c] = int(s.groupby(grp).transform("size").max()) if len(s) else 0
    rep["longest_constant_run_minutes"] = stuck

    warnings = []
    if rep["duplicate_timestamps"]:
        warnings.append(f"{rep['duplicate_timestamps']} duplicate timestamps found.")
    if rep["gap_count"]:
        warnings.append(f"{rep['gap_count']} timestamp gaps (largest {rep['largest_gap_minutes']:.0f} min).")
    if any(rep["missing_values"].values()):
        warnings.append(f"Missing sensor values: {rep['missing_values']}")
    if not rep["is_monotonic_increasing"]:
        warnings.append("Timestamps are not sorted; pipeline will sort chronologically.")
    for c, n in stuck.items():
        if n >= 240:
            warnings.append(f"{c}: constant for {n} consecutive minutes (possible stuck sensor / shutdown).")
    rep["warnings"] = warnings
    return rep


def save_report(rep: dict[str, Any]) -> None:
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (C.METRICS_DIR / "data_quality_report.json").write_text(json.dumps(rep, indent=2))
