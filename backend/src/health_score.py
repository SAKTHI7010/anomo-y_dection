"""Interpretable 0-100 health score, prototype state machine and simple 24 h trend forecast."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class HealthScorer:
    """health = 100 * (1 - w1*norm(vibration) - w2*norm(torque) - w3*norm(temp_max)), clipped to [0, 100].

    norm(x) = clip((x - lo) / (hi - lo), 0, 1) where lo / hi are *training-period* percentiles
    (defaults: median and 99.9th percentile of valid running rows). Values outside the range are
    clamped explicitly. Weights are configurable and normalised to sum to 1.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(C.HEALTH_WEIGHTS))
    low_q: float = 0.50
    high_q: float = 0.999
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame) -> "HealthScorer":
        """Fit normalisation bounds on training rows only (valid + running)."""
        d = train[train["row_valid"] & train["running"]].copy()
        if d.empty:
            raise ValueError("No valid running rows available to fit the health score.")
        d["temp_max"] = d[["temp_solid", "temp_liquid"]].max(axis=1)
        for key, col in (("vibration", "vibration"), ("torque", "torque"), ("temp_max", "temp_max")):
            lo, hi = float(d[col].quantile(self.low_q)), float(d[col].quantile(self.high_q))
            if hi <= lo:  # degenerate -> widen so the division is safe
                hi = lo + 1e-6
            self.bounds[key] = (lo, hi)
        return self

    def _norm(self, x: pd.Series, key: str) -> pd.Series:
        lo, hi = self.bounds[key]
        return ((x - lo) / (hi - lo)).clip(0.0, 1.0)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame[health, norm_vibration, norm_torque, norm_temp_max]; NaN where inputs invalid."""
        if not self.bounds:
            raise RuntimeError("HealthScorer.fit() must be called before score().")
        w = self.weights
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}  # guarantee penalties can never exceed 1
        temp_max = df[["temp_solid", "temp_liquid"]].max(axis=1, skipna=False)
        nv = self._norm(df["vibration"], "vibration")
        nt = self._norm(df["torque"], "torque")
        nm = self._norm(temp_max, "temp_max")
        h = 100.0 * (1.0 - w["vibration"] * nv - w["torque"] * nt - w["temp_max"] * nm)
        h = h.clip(0.0, 100.0)
        return pd.DataFrame({"health": h, "norm_vibration": nv, "norm_torque": nt, "norm_temp_max": nm}, index=df.index)

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights, "low_q": self.low_q, "high_q": self.high_q, "bounds": self.bounds}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HealthScorer":
        return cls(weights=d["weights"], low_q=d["low_q"], high_q=d["high_q"], bounds={k: tuple(v) for k, v in d["bounds"].items()})


def health_state(h: float | None) -> tuple[str, str, str]:
    """Map a health score to (state, colour, interpretation). NaN/None -> 'No data'."""
    if h is None or not np.isfinite(h):
        return "No data", "#64748b", "Sensor data unavailable or invalid for this period"
    # bands: >70 Healthy | 40-70 Attention | 20-40 Warning | <20 Critical (lower bounds inclusive except Healthy)
    idx = 0 if h > 70 else 1 if h >= 40 else 2 if h >= 20 else 3
    _, name, colour, text = C.HEALTH_STATES[idx]
    return name, colour, text


def states_series(h: pd.Series) -> pd.Series:
    """Vectorised state labels (>70 Healthy, 40-70 Attention, 20-40 Warning, <20 Critical)."""
    out = pd.Series("No data", index=h.index, dtype=object)
    out[h > 70] = "Healthy"
    out[(h <= 70) & (h >= 40)] = "Attention"
    out[(h < 40) & (h >= 20)] = "Warning"
    out[h < 20] = "Critical"
    return out


def forecast_linear(health: pd.Series, horizon_h: int = 24, lookback_h: int = 24) -> dict[str, Any] | None:
    """Simple linear extrapolation of hourly mean health over the next ``horizon_h`` hours.

    Returns dict(ts, mean, lo, hi, slope_per_h) with an approximate 95 % band built from the
    residual standard deviation, or ``None`` if there is not enough valid data. This is an
    estimate only - it has NOT been validated as a failure predictor.
    """
    h = health.dropna()
    if h.empty:
        return None
    end = h.index.max()
    hourly = h[h.index > end - pd.Timedelta(hours=lookback_h)].resample("1h").mean().dropna()
    if len(hourly) < 6:
        return None
    x = (hourly.index - hourly.index[0]) / pd.Timedelta(hours=1)
    x = np.asarray(x, dtype=float)
    y = hourly.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    sd = float(resid.std(ddof=2)) if len(y) > 2 else 0.0
    fx = np.arange(x[-1] + 1, x[-1] + horizon_h + 1)
    fts = hourly.index[0] + pd.to_timedelta(fx, unit="h")
    mean = np.clip(slope * fx + intercept, 0, 100)
    band = 1.96 * sd * np.sqrt(1 + (fx - x.mean()) ** 2 / max(((x - x.mean()) ** 2).sum(), 1e-9))
    return {
        "ts": fts, "mean": mean, "lo": np.clip(mean - band, 0, 100), "hi": np.clip(mean + band, 0, 100),
        "slope_per_h": float(slope), "resid_sd": sd,
    }
