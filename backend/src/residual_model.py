"""Optional digital-twin upgrade: torque -> bearing-temperature residual.

Expected bearing temperature is learned from torque only (no ambient temperature exists in the
dataset) on *training-period* valid running rows. residual = actual - expected. A *persistently*
positive residual (rolling median over 60 min above a training-derived 99th percentile) is flagged
as "worth investigating". It does NOT prove bearing degradation or poor lubrication.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config as C

TARGETS = ("temp_solid", "temp_liquid")
PERSIST_WINDOW = 60


def _design(df: pd.DataFrame) -> pd.DataFrame:
    """Torque now plus 60-min trailing mean torque (thermal inertia). Causal."""
    tq = df["torque"]
    return pd.DataFrame({"torque": tq, "torque_mean_60": tq.rolling(60, min_periods=60).mean()}, index=df.index)


def fit_residual_model(df: pd.DataFrame) -> dict[str, Any]:
    """Fit on train split only; returns models + persistence thresholds."""
    X = _design(df)
    out: dict[str, Any] = {"models": {}, "thresholds": {}}
    for t in TARGETS:
        m = (df["split"] == "train") & df["row_valid"] & df["running"] & X.notna().all(axis=1)
        if m.sum() < 1000:
            continue
        mdl = make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0))
        mdl.fit(X[m], df.loc[m, t])
        res = df.loc[m, t] - mdl.predict(X[m])
        pers = res.rolling(PERSIST_WINDOW, min_periods=PERSIST_WINDOW).median()
        out["models"][t] = mdl
        out["thresholds"][t] = float(np.nanquantile(pers.dropna(), 0.99))
        out[f"{t}_train_resid_std"] = float(res.std())
    return out


def apply_residual_model(df: pd.DataFrame, fit: dict[str, Any]) -> pd.DataFrame:
    """Return residuals, persistent residuals and persistent-flag per bearing (NaN where inputs invalid)."""
    X = _design(df)
    ok_x = X.notna().all(axis=1)
    cols: dict[str, pd.Series] = {}
    for t, mdl in fit["models"].items():
        valid = df["row_valid"] & ok_x & df["running"]
        exp = pd.Series(np.nan, index=df.index)
        exp[valid] = mdl.predict(X[valid])
        resid = df[t] - exp
        pers = resid.rolling(PERSIST_WINDOW, min_periods=PERSIST_WINDOW // 2).median()
        cols[f"resid_{t}"] = resid
        cols[f"resid_{t}_persist"] = pers
        cols[f"resid_{t}_flag"] = (pers > fit["thresholds"][t]).fillna(False)
    out = pd.DataFrame(cols, index=df.index)
    if out.empty:
        return out
    flag_cols = [c for c in out.columns if c.endswith("_flag")]
    out["resid_flag_any"] = out[flag_cols].any(axis=1)
    return out
