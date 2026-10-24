"""Route A (unsupervised): Isolation Forest baseline.

Training uses *training-period rows only*. The alert threshold is a quantile of the training
anomaly-score distribution, i.e. the contamination assumption (documented, compared across
several settings rather than assumed correct).
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C
from .event_evaluation import event_evaluation, pointwise_metrics, strip_private
from .features import feature_valid, isolation_forest_features, raw_features
from .preprocessing import healthy_mask

CONTAMINATIONS = (0.005, 0.01, 0.02)
LABELS = ("weak_vibration", "weak_combined", "strong_any")


def fit_isolation_forest(X: pd.DataFrame, n_estimators: int = 300, random_state: int = 42) -> IsolationForest:
    model = IsolationForest(n_estimators=n_estimators, max_samples=1024, contamination="auto", random_state=random_state, n_jobs=-1)
    model.fit(X.to_numpy(dtype=float))
    return model


def anomaly_score(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous (negated sklearn score_samples)."""
    return -model.score_samples(X.to_numpy(dtype=float))


def train_isolation_forest(
    df: pd.DataFrame, feats: pd.DataFrame, n_estimators: int = 300, random_state: int = 42
) -> dict[str, Any]:
    """Train several configurations on the train split; evaluate each on the held-out test split.

    Configurations compared: feature set (raw 4 channels vs raw + rolling), training rows
    (all valid vs healthy-only) and contamination (threshold quantile). The default configuration
    ("selected" features, all valid training rows, contamination 1 %) was fixed *before* looking at
    test results; all other variants are reported so nothing is cherry-picked.
    """
    train_idx = df["split"] == "train"
    test_idx = df["split"] == "test"
    fsets = {"raw": raw_features(), "selected": isolation_forest_features()}
    results: list[dict[str, Any]] = []
    artefacts: dict[str, Any] = {}

    for fs_name, cols in fsets.items():
        ok = feature_valid(feats, df["row_valid"], cols)
        tr_all = train_idx & ok
        tr_healthy = tr_all & healthy_mask(df)
        for tr_name, tr_mask in (("all_valid", tr_all), ("healthy_only", tr_healthy)):
            if tr_mask.sum() < 500:
                continue
            model = fit_isolation_forest(feats.loc[tr_mask, cols], n_estimators, random_state)
            tr_score = anomaly_score(model, feats.loc[tr_all, cols])
            te_mask = test_idx & ok
            te_score = pd.Series(np.nan, index=df.index)
            te_score[te_mask] = anomaly_score(model, feats.loc[te_mask, cols])
            for cont in CONTAMINATIONS:
                thr = float(np.quantile(tr_score, 1 - cont))
                alert = (te_score >= thr).fillna(False)
                for lab in LABELS:
                    y = df[f"label_{lab}"]
                    pw = pointwise_metrics(y[te_mask].to_numpy(), alert[te_mask].to_numpy(), te_score[te_mask].to_numpy())
                    ev = strip_private(event_evaluation(y[te_mask].reindex(df.index, fill_value=False), alert, te_mask))
                    results.append({
                        "features": fs_name, "train_rows": tr_name, "contamination": cont, "threshold": thr,
                        "label": lab, **{f"pw_{k}": v for k, v in pw.items()}, **{f"ev_{k}": v for k, v in ev.items()},
                    })
            artefacts[(fs_name, tr_name)] = {"model": model, "cols": cols, "train_scores": tr_score}

    default_key = ("selected", "all_valid")
    art = artefacts[default_key]
    return {"results": pd.DataFrame(results), "artefacts": artefacts, "default": default_key, "default_art": art}


def score_all(art: dict[str, Any], df: pd.DataFrame, feats: pd.DataFrame, contamination: float = 0.01) -> tuple[pd.Series, float]:
    """Score every usable row with the chosen model; returns (score series, threshold)."""
    cols = art["cols"]
    ok = feature_valid(feats, df["row_valid"], cols)
    s = pd.Series(np.nan, index=df.index, name="if_score")
    s[ok] = anomaly_score(art["model"], feats.loc[ok, cols])
    thr = float(np.quantile(art["train_scores"], 1 - contamination))
    return s, thr


def save_model(art: dict[str, Any], threshold: float, contamination: float, meta: dict[str, Any]) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": art["model"], "feature_cols": art["cols"], "threshold": threshold, "contamination": contamination, "meta": meta},
        C.MODELS_DIR / "isolation_forest.joblib",
    )
