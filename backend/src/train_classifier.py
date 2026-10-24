"""Route B (supervised weak-label): XGBoost with class-imbalance handling.

Imbalance strategies compared (all strictly *after* the chronological split and fitted on the
training portion only - SMOTE never sees validation/test rows or crosses the time boundary):

* ``none``          - plain XGBoost
* ``class_weight``  - scale_pos_weight = n_negative / n_positive  (class_weight='balanced' equivalent)
* ``smote``         - SMOTE over-sampling of the minority class (imbalanced-learn), partial balance
                      (minority = ``SMOTE_RATIO`` x majority) to avoid drowning the real signal in synthetic points.

Labels are threshold-derived weak labels - the model approximates plant rules and is not proof of
true mechanical failure. With ``horizon > 0`` the target becomes "threshold alarm within the next
H minutes", which uses only past/current features and therefore tests genuine early warning.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .event_evaluation import event_evaluation, pointwise_metrics, strip_private
from .features import feature_valid
from .preprocessing import future_label

SMOTE_RATIO = 0.1  # minority/majority after SMOTE
VARIANTS = ("none", "class_weight", "smote")


class OptionalDependencyError(RuntimeError):
    """Raised with a user-friendly message when an optional package is missing."""


def _import_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier
    except Exception as e:  # pragma: no cover - depends on environment
        raise OptionalDependencyError("xgboost is not installed (pip install xgboost) - supervised route skipped.") from e


def _smote(X: np.ndarray, y: np.ndarray, random_state: int):
    try:
        from imblearn.over_sampling import SMOTE
    except Exception as e:  # pragma: no cover
        raise OptionalDependencyError("imbalanced-learn is not installed (pip install imbalanced-learn) - SMOTE variant skipped.") from e
    n_pos = int(y.sum())
    if n_pos < 2:
        return X, y, 0
    cur = n_pos / max(int((~y.astype(bool)).sum()), 1)
    if cur >= 1.0:
        return X, y, 0
    # SMOTE can only *add* minority points, so the target ratio must exceed the current one
    ratio = SMOTE_RATIO if cur < SMOTE_RATIO else min(1.0, 2 * cur)
    k = max(1, min(5, n_pos - 1))
    sm = SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=random_state)
    Xr, yr = sm.fit_resample(X, y)
    return Xr, yr, int(len(yr) - len(y))


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Validation-only alert threshold.

    Starts from the F1-maximising threshold, then enforces an *alert budget*: no more than
    ``max(1 %, 1.5 x validation positive rate)`` of validation rows may alert. With only a handful of
    validation positives the raw F1 optimum is unstable and can alert on most rows.
    If there are no validation positives the budget quantile alone is used.
    """
    budget = max(0.01, 1.5 * float(y.mean())) if len(y) else 0.01
    cap = float(np.quantile(p, 1 - min(budget, 0.99))) if len(p) else 0.5
    if y.sum() == 0:
        return cap
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    best = float(th[int(np.nanargmax(f1))]) if len(th) else 0.5
    return max(best, cap)


def make_target(df: pd.DataFrame, feats: pd.DataFrame, cols: list[str], label: str, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Return (y, usable) for a label/horizon. horizon=0 -> label at the same minute."""
    ok = feature_valid(feats, df["row_valid"], cols)
    lab = df[f"label_{label}"]
    if horizon <= 0:
        return lab, ok
    y, usable = future_label(lab, df["row_valid"], horizon)
    return y, usable & ok


def train_classifier(
    df: pd.DataFrame,
    feats: pd.DataFrame,
    cols: list[str],
    label: str = "weak_vibration",
    horizon: int = 0,
    variants: tuple[str, ...] = VARIANTS,
    random_state: int = 42,
    n_estimators: int = 300,
    val_fraction: float = 0.2,
) -> dict[str, Any]:
    """Train each imbalance variant for one (label, horizon); evaluate on the held-out test split."""
    XGBClassifier = _import_xgb()
    y, usable = make_target(df, feats, cols, label, horizon)
    tr = (df["split"] == "train") & usable
    te = (df["split"] == "test") & usable
    tr_idx = df.index[tr]
    n_val = int(len(tr_idx) * val_fraction)
    fit_idx, val_idx = tr_idx[: len(tr_idx) - n_val], tr_idx[len(tr_idx) - n_val :]  # chronological: val is the latest slice

    Xfit, yfit = feats.loc[fit_idx, cols].to_numpy(float), y.loc[fit_idx].to_numpy(bool)
    Xval, yval = feats.loc[val_idx, cols].to_numpy(float), y.loc[val_idx].to_numpy(bool)
    Xte, yte = feats.loc[te, cols].to_numpy(float), y[te].to_numpy(bool)

    out: dict[str, Any] = {"label": label, "horizon": horizon, "cols": cols, "models": {}, "rows": []}
    counts = {
        "train_fit_rows": int(len(fit_idx)), "train_fit_pos": int(yfit.sum()),
        "val_rows": int(len(val_idx)), "val_pos": int(yval.sum()),
        "test_rows": int(len(yte)), "test_pos": int(yte.sum()),
    }
    out["counts"] = counts
    if yfit.sum() < 2:
        out["skipped"] = "Fewer than two positive training examples - supervised route not meaningful for this label."
        return out

    for var in variants:
        Xf, yf, n_syn = Xfit, yfit, 0
        spw = 1.0
        if var == "class_weight":
            spw = float((~yfit).sum() / max(yfit.sum(), 1))
        elif var == "smote":
            try:
                Xf, yf, n_syn = _smote(Xfit, yfit, random_state)
            except OptionalDependencyError as e:
                out["rows"].append({"variant": var, "skipped": str(e)})
                continue
        model = XGBClassifier(
            n_estimators=n_estimators, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist", random_state=random_state, n_jobs=-1,
        )
        model.fit(Xf, yf.astype(int))
        pval = model.predict_proba(Xval)[:, 1]
        thr = _best_f1_threshold(yval, pval)
        pte = model.predict_proba(Xte)[:, 1]
        pw = pointwise_metrics(yte, pte >= thr, pte)
        # val PR-AUC used for *selection* (never the test set)
        from .event_evaluation import curve_metrics
        val_pr = curve_metrics(yval, pval).get("pr_auc")

        # event-level: probabilities for every feature-valid test row vs horizon-0 label events
        ok_all = feature_valid(feats, df["row_valid"], cols) & (df["split"] == "test")
        prob_all = pd.Series(np.nan, index=df.index)
        prob_all[ok_all] = model.predict_proba(feats.loc[ok_all, cols].to_numpy(float))[:, 1]
        alert = (prob_all >= thr).fillna(False)
        ev = strip_private(event_evaluation(df[f"label_{label}"][ok_all].reindex(df.index, fill_value=False), alert, ok_all))
        out["rows"].append({
            "variant": var, "label": label, "horizon_min": horizon, "threshold": thr, "synthetic_samples": n_syn,
            "val_pr_auc": val_pr, **{f"pw_{k}": v for k, v in pw.items()}, **{f"ev_{k}": v for k, v in ev.items()},
        })
        out["models"][var] = {"model": model, "threshold": thr, "val_pr_auc": val_pr, "spw": spw, "n_synthetic": n_syn}
    return out


def pick_default(result: dict[str, Any]) -> str | None:
    """Pick the variant with the highest *validation* PR-AUC (ties -> first)."""
    best, best_v = None, -1.0
    for var, m in result.get("models", {}).items():
        v = m["val_pr_auc"] if m["val_pr_auc"] is not None else -1.0
        if v > best_v:
            best, best_v = var, v
    return best if best is not None else (next(iter(result.get("models", {})), None))


def save_classifier(result: dict[str, Any], variant: str, meta: dict[str, Any]) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    m = result["models"][variant]
    joblib.dump(
        {"model": m["model"], "threshold": m["threshold"], "feature_cols": result["cols"], "label": result["label"],
         "horizon_min": result["horizon"], "variant": variant, "meta": meta},
        C.MODELS_DIR / f"xgb_{result['label']}_h{result['horizon']}.joblib",
    )
