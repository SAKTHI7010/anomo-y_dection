"""Point-wise and event-level evaluation (safe with imbalance and zero-positive cases)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

MINUTES_PER_WEEK = 7 * 24 * 60


# ----------------------------------------------------------------------------- point-wise
def pointwise_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, Any]:
    """Precision/recall/F1/confusion matrix (+ PR-AUC if ``score`` given). Never divides by zero."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if (tp + fp) else None  # undefined when nothing predicted
    rec = tp / (tp + fn) if (tp + fn) else None  # undefined when no positives
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if (prec is not None and rec is not None) else None)
    out: dict[str, Any] = {
        "n": int(len(y_true)), "positives": int(y_true.sum()), "positive_rate_pct": float(y_true.mean() * 100) if len(y_true) else None,
        "predicted_positives": int(y_pred.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }
    if score is not None:
        out.update(curve_metrics(y_true, np.asarray(score, dtype=float)))
    return out


def curve_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(bool)
    ok = np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    if y_true.sum() == 0 or (~y_true).sum() == 0:
        return {"pr_auc": None, "roc_auc": None, "pr_baseline": float(y_true.mean()) if len(y_true) else None}
    return {
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_baseline": float(y_true.mean()),
    }


def pr_curve(y_true: np.ndarray, score: np.ndarray, max_points: int = 600):
    """Precision-recall curve arrays (downsampled for plotting); None if undefined."""
    y_true = np.asarray(y_true).astype(bool)
    ok = np.isfinite(score)
    y_true, score = y_true[ok], np.asarray(score)[ok]
    if y_true.sum() == 0:
        return None
    p, r, t = precision_recall_curve(y_true, score)
    if len(p) > max_points:
        idx = np.linspace(0, len(p) - 1, max_points).astype(int)
        p, r = p[idx], r[idx]
        t = t[np.clip(idx, 0, len(t) - 1)]
    return p, r, t


# ----------------------------------------------------------------------------- events
def merge_events(flag: pd.Series, max_gap_min: int = 30) -> pd.DataFrame:
    """Merge consecutive True minutes into events; gaps <= ``max_gap_min`` minutes are bridged.

    ``flag`` must be indexed by a datetime index. Returns start, end, duration_min, n_points.
    """
    f = flag.fillna(False).astype(bool)
    if not f.any():
        return pd.DataFrame(columns=["start", "end", "duration_min", "n_points"])
    t = f.index[f.to_numpy()]
    gaps = np.diff(t.values) / np.timedelta64(1, "m")
    new_event = np.concatenate([[True], gaps > max_gap_min])
    ev_id = np.cumsum(new_event) - 1
    grp = pd.DataFrame({"t": t, "ev": ev_id}).groupby("ev")["t"]
    ev = pd.DataFrame({"start": grp.min(), "end": grp.max(), "n_points": grp.size()}).reset_index(drop=True)
    ev["duration_min"] = ((ev["end"] - ev["start"]) / pd.Timedelta(minutes=1) + 1).astype(int)
    return ev


def event_evaluation(
    label: pd.Series,
    alert: pd.Series,
    valid: pd.Series,
    merge_gap_min: int = 30,
    lookback_min: int = 120,
    tolerance_min: int = 30,
) -> dict[str, Any]:
    """Event-level metrics comparing model alerts with threshold-derived label events.

    Definitions (all documented in the README):
    * label events / alert events: consecutive flagged minutes merged when gaps <= ``merge_gap_min``.
    * A label event is *detected* if an alert occurs within [start - lookback, end].
    * lead time = event start - first alert in that window (minutes). >0 early, 0 concurrent,
      <0 late detection (alert only after the threshold alarm already started).
    * A model alert event is a *false alarm* if it does not fall within [label start - lookback,
      label end + tolerance] of any label event.
    * False alarms per week uses the total valid observation time of the evaluated period.
    """
    label = label & valid
    alert = alert & valid
    lab_ev = merge_events(label, merge_gap_min)
    al_ev = merge_events(alert, merge_gap_min)
    weeks = float(valid.sum()) / MINUTES_PER_WEEK if valid.sum() else np.nan

    rows = []
    alert_idx = alert.index[alert.to_numpy()]
    for _, ev in lab_ev.iterrows():
        w0, w1 = ev["start"] - pd.Timedelta(minutes=lookback_min), ev["end"]
        i0, i1 = alert_idx.searchsorted(w0), alert_idx.searchsorted(w1, side="right")
        hits = alert_idx[i0:i1]
        if len(hits):
            lead = (ev["start"] - hits[0]) / pd.Timedelta(minutes=1)
            rows.append({**ev.to_dict(), "detected": True, "first_alert": hits[0], "lead_min": float(lead)})
        else:
            rows.append({**ev.to_dict(), "detected": False, "first_alert": pd.NaT, "lead_min": np.nan})
    per_event = pd.DataFrame(rows, columns=["start", "end", "duration_min", "n_points", "detected", "first_alert", "lead_min"])

    fa_rows = []
    if len(al_ev):
        if len(lab_ev):
            ls = lab_ev["start"].to_numpy() - np.timedelta64(lookback_min, "m")
            le = lab_ev["end"].to_numpy() + np.timedelta64(tolerance_min, "m")
            a_s = al_ev["start"].to_numpy()[:, None]
            a_e = al_ev["end"].to_numpy()[:, None]
            matched = ((a_e >= ls[None, :]) & (a_s <= le[None, :])).any(axis=1)
        else:
            matched = np.zeros(len(al_ev), dtype=bool)
        fa = int((~matched).sum())
        fa_rows = al_ev[~matched].to_dict("records")
    else:
        fa = 0

    n_lab = int(len(lab_ev))
    detected = int(per_event["detected"].sum()) if n_lab else 0
    early = per_event[per_event["lead_min"] > 0]["lead_min"] if n_lab else pd.Series(dtype=float)
    out = {
        "label_events": n_lab,
        "alert_events": int(len(al_ev)),
        "events_detected": detected,
        "event_recall": (detected / n_lab) if n_lab else None,
        "events_early_warning": int(len(early)),
        "median_lead_min": float(early.median()) if len(early) else None,
        "mean_lead_min": float(early.mean()) if len(early) else None,
        "false_alarm_events": int(fa),
        "alert_event_precision": ((len(al_ev) - fa) / len(al_ev)) if len(al_ev) else None,
        "observed_weeks": weeks,
        "false_alarms_per_week": (fa / weeks) if weeks and weeks == weeks else None,
        "merge_gap_min": merge_gap_min, "lookback_min": lookback_min, "tolerance_min": tolerance_min,
    }
    out["_per_event"] = per_event
    out["_false_alarms"] = pd.DataFrame(fa_rows)
    return out


def strip_private(d: dict[str, Any]) -> dict[str, Any]:
    """Drop DataFrame entries (keys starting with '_') so the dict is JSON serialisable."""
    return {k: v for k, v in d.items() if not k.startswith("_")}
