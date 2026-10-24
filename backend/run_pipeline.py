"""End-to-end pipeline: validate -> preprocess -> features -> models -> evaluate -> persist artefacts.

Run from the ``backend`` folder:

    python run_pipeline.py            # full run (Isolation Forest, XGBoost +/- SMOTE, autoencoder, residual twin)
    python run_pipeline.py --fast     # skip autoencoder, fewer trees (smoke run)

Everything written is derived from the real dataset; nothing is simulated.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # avoids joblib's wmic probe on Windows

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from src import config as C  # noqa: E402
from src.data_loader import DatasetNotFoundError, load_raw  # noqa: E402
from src.event_evaluation import event_evaluation, pointwise_metrics, pr_curve  # noqa: E402
from src.features import build_features  # noqa: E402
from src.health_score import HealthScorer  # noqa: E402
from src.preprocessing import PrepInfo, add_labels, assign_split, chronological_split, prepare  # noqa: E402
from src.residual_model import apply_residual_model, fit_residual_model  # noqa: E402
from src.train_classifier import OptionalDependencyError, pick_default, save_classifier, train_classifier  # noqa: E402
from src.train_isolation_forest import save_model as save_if, score_all as if_score_all, train_isolation_forest  # noqa: E402
from src.validation import save_report, validate  # noqa: E402

SCORED_PATH = C.PROCESSED_DIR / "scored.parquet"
SHAP_PATH = C.PROCESSED_DIR / "shap_sample.parquet"
SUMMARY_PATH = C.METRICS_DIR / "summary.json"
DEFAULT_IF_CONTAMINATION = 0.01
XGB_RUNS = (("weak_vibration", 30), ("weak_vibration", 0), ("strong_any", 0))
DEFAULT_XGB = ("weak_vibration", 30)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _event_details(model: str, label: str, alert: pd.Series, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    y = df[f"label_{label}"][mask].reindex(df.index, fill_value=False)
    ev = event_evaluation(y, alert, mask)
    t = ev["_per_event"].copy()
    t.insert(0, "label", label)
    t.insert(0, "model", model)
    return t


def compute_shap(model, feats: pd.DataFrame, cols: list[str], df: pd.DataFrame, y: pd.Series, usable: pd.Series, n: int = 2500) -> dict:
    """SHAP TreeExplainer on a test sample (all positives + random negatives). Skipped if shap is missing."""
    try:
        import shap
    except Exception:
        log("shap not installed - skipping SHAP (pip install shap).")
        return {}
    te = (df["split"] == "test") & usable
    pos = df.index[te & y]
    neg = df.index[te & ~y]
    rng = np.random.default_rng(0)
    neg_s = rng.choice(neg, size=min(n - min(len(pos), n // 2), len(neg)), replace=False)
    pos_s = pos if len(pos) <= n // 2 else rng.choice(pos, size=n // 2, replace=False)
    idx = pd.DatetimeIndex(pos_s).union(pd.DatetimeIndex(neg_s))
    X = feats.loc[idx, cols]
    sv = shap.TreeExplainer(model).shap_values(X.to_numpy(float))
    sv = np.asarray(sv)
    out = pd.DataFrame(sv, index=idx, columns=[f"shap__{c}" for c in cols]).join(X.add_prefix("val__"))
    out["target"] = y.loc[idx].astype(int)
    out["prob"] = model.predict_proba(X.to_numpy(float))[:, 1]
    out.to_parquet(SHAP_PATH)
    return {"shap_rows": int(len(out)), "shap_positives": int(out["target"].sum())}


def main(fast: bool = False, skip_ae: bool = False) -> dict:
    t0 = time.time()
    for d in (C.PROCESSED_DIR, C.MODELS_DIR, C.METRICS_DIR, C.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
    summary: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version()}

    # 1. load + validate -----------------------------------------------------------------
    log("Loading dataset ...")
    raw = load_raw()
    rep = validate(raw)
    save_report(rep)
    summary["data_quality_warnings"] = rep["warnings"]
    log(f"Rows {rep['rows']:,}; {rep['timestamp_min']} -> {rep['timestamp_max']}; gaps={rep['gap_count']}; missing={rep['missing_values']}")

    # 2. preprocess + labels + split -----------------------------------------------------------
    info = PrepInfo()
    df = add_labels(prepare(raw, info))
    cutoff, split_note = chronological_split(df)
    df["split"] = assign_split(df, cutoff)
    summary["prep_decisions"] = info.decisions
    summary["split"] = {"cutoff": str(cutoff), "note": split_note,
                        "train_valid_rows": int((df["row_valid"] & (df["split"] == "train")).sum()),
                        "test_valid_rows": int((df["row_valid"] & (df["split"] == "test")).sum())}
    label_stats = {}
    for lab in ("strong_any", "weak_vibration", "weak_combined"):
        for sp in ("train", "test"):
            m = df["row_valid"] & (df["split"] == sp)
            label_stats[f"{lab}__{sp}"] = {"valid_rows": int(m.sum()), "positives": int(df.loc[m, f"label_{lab}"].sum()),
                                           "rate_pct": float(df.loc[m, f"label_{lab}"].mean() * 100)}
    summary["label_stats"] = label_stats
    log(f"Split @ {cutoff} ({split_note}). Label stats: " + ", ".join(f"{k}={v['positives']}" for k, v in label_stats.items()))

    feats = build_features(df)
    log(f"Features built: {feats.shape[1]} columns")

    # 3. health score -------------------------------------------------------------------------
    scorer = HealthScorer().fit(df[df["split"] == "train"])
    hs = scorer.score(df)
    df = df.join(hs)
    (C.MODELS_DIR / "health_scorer.json").write_text(json.dumps(scorer.to_dict(), indent=2))
    summary["health_scorer"] = scorer.to_dict()

    # 4. Isolation Forest ---------------------------------------------------------------------
    log("Training Isolation Forest variants ...")
    ifres = train_isolation_forest(df, feats, n_estimators=100 if fast else 300)
    ifres["results"].to_csv(C.METRICS_DIR / "isolation_forest_results.csv", index=False)
    art = ifres["default_art"]
    ifs, ifthr = if_score_all(art, df, feats, DEFAULT_IF_CONTAMINATION)
    df["if_score"] = ifs
    df["alert_if"] = (ifs >= ifthr).fillna(False)
    save_if(art, ifthr, DEFAULT_IF_CONTAMINATION, {"features": art["cols"], "default_key": list(ifres["default"]), "cutoff": str(cutoff)})
    summary["isolation_forest_default"] = {"features": list(ifres["default"]), "contamination": DEFAULT_IF_CONTAMINATION, "threshold": ifthr}
    log(f"Isolation Forest done. Threshold={ifthr:.4f}")

    te_if = (df["split"] == "test") & df["if_score"].notna()
    details = [_event_details("isolation_forest", lab, df["alert_if"], df, te_if) for lab in ("weak_vibration", "weak_combined", "strong_any")]
    default_row = ifres["results"].query(
        "features=='selected' and train_rows=='all_valid' and contamination==@DEFAULT_IF_CONTAMINATION"
    )
    summary["isolation_forest_default_metrics"] = json.loads(default_row.to_json(orient="records"))

    # PR curve data for the default IF model (test split)
    pr_store = {}
    for lab in ("weak_vibration", "weak_combined", "strong_any"):
        crv = pr_curve(df.loc[te_if, f"label_{lab}"].to_numpy(), df.loc[te_if, "if_score"].to_numpy())
        if crv is not None:
            pr_store[f"isolation_forest__{lab}"] = pd.DataFrame({"precision": crv[0], "recall": crv[1]})

    # 5. XGBoost (+ SMOTE comparison) ---------------------------------------------------------
    xgb_rows = []
    feat_cols = list(feats.columns)
    try:
        for lab, h in XGB_RUNS:
            log(f"XGBoost label={lab} horizon={h} (none / class_weight / SMOTE) ...")
            res = train_classifier(df, feats, feat_cols, label=lab, horizon=h, n_estimators=100 if fast else 300)
            if "skipped" in res:
                log(f"  skipped: {res['skipped']}")
                continue
            for r in res["rows"]:
                r.update({"counts": json.dumps(res["counts"])})
            xgb_rows += res["rows"]
            best = pick_default(res)
            save_classifier(res, best, {"cutoff": str(cutoff)})
            log(f"  default variant by validation PR-AUC: {best}")
            if (lab, h) == DEFAULT_XGB and best:
                m = res["models"][best]
                from src.train_classifier import make_target
                y, usable = make_target(df, feats, feat_cols, lab, h)
                ok = (df["split"] == "test") & usable
                prob = pd.Series(np.nan, index=df.index)
                from src.features import feature_valid
                okf = feature_valid(feats, df["row_valid"], feat_cols)
                prob[okf] = m["model"].predict_proba(feats.loc[okf, feat_cols].to_numpy(float))[:, 1]
                df["xgb_prob"] = prob
                df["alert_xgb"] = (prob >= m["threshold"]).fillna(False)
                summary["xgb_default"] = {"label": lab, "horizon_min": h, "variant": best, "threshold": m["threshold"],
                                          "n_synthetic": m["n_synthetic"]}
                te_x = (df["split"] == "test") & okf
                details.append(_event_details("xgboost", lab, df["alert_xgb"], df, te_x))
                crv = pr_curve(y[ok].to_numpy(), prob[ok].to_numpy())
                if crv is not None:
                    pr_store[f"xgboost__{lab}_h{h}"] = pd.DataFrame({"precision": crv[0], "recall": crv[1]})
                summary["xgb_default"].update(compute_shap(m["model"], feats, feat_cols, df, y, usable))
    except OptionalDependencyError as e:
        log(f"Supervised route skipped: {e}")
        summary["xgb_skipped"] = str(e)
    if xgb_rows:
        pd.DataFrame(xgb_rows).to_csv(C.METRICS_DIR / "xgboost_results.csv", index=False)
    if "xgb_prob" not in df:
        df["xgb_prob"] = np.nan
        df["alert_xgb"] = False

    # 6. Autoencoder (optional) ---------------------------------------------------------------
    df["ae_error"] = np.nan
    df["alert_ae"] = False
    if not (fast or skip_ae):
        try:
            from src.train_autoencoder import AutoencoderUnavailable, save_autoencoder, train_autoencoder
            log("Training autoencoder ...")
            ae = train_autoencoder(df, feats)
            df["ae_error"] = ae["error"]
            df["alert_ae"] = (ae["error"] >= ae["threshold"]).fillna(False)
            ae["results"].to_csv(C.METRICS_DIR / "autoencoder_results.csv", index=False)
            save_autoencoder(ae)
            summary["autoencoder"] = {"threshold": ae["threshold"], "epochs": ae["epochs_run"], "val_loss": ae["final_val_loss"], "backend": ae["backend"],
                                      "healthy_fit_rows": ae["healthy_fit_rows"], "healthy_val_rows": ae["healthy_val_rows"]}
            te_a = (df["split"] == "test") & df["ae_error"].notna()
            details += [_event_details("autoencoder", lab, df["alert_ae"], df, te_a) for lab in ("weak_vibration", "weak_combined", "strong_any")]
        except Exception as e:  # missing TF or any runtime issue must not crash the pipeline
            log(f"Autoencoder skipped: {e}")
            summary["autoencoder_skipped"] = str(e)

    # 7. torque -> temperature residual twin ----------------------------------------------------
    log("Fitting torque-temperature residual model ...")
    rfit = fit_residual_model(df)
    res_df = apply_residual_model(df, rfit)
    df = df.join(res_df)
    if "resid_flag_any" in df:
        te_r = (df["split"] == "test") & df["row_valid"]
        for lab in ("weak_vibration", "strong_any"):
            pw = pointwise_metrics(df.loc[te_r, f"label_{lab}"].to_numpy(), df.loc[te_r, "resid_flag_any"].to_numpy())
            ev = event_evaluation(df[f"label_{lab}"][te_r].reindex(df.index, fill_value=False), df["resid_flag_any"], te_r)
            summary.setdefault("residual_vs_labels", {})[lab] = {
                "pointwise": pw, "events_detected": ev["events_detected"], "label_events": ev["label_events"],
                "alert_events": ev["alert_events"], "false_alarm_events": ev["false_alarm_events"],
                "false_alarms_per_week": ev["false_alarms_per_week"], "median_lead_min": ev["median_lead_min"],
            }
        summary["residual_thresholds"] = rfit["thresholds"]

    # 8. persist ---------------------------------------------------------------------------------
    pd.concat(details, ignore_index=True).to_csv(C.METRICS_DIR / "event_details.csv", index=False)
    for k, v in pr_store.items():
        v.to_csv(C.METRICS_DIR / f"pr_curve__{k}.csv", index=False)
    keep = (
        C.CHANNELS + ["row_valid", "running", "split"]
        + [f"label_{x}" for x in ("strong_any", "weak_vibration", "weak_combined")]
        + [f"exceed_{c}" for c in C.CHANNELS]
        + ["health", "norm_vibration", "norm_torque", "norm_temp_max", "if_score", "alert_if", "xgb_prob", "alert_xgb", "ae_error", "alert_ae"]
        + [c for c in df.columns if c.startswith("resid_")]
    )
    out = df[keep].copy()
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    out.to_parquet(SCORED_PATH)
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_s"] = round(time.time() - t0, 1)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log(f"Done in {summary['elapsed_s']} s. Artefacts: {SCORED_PATH}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="smoke run: fewer trees, no autoencoder")
    ap.add_argument("--skip-ae", action="store_true", help="skip the optional autoencoder")
    ap.add_argument("--no-tf", action="store_true", help="do not try to import TensorFlow (uses the scikit-learn MLP autoencoder)")
    a = ap.parse_args()
    if a.no_tf:
        os.environ["CENTRIFUGE_SKIP_TF"] = "1"
    try:
        main(fast=a.fast, skip_ae=a.skip_ae)
    except DatasetNotFoundError as e:
        print(e)
        sys.exit(2)
