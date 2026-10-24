"""Optional advanced model: dense autoencoder 4 -> 16 -> 8 -> 4 (bottleneck) -> 8 -> 16 -> 4.

Trained only on healthy-period training rows (brief's conservative selection rules). The alert
threshold is the 99th percentile of reconstruction error on a separate, later normal validation
slice of the training period. Requires TensorFlow/Keras; if unavailable a clear error is raised
and the pipeline continues without this route.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config as C
from .event_evaluation import event_evaluation, pointwise_metrics, strip_private
from .features import feature_valid, raw_features
from .preprocessing import healthy_mask


class AutoencoderUnavailable(RuntimeError):
    pass


def _keras():
    try:
        import os
        if os.environ.get("CENTRIFUGE_SKIP_TF") == "1":
            raise ImportError("TensorFlow disabled via CENTRIFUGE_SKIP_TF=1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf
        return tf
    except Exception as e:  # pragma: no cover - environment dependent
        raise AutoencoderUnavailable("TensorFlow/Keras could not be loaded - using the scikit-learn MLP autoencoder instead.") from e


def build_autoencoder(tf, n_in: int):
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Input(shape=(n_in,)),
        L.Dense(16, activation="relu"), L.Dense(8, activation="relu"), L.Dense(4, activation="relu"),
        L.Dense(8, activation="relu"), L.Dense(16, activation="relu"), L.Dense(n_in, activation="linear"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


class _SklearnAE:
    """scikit-learn MLP autoencoder with the brief's 4-16-8-4-8-16-4 shape (used if TensorFlow cannot load)."""

    def __init__(self, seed: int, epochs: int):
        from sklearn.neural_network import MLPRegressor
        self.net = MLPRegressor(hidden_layer_sizes=(16, 8, 4, 8, 16), activation="relu", solver="adam", learning_rate_init=1e-3,
                                batch_size=512, max_iter=epochs, early_stopping=False, random_state=seed, tol=1e-6, n_iter_no_change=8)

    def fit(self, Z: np.ndarray) -> "_SklearnAE":
        self.net.fit(Z, Z)
        return self

    def predict(self, Z, batch_size=None, verbose=0):
        return self.net.predict(Z)


def recon_error(model, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Z = scaler.transform(X)
    R = model.predict(Z, batch_size=8192, verbose=0)
    return ((Z - R) ** 2).mean(axis=1)


def train_autoencoder(df: pd.DataFrame, feats: pd.DataFrame, epochs: int = 40, seed: int = 42) -> dict[str, Any]:
    try:
        tf = _keras()
        tf.keras.utils.set_random_seed(seed)
        backend = "keras"
    except AutoencoderUnavailable:
        tf, backend = None, "sklearn-mlp"  # TensorFlow unavailable -> same architecture via scikit-learn
    cols = raw_features()
    ok = feature_valid(feats, df["row_valid"], cols)
    healthy_tr = ok & (df["split"] == "train") & healthy_mask(df)
    idx = df.index[healthy_tr]
    if len(idx) < 2000:
        raise AutoencoderUnavailable("Not enough healthy training rows for the autoencoder.")
    n_val = int(len(idx) * 0.2)
    fit_idx, val_idx = idx[:-n_val], idx[-n_val:]  # later slice = separate normal validation period
    scaler = StandardScaler().fit(feats.loc[fit_idx, cols].to_numpy(float))
    Zfit = scaler.transform(feats.loc[fit_idx, cols].to_numpy(float))
    Zval = scaler.transform(feats.loc[val_idx, cols].to_numpy(float))
    if backend == "keras":
        model = build_autoencoder(tf, len(cols))
        es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
        hist = model.fit(Zfit, Zfit, validation_data=(Zval, Zval), epochs=epochs, batch_size=512, verbose=0, callbacks=[es])
        epochs_run, val_loss = len(hist.history["loss"]), float(hist.history["val_loss"][-1])
    else:
        model = _SklearnAE(seed, epochs * 5).fit(Zfit)
        epochs_run = int(model.net.n_iter_)
        val_loss = float(((Zval - model.predict(Zval)) ** 2).mean())
    val_err = recon_error(model, scaler, feats.loc[val_idx, cols].to_numpy(float))
    thr = float(np.quantile(val_err, 0.99))

    te = ok & (df["split"] == "test")
    err = pd.Series(np.nan, index=df.index, name="ae_error")
    err[ok] = recon_error(model, scaler, feats.loc[ok, cols].to_numpy(float))
    alert = (err >= thr).fillna(False)
    rows = []
    for lab in ("weak_vibration", "weak_combined", "strong_any"):
        y = df[f"label_{lab}"]
        pw = pointwise_metrics(y[te].to_numpy(), alert[te].to_numpy(), err[te].to_numpy())
        ev = strip_private(event_evaluation(y[te].reindex(df.index, fill_value=False), alert, te))
        rows.append({"label": lab, "threshold": thr, **{f"pw_{k}": v for k, v in pw.items()}, **{f"ev_{k}": v for k, v in ev.items()}})
    return {
        "model": model, "scaler": scaler, "threshold": thr, "error": err, "results": pd.DataFrame(rows), "backend": backend,
        "epochs_run": epochs_run, "final_val_loss": val_loss,
        "healthy_fit_rows": int(len(fit_idx)), "healthy_val_rows": int(len(val_idx)),
    }


def save_autoencoder(res: dict[str, Any]) -> None:
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if res.get("backend") == "keras":
        res["model"].save(C.MODELS_DIR / "autoencoder.keras")
    else:
        joblib.dump(res["model"].net, C.MODELS_DIR / "autoencoder_sklearn.joblib")
    joblib.dump({"scaler": res["scaler"], "threshold": res["threshold"], "feature_cols": raw_features(), "backend": res.get("backend")},
                C.MODELS_DIR / "autoencoder_meta.joblib")
