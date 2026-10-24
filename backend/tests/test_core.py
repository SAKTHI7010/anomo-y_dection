from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.event_evaluation import event_evaluation, merge_events, pointwise_metrics
from src.features import build_features
from src.health_score import HealthScorer, forecast_linear, health_state, states_series
from src.preprocessing import add_labels, chronological_split, future_label, prepare
from src.train_classifier import _smote


# --------------------------------------------------------------------------- loading
def test_missing_dataset_message(monkeypatch, tmp_path):
    from src import data_loader as dl
    monkeypatch.setattr(C, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(C, "PROJECT_ROOT", tmp_path)
    with pytest.raises(dl.DatasetNotFoundError) as e:
        dl.find_dataset()
    assert "zenodo" in str(e.value).lower() and "data" in str(e.value).lower()


# --------------------------------------------------------------------------- preprocessing
def test_prepare_reindexes_gaps_and_flags_invalid(frame):
    f = frame.copy()
    f = f.drop(f.index[100:110])  # 10-minute gap
    f.iloc[200:210, f.columns.get_loc("temp_liquid")] = 0.0  # sensor outage as 0 °C
    out = prepare(f.reset_index())
    assert len(out) == len(frame)  # grid restored, nothing interpolated
    assert out.loc[frame.index[100:110], C.CHANNELS].isna().all().all()
    assert not out["row_valid"].iloc[100:110].any()
    assert not out.loc[frame.index[210:220], "valid_temp_liquid"].any()  # f.iloc[200:210] == frame.index[210:220]
    assert out["row_valid"].iloc[0]


def test_prepare_drops_duplicates_and_sorts(frame):
    f = frame.reset_index()
    f = pd.concat([f.iloc[::-1], f.iloc[:5]])
    out = prepare(f)
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()


def test_labels_threshold_semantics(frame):
    f = frame.copy()
    f.iloc[10, f.columns.get_loc("vibration")] = 12.0  # == threshold counts
    f.iloc[20, f.columns.get_loc("torque")] = 90.0
    df = add_labels(prepare(f.reset_index()))
    assert df["label_weak_vibration"].iloc[10]
    assert df["label_strong_any"].iloc[10] and df["label_strong_any"].iloc[20]
    assert not df["label_weak_vibration"].iloc[20]


def test_split_is_chronological_and_reproducible(prepared):
    c1, _ = chronological_split(prepared)
    c2, _ = chronological_split(prepared)
    assert c1 == c2
    tr = prepared.index[prepared.index < c1]
    te = prepared.index[prepared.index >= c1]
    assert tr.max() < te.min()


def test_future_label_excludes_active_and_unknown_futures():
    idx = pd.date_range("2023-01-01", periods=10, freq="1min")
    label = pd.Series([False] * 10, index=idx)
    label.iloc[6] = True
    valid = pd.Series(True, index=idx)
    valid.iloc[8] = False
    y, usable = future_label(label, valid, horizon=3)
    assert y.iloc[4] and y.iloc[5]          # alarm at t+2 / t+1
    assert not y.iloc[1]                    # alarm too far ahead
    assert not usable.iloc[6]               # alarm already active
    assert not usable.iloc[7]               # future window contains an invalid minute
    assert usable.iloc[0]


# --------------------------------------------------------------------------- features
def test_rolling_features_are_causal(frame):
    f = frame
    full = build_features(f.assign(row_valid=True))
    cut = 300
    trunc = build_features(f.iloc[: cut + 1].assign(row_valid=True))
    a = full.iloc[: cut + 1].fillna(-999)
    b = trunc.fillna(-999)
    pd.testing.assert_frame_equal(a, b)  # future rows never change past features


def test_features_warmup_nan_and_division_guard(frame):
    f = frame.copy()
    f.iloc[50, f.columns.get_loc("vibration")] = 0.0
    feats = build_features(f)
    assert feats["vibration_mean_60"].iloc[:59].isna().all()
    assert feats["vibration_mean_60"].iloc[59:].notna().all()
    assert np.isfinite(feats["torque_per_vib"].iloc[50])


# --------------------------------------------------------------------------- health
def test_health_bounded_with_edge_cases(prepared):
    sc = HealthScorer().fit(prepared)
    edge = prepared.iloc[:6].copy()
    edge.loc[edge.index[0], ["vibration", "torque", "temp_solid", "temp_liquid"]] = [0, 0, 0, 0]
    edge.loc[edge.index[1], ["vibration", "torque", "temp_solid", "temp_liquid"]] = [1e6, 1e6, 1e6, 1e6]
    edge.loc[edge.index[2], ["vibration", "torque", "temp_solid", "temp_liquid"]] = [-5, -5, -5, -5]
    edge.loc[edge.index[3], "vibration"] = np.nan
    h = sc.score(edge)["health"]
    ok = h.dropna()
    assert ((ok >= 0) & (ok <= 100)).all() and np.isfinite(ok).all()
    assert h.iloc[1] == pytest.approx(0.0)  # worst case
    assert h.iloc[0] == pytest.approx(100.0)  # clamped best case
    assert np.isnan(h.iloc[3])  # missing input -> no score, not a fake number


def test_health_weights_are_normalised(prepared):
    sc = HealthScorer(weights={"vibration": 5, "torque": 3, "temp_max": 2}).fit(prepared)
    h = sc.score(prepared)["health"].dropna()
    assert h.min() >= 0 and h.max() <= 100


def test_state_bands():
    assert health_state(95)[0] == "Healthy" and health_state(70)[0] == "Attention"
    assert health_state(40)[0] == "Attention" and health_state(39.9)[0] == "Warning"
    assert health_state(20)[0] == "Warning" and health_state(19.9)[0] == "Critical"
    assert health_state(float("nan"))[0] == "No data"
    s = states_series(pd.Series([100, 55, 30, 5, np.nan]))
    assert list(s) == ["Healthy", "Attention", "Warning", "Critical", "No data"]


def test_forecast_needs_enough_data_and_is_bounded():
    idx = pd.date_range("2023-01-01", periods=24 * 60, freq="1min")
    h = pd.Series(np.linspace(90, 60, len(idx)), index=idx)
    fc = forecast_linear(h)
    assert fc is not None and len(fc["mean"]) == 24
    assert (fc["lo"] >= 0).all() and (fc["hi"] <= 100).all()
    assert forecast_linear(h.iloc[:30]) is None
    assert forecast_linear(pd.Series(dtype=float)) is None


# --------------------------------------------------------------------------- metrics
def test_pointwise_metrics_no_positives_and_no_predictions():
    r = pointwise_metrics(np.zeros(100, bool), np.zeros(100, bool), np.random.rand(100))
    assert r["precision"] is None and r["recall"] is None and r["pr_auc"] is None
    r = pointwise_metrics(np.array([1, 0, 0, 1], bool), np.zeros(4, bool), np.array([.9, .1, .2, .8]))
    assert r["precision"] is None and r["recall"] == 0 and r["pr_auc"] == pytest.approx(1.0)


def test_merge_events_gap_rule():
    idx = pd.date_range("2023-01-01", periods=100, freq="1min")
    f = pd.Series(False, index=idx)
    f.iloc[[10, 11, 12, 20, 70]] = True
    assert len(merge_events(f, 5)) == 3
    assert len(merge_events(f, 10)) == 2  # 12->20 bridged
    assert len(merge_events(pd.Series(False, index=idx))) == 0


def test_event_evaluation_lead_time_and_false_alarms():
    idx = pd.date_range("2023-01-01", periods=2000, freq="1min")
    label = pd.Series(False, index=idx)
    label.iloc[500:510] = True
    alert = pd.Series(False, index=idx)
    alert.iloc[480:485] = True    # 20 min early
    alert.iloc[1500:1505] = True  # false alarm
    valid = pd.Series(True, index=idx)
    ev = event_evaluation(label, alert, valid, merge_gap_min=5, lookback_min=60)
    assert ev["label_events"] == 1 and ev["events_detected"] == 1
    assert ev["median_lead_min"] == pytest.approx(20.0)
    assert ev["false_alarm_events"] == 1 and ev["false_alarms_per_week"] > 0
    empty = event_evaluation(label & False, alert, valid)
    assert empty["event_recall"] is None


# --------------------------------------------------------------------------- SMOTE
def test_smote_adds_only_minority_rows_and_handles_edge_cases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 4))
    y = np.zeros(1000, bool)
    y[:20] = True
    Xr, yr, n = _smote(X, y, 0)
    assert n > 0 and len(yr) == len(y) + n
    assert yr.sum() - y.sum() == n                 # only positives were synthesised
    assert np.array_equal(Xr[:1000], X)            # originals untouched
    X2, y2, n2 = _smote(X, np.zeros(1000, bool), 0)  # no positives -> unchanged
    assert n2 == 0 and len(y2) == 1000
    yb = np.zeros(1000, bool)
    yb[:300] = True                                # not rare: ratio must still be valid
    _, _, n3 = _smote(X, yb, 0)
    assert n3 > 0


def test_classifier_smote_uses_training_rows_only(prepared):
    from src.train_classifier import train_classifier
    df = prepared.copy()
    feats = build_features(df)
    # create labelled events in the TRAIN portion only
    tr_idx = df.index[(df["split"] == "train")][100:400]
    df.loc[tr_idx[::37], "label_weak_vibration"] = True
    res = train_classifier(df, feats, ["vibration", "torque", "temp_solid", "temp_liquid"], label="weak_vibration", horizon=0,
                           variants=("smote",), n_estimators=10)
    assert res["rows"] and "pw_pr_auc" in res["rows"][0]
    assert res["counts"]["test_pos"] == 0 and res["rows"][0]["synthetic_samples"] > 0
