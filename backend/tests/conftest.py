"""Pytest suite. Tiny synthetic frames are used ONLY here, to unit-test logic with known answers.
Real-data integration checks run when the pipeline artefacts exist."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src import config as C  # noqa: E402


def make_frame(n: int = 600, seed: int = 0, start: str = "2023-01-01") -> pd.DataFrame:
    """Small plausible frame on a 1-min grid with the dataset's column layout."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min", name="timestamp")
    return pd.DataFrame({
        "vibration": rng.normal(6, 0.8, n).clip(0.1),
        "torque": rng.normal(75, 4, n).clip(0),
        "temp_solid": rng.normal(66, 2, n),
        "temp_liquid": rng.normal(60, 2, n),
    }, index=idx)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture()
def prepared(frame):
    from src.preprocessing import add_labels, assign_split, prepare
    raw = frame.reset_index()
    df = add_labels(prepare(raw))
    df["split"] = assign_split(df, df.index[int(len(df) * 0.7)])
    return df
