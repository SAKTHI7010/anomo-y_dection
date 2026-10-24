"""Dataset discovery and loading.

The original Excel file is never modified. A parquet cache of the *unchanged* values is
written to ``data/processed`` so that the (slow) Excel parse happens only once.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config as C


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Sulzer CFG-42 dataset cannot be located."""


MISSING_MSG = (
    "Sulzer CFG-42 dataset not found.\n"
    f"1. Download it from {C.DATASET_DOI}\n"
    f"2. Place the file (one of {', '.join(C.DATASET_FILENAMES)}) in:  {C.RAW_DIR}\n"
    "3. Re-run the app / pipeline. No synthetic data is substituted silently."
)


def find_dataset() -> Path:
    """Look in data/raw first, then the project root. Returns the first match."""
    for folder in (C.RAW_DIR, C.PROJECT_ROOT):
        for name in C.DATASET_FILENAMES:
            p = folder / name
            if p.exists():
                return p
        if folder.exists():  # tolerate renamed variants such as 'CFG_42_EDS_Data (2).xlsx'
            for p in sorted(folder.glob("CFG_42_EDS_Data*")):
                if p.suffix.lower() in (".xlsx", ".csv"):
                    return p
    raise DatasetNotFoundError(MISSING_MSG)


def _read_excel_fast(path: Path) -> pd.DataFrame:
    """Stream the workbook with openpyxl (much faster than pandas.read_excel on 500k rows)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    ncols = max(i for i, h in enumerate(header) if h is not None) + 1
    next(rows)  # second row holds human-readable descriptions, not data
    data = [r[:ncols] for r in rows]
    wb.close()
    df = pd.DataFrame(data, columns=[str(h) for h in header[:ncols]])
    return df


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = _read_excel_fast(path)
    else:
        df = pd.read_csv(path)
        if len(df) and not str(df.iloc[0, 0])[:2].isdigit():  # description row
            df = df.iloc[1:].reset_index(drop=True)
    return df


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename tags to channel names, parse timestamps strictly, coerce numerics (no value edits)."""
    ts_col = df.columns[0]
    missing = [t for t in C.TAG_TO_CHANNEL if t not in df.columns]
    if missing:
        raise ValueError(f"Expected sensor columns missing from dataset: {missing}. Found: {list(df.columns)}")
    out = df[[ts_col] + list(C.TAG_TO_CHANNEL)].rename(columns={ts_col: "timestamp", **C.TAG_TO_CHANNEL})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for c in C.CHANNELS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out[["timestamp"] + C.CHANNELS]


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Return the dataset exactly as supplied (unsorted/with any gaps or duplicates intact)."""
    src = find_dataset()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = C.PROCESSED_DIR / "cfg42_raw.parquet"
    meta = C.PROCESSED_DIR / "cfg42_raw.meta.json"
    sig = {"name": src.name, "size": src.stat().st_size, "mtime": int(src.stat().st_mtime)}
    if use_cache and cache.exists() and meta.exists():
        try:
            if json.loads(meta.read_text()) == sig:
                return pd.read_parquet(cache)
        except Exception:
            pass  # fall through and rebuild
    df = _standardise(_read_raw(src))
    df.to_parquet(cache, index=False)
    meta.write_text(json.dumps(sig))
    return df
