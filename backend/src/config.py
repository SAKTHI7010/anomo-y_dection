"""Central configuration for the Centrifuge Health Digital Twin (Project 2).

All plant thresholds here are the *plant-specific* values quoted in the project brief.
They are candidate-anomaly rules, NOT universal limits or validated shutdown values.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_DOI = "https://doi.org/10.5281/zenodo.17885723"
DATASET_NAME = "Multisensor Condition Monitoring Dataset from a Sulzer Industrial Centrifuge"
DATASET_FILENAMES = ("CFG_42_EDS_Data.xlsx", "CFG_42_EDS_Data (1).xlsx", "CFG_42_EDS_Data.csv")

# Raw tag -> short channel name
TAG_TO_CHANNEL = {
    "Analogue_Signals.30-NI-3271-A": "torque",
    "Analogue_Signals.30-VI-3274-A": "vibration",
    "Analogue_Signals.30-TI-3273": "temp_solid",
    "Analogue_Signals.30-TI-3275": "temp_liquid",
}
CHANNELS = ["vibration", "torque", "temp_solid", "temp_liquid"]

CHANNEL_INFO = {
    "vibration": {"label": "Vibration velocity", "unit": "mm/s RMS", "threshold": 12.0},
    "torque": {"label": "Torque", "unit": "% rated load", "threshold": 85.0},
    "temp_solid": {"label": "Bearing temp - solid end", "unit": "°C", "threshold": 85.0},
    "temp_liquid": {"label": "Bearing temp - liquid end", "unit": "°C", "threshold": 85.0},
}
THRESHOLDS = {c: v["threshold"] for c, v in CHANNEL_INFO.items()}

# Conservative "healthy period" selection rules (brief) - training-selection only.
HEALTHY_RULES = {"vibration": 8.0, "torque": 70.0, "temp_solid": 75.0, "temp_liquid": 75.0}

EXPECTED_FREQ = "1min"
TRAIN_FRACTION_MONTHS = 9  # first 9 months train, remainder test (if coverage supports it)

ROLL_WINDOWS = (5, 15, 60)

# Health score (brief): weights configurable, must sum to 1.0
HEALTH_WEIGHTS = {"vibration": 0.5, "torque": 0.3, "temp_max": 0.2}
HEALTH_STATES = [  # (lower bound inclusive, name, colour, interpretation)
    (70.0, "Healthy", "#22c55e", "Normal trend monitoring"),
    (40.0, "Attention", "#eab308", "Increase monitoring / review trend"),
    (20.0, "Warning", "#f97316", "Maintenance review recommended"),
    (-1e9, "Critical", "#ef4444", "Urgent human inspection / follow plant procedure"),
]

DISCLAIMER = (
    "Decision-support prototype built on **historical** sensor records. Thresholds are plant-specific "
    "values from the project brief, labels are *weak* (threshold-derived), and the health score is an "
    "unvalidated indicator. Not a certified safety system - plant procedures and engineering review govern real operations."
)
