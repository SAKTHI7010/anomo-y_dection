# Centrifuge Anomaly Detection & Equipment Health Digital Twin ⚙️

An end-to-end industrial machine learning and equipment-health monitoring dashboard for the **Sulzer CFG-42 industrial centrifuge**, built using **Streamlit**, **Plotly**, **Scikit-Learn**, **XGBoost**, and **SHAP**.

---

## 🚀 Quickstart

### 1. Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/SAKTHI7010/anomo-y_dection.git
cd anomo-y_dection
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
Launch the interactive web dashboard:
```bash
streamlit run frontend/app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Dashboard Pages & Detailed Explanation

The application contains 5 dedicated analytical views accessible from the sidebar:

### 1. 🩺 Health Monitor
- **Core Purpose**: Real-time style equipment health monitor providing an instantaneous assessment of centrifuge operating status by replaying recorded historical telemetry.
- **Key Capabilities**:
  - **As-of Time Slider**: Replay the equipment state at any point in history.
  - **Equipment Health Score Gauge (0–100)**: Composite index broken down into standard operational tiers:
    - 🟢 **Healthy (70–100)**: Normal steady-state operation.
    - 🟡 **Attention (40–70)**: Minor degradation or thermal/vibration drift.
    - 🟠 **Warning (20–40)**: Elevated distress or sustained parameter departures.
    - 🔴 **Critical (0–20)**: Severe anomaly requiring immediate maintenance intervention.
  - **4-Channel Sensor Grid**: Real-time synchronized time-series plots for:
    - Motor Current (A)
    - Bearing Temperature (°C)
    - Hydraulic Pressure (bar)
    - Vibration Velocity (mm/s)
    Each channel displays the DCS alarm threshold alongside detected candidate anomalies.
  - **24-Hour Trend & Linear Extrapolation**: Visualizes the equipment health trend over the previous 24 hours with an optional linear regression projection and 95% confidence bounds.

---

### 2. 🎯 Anomaly Detection Results
- **Core Purpose**: Comprehensive offline model evaluation on a strictly chronologically held-out test split, comparing model alerts against plant DCS threshold exceedances.
- **Key Capabilities**:
  - **Model Benchmarking**: Compare multiple detection approaches:
    - *Isolation Forest*: Unsupervised baseline anomaly detector.
    - *XGBoost Supervised Weak-Label*: Gradient-boosted decision trees predicting threshold events within a 30-minute lookahead horizon.
    - *Autoencoder*: Reconstruction error thresholding trained on healthy operating regimes.
  - **Point-wise & Event-level Metrics**:
    - Point-wise: Precision, Recall, F1-score, PR-AUC.
    - Event-level: Event Recall (percentage of continuous alarm episodes caught) and False Alarm Rate per week.
  - **4 Detailed Tabs**:
    1. *Scores & Thresholds*: Overlay of raw continuous anomaly scores vs. sensor traces.
    2. *Precision–Recall & Confusion Matrix*: PR curves highlighting performance against severe class imbalance.
    3. *Events & Lead Time*: Quantifies early-warning lead times (minutes of advance notice before a DCS alarm triggers).
    4. *Model Comparison & Imbalance (SMOTE)*: Comparative analysis of precision-recall tradeoffs under imbalanced class conditions.

---

### 3. ⏪ Historical Replay
- **Core Purpose**: Deep-dive diagnostic playback across extended multi-day or custom time windows.
- **Key Capabilities**:
  - **Dual Time-Range Slider**: Select and zoom into arbitrary historical operating windows.
  - **Multi-tier Synchronized Subplots**: Displays the composite Equipment Health Score stacked synchronously above the 4 individual sensor traces.
  - **Model Alert Overlays**: Compare markers for DCS threshold breaches against Isolation Forest, XGBoost, or Autoencoder triggers.
  - **Operational State Breakdown**: Summary of total minutes spent in Healthy, Attention, Warning, or Critical states during the selected window.

---

### 4. 🔍 Model Explanation
- **Core Purpose**: Explainable AI (XAI) and root cause attribution for machine learning predictions.
- **Key Capabilities**:
  - **Global Feature Importance**: Top 15 ranked features contributing to risk scores based on mean absolute SHAP values (`mean |SHAP|`).
  - **SHAP Summary (Beeswarm) Plot**: Shows how high vs. low feature values drive the risk score up (red) or down (blue).
  - **Local Instance Waterfall**: Inspect any individual high-risk timestamp to see exact feature contributions explaining why the model flagged that minute.
  - **Physics/Statistical Residual Twin**: Evaluates deviations between observed temperature/vibration vs. expected operating state.

---

### 5. 📋 Data Quality & Method
- **Core Purpose**: Data governance, telemetry validation, methodology explanation, and transparency.
- **Key Capabilities**:
  - **Data Hygiene Audit**: Summary of total records, sampling rate consistency (1-minute intervals), timestamp gaps, duplicates, and missing values.
  - **Zero-Interpolation & Cleaning Policy**: Transparently documents data preprocessing rules without synthetic distortion of physical signals.
  - **Train/Test Chronological Split**: Details the time-based cutoff to ensure zero lookahead bias or data leakage.
  - **Weak Labeling & Operational Limitations**: Clarifies that threshold exceedances serve as operational weak labels rather than confirmed mechanical teardown logs.

---

## 🛠️ Project Structure

```
├── backend/
│   ├── data/             # Raw and processed Parquet / Excel data
│   ├── docs/             # Technical and architecture documentation
│   ├── models/           # Trained models (Isolation Forest, XGBoost, Autoencoder)
│   ├── reports/          # Evaluation figures and metrics JSON
│   ├── src/              # Feature engineering, preprocessing, and model pipelines
│   └── run_pipeline.py   # Pipeline execution script
├── frontend/
│   ├── app.py            # Streamlit dashboard entry point
│   ├── data_access.py    # Cached data loader utilities
│   ├── theme.py          # Design system, cards, and Plotly styling
│   └── views/            # 5 dashboard analytical views
│       ├── health_monitor.py
│       ├── anomaly_results.py
│       ├── historical_replay.py
│       ├── explanation.py
│       └── data_quality.py
├── requirements.txt      # Project Python dependencies
└── README.md             # Project documentation
```

---

## ⚠️ Disclaimer
This digital twin dashboard operates on recorded sensor history from the Sulzer CFG-42 centrifuge. All health scores, anomaly alerts, and forecasts are analytical aids and prototypes, not certified automated safety interlocks.
