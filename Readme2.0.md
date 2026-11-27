# Industrial Temperature Drift Detection and Early Warning

An explainable, trend-based temperature monitoring and early-warning prototype built with Python and Streamlit. The system analyzes timestamped temperature readings over time, establishes a normal operating baseline, computes drift indicators, and raises structured early warnings before fixed alarm limits are crossed.

**Author:** Ashutosh Saxena (Roll No: 2300320210019)  
**Department:** Electrical and Electronics Engineering, ABES Engineering College  
**Project Guide:** Raveendra Reddy  
**Academic Year:** 2026 – 2027  

---

## Key Features

- **6-Layer Monitoring Pipeline:** Data Acquisition → Data Validation → Preprocessing → Baseline & Feature Calculation → Decision & Warning Engine → Dashboard & Event Logging.
- **Data Quality & Fault Detection:** Automatically flags missing values, non-numeric inputs, duplicate timestamps, sampling gaps (stale data), and readings outside the configured sensor range as `SENSOR/DATA FAULT`.
- **Configurable Baseline Estimation:** Supports a verified initial baseline (fixed median) as well as rolling median/mean baselines with an option to freeze updates during active alerts so slow drift is not absorbed.
- **Drift Indicators:** Computes deviation from baseline $D(t) = T(t) - B(t)$ (unidirectional or bidirectional $\vert{}D(t)\vert{}$), rate of change $R(t)$ in °C/min, and consecutive persistence counters.
- **Persistence & Hysteresis Warning Engine:** Classifies states into `NORMAL`, `CAUTION`, `WARNING`, `CRITICAL`, and `SENSOR/DATA FAULT` with human-readable trigger explanations and recovery hysteresis to prevent alert chattering.
- **Built-in Test Scenarios & Metrics:** Includes labelled scenarios (stable operation, slow upward/downward drift, single spike, noisy fluctuations, missing data/sensor faults, and recovery) with Precision, Recall, and Detection Delay evaluation.
- **CSV Upload & Export:** Accepts custom CSV files and exports timestamped event logs and processed datasets.

---

## Repository Structure

```text
.
├── app.py              # Main Streamlit dashboard & drift detection engine
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation