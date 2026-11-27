"""
Industrial Temperature Drift Detection and Early Warning System
Author: Ashutosh Saxena (Roll No: 2300320210019)
Department of Electrical and Electronics Engineering, ABES Engineering College
Academic Year: 2026 - 2027

Run with:
    streamlit run app.py

Note: Uses only core Streamlit dependencies (streamlit, pandas, numpy, altair)
so it deploys cleanly on Streamlit Community Cloud without extra packages.
This is an academic decision-support prototype and does not replace certified
industrial over-temperature protection or safety instrumented systems.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. CONFIGURATION MODEL (Section 9.3: Configuration Management)
# =============================================================================
@dataclass
class DriftConfig:
    sensor_min_c: float = 0.0
    sensor_max_c: float = 150.0
    max_gap_seconds: float = 30.0
    baseline_method: str = "fixed_initial"  # "fixed_initial", "rolling_median", or "rolling_mean"
    baseline_window: int = 20
    freeze_baseline_on_alert: bool = True
    smoothing_window: int = 1  # 1 = no smoothing (raw)
    bidirectional_drift: bool = False  # True checks |D(t)|, False checks upward D(t)
    caution_threshold_c: float = 3.0
    caution_persistence: int = 5
    warning_threshold_c: float = 6.0
    warning_persistence: int = 3
    critical_limit_c: float = 75.0
    recovery_threshold_c: float = 1.5
    recovery_persistence: int = 4
    roc_window_samples: int = 3


# =============================================================================
# 2. BUILT-IN TEST SCENARIO GENERATOR (Section 10.2 & Appendix A)
# =============================================================================
def generate_scenario_data(
    scenario: str,
    n_samples: int = 120,
    sample_interval_sec: int = 5,
    base_temp: float = 60.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates illustrative timestamped datasets matching Appendix A format."""
    rng = np.random.default_rng(seed)
    start_time = datetime(2026, 1, 1, 10, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * sample_interval_sec) for i in range(n_samples)]
    temps = base_temp + rng.normal(0.0, 0.25, size=n_samples)
    modes = ["steady"] * n_samples
    is_drift_label = [0] * n_samples

    if scenario == "Slow Upward Drift + Recovery":
        for i in range(35, 85):
            temps[i] += (i - 35) * 0.22
            is_drift_label[i] = 1
        for i in range(85, n_samples):
            decay = max(0.0, 11.0 - (i - 85) * 0.55)
            temps[i] += decay
            if decay > 3.0:
                is_drift_label[i] = 1

    elif scenario == "Slow Upward Drift (To Critical)":
        for i in range(30, n_samples):
            temps[i] += (i - 30) * 0.22
            is_drift_label[i] = 1

    elif scenario == "Slow Downward Drift":
        for i in range(35, n_samples):
            temps[i] -= (i - 35) * 0.16
            is_drift_label[i] = 1

    elif scenario == "Single Spike (Noise Resilience)":
        temps[50] = base_temp + 10.5
        temps[85] = base_temp + 8.0

    elif scenario == "Missing Data & Sensor Fault":
        temps[40] = np.nan
        temps[41] = np.nan
        temps[70] = 210.0  # Out of sensor range
        temps[90] = -25.0  # Out of sensor range
        for j in range(55, n_samples):
            timestamps[j] = timestamps[j] + timedelta(seconds=45)

    elif scenario == "Repeated Noisy Fluctuations":
        temps = base_temp + rng.normal(0.0, 1.4, size=n_samples)
        for i in range(45, 75):
            temps[i] += 2.8

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": "TEMP-01",
            "temperature_c": np.round(temps, 2),
            "operating_mode": modes,
            "ground_truth_drift": is_drift_label,
        }
    )


# =============================================================================
# 3. VALIDATION, PREPROCESSING & DRIFT DETECTION ENGINE (Sections 5, 7, 8)
# =============================================================================
def run_drift_pipeline(df_raw: pd.DataFrame, cfg: DriftConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the 6-layer pipeline:
    Validation -> Preprocessing -> Baseline -> Features -> Warning Engine -> Event Log.
    """
    if df_raw is None or df_raw.empty:
        raise ValueError("Input dataset is empty.")

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"timestamp", "temperature_c"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing_cols))}")

    if "sensor_id" not in df.columns:
        df["sensor_id"] = "TEMP-01"
    if "operating_mode" not in df.columns:
        df["operating_mode"] = "steady"

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["temperature_raw"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)

    valid_history: List[float] = []
    valid_times: List[pd.Timestamp] = []

    current_status = "NORMAL"
    caution_count = 0
    warning_count = 0
    recovery_count = 0
    last_baseline: float = np.nan
    fixed_baseline_locked: Optional[float] = None
    last_valid_ts: Optional[pd.Timestamp] = None

    records: List[Dict] = []
    events: List[Dict] = []

    for idx, row in df.iterrows():
        ts = row["timestamp"]
        raw_t = row["temperature_raw"]
        sensor_id = row["sensor_id"]
        mode = row["operating_mode"]

        # --- Layer 2: Data Validation ---
        fault_reason = None
        if pd.isna(ts):
            fault_reason = "Invalid or missing timestamp"
        elif idx > 0 and pd.notna(df.loc[idx - 1, "timestamp"]) and ts == df.loc[idx - 1, "timestamp"]:
            fault_reason = "Duplicate timestamp detected"
        elif pd.isna(raw_t):
            fault_reason = "Missing or non-numeric temperature value"
        elif raw_t < cfg.sensor_min_c or raw_t > cfg.sensor_max_c:
            fault_reason = (
                f"Implausible reading ({raw_t:.2f} °C) outside sensor range "
                f"[{cfg.sensor_min_c}, {cfg.sensor_max_c}] °C"
            )
        elif last_valid_ts is not None and (ts - last_valid_ts).total_seconds() > cfg.max_gap_seconds:
            gap_s = (ts - last_valid_ts).total_seconds()
            fault_reason = f"Stale data / sampling gap detected ({gap_s:.1f} s > {cfg.max_gap_seconds:.1f} s)"

        if fault_reason is not None:
            new_status = "SENSOR/DATA FAULT"
            if new_status != current_status:
                events.append(
                    {
                        "timestamp": ts,
                        "sensor_id": sensor_id,
                        "from_status": current_status,
                        "to_status": new_status,
                        "temperature_c": raw_t,
                        "baseline_c": last_baseline,
                        "deviation_c": np.nan,
                        "reason": fault_reason,
                    }
                )
            if pd.notna(ts) and pd.notna(raw_t) and cfg.sensor_min_c <= raw_t <= cfg.sensor_max_c:
                last_valid_ts = ts
            current_status = new_status
            records.append(
                {
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "operating_mode": mode,
                    "temperature_raw": raw_t,
                    "temperature_proc": np.nan,
                    "baseline_c": last_baseline,
                    "deviation_c": np.nan,
                    "rate_c_per_min": np.nan,
                    "status": current_status,
                    "reason": fault_reason,
                }
            )
            continue

        last_valid_ts = ts
        valid_history.append(float(raw_t))
        valid_times.append(ts)

        # --- Layer 3: Preprocessing (Optional Smoothing) ---
        if cfg.smoothing_window > 1 and len(valid_history) >= cfg.smoothing_window:
            proc_t = float(np.mean(valid_history[-cfg.smoothing_window :]))
        else:
            proc_t = float(raw_t)

        # --- Layer 4: Baseline & Feature Calculation ---
        if cfg.baseline_method == "fixed_initial":
            if fixed_baseline_locked is None:
                baseline = float(np.median(valid_history))
                if len(valid_history) >= cfg.baseline_window:
                    fixed_baseline_locked = baseline
            else:
                baseline = fixed_baseline_locked
            last_baseline = baseline
        else:
            alert_active = current_status in ("CAUTION", "WARNING", "CRITICAL")
            if pd.isna(last_baseline) or not (cfg.freeze_baseline_on_alert and alert_active):
                window_slice = valid_history[-cfg.baseline_window :]
                if cfg.baseline_method == "rolling_mean":
                    baseline = float(np.mean(window_slice))
                else:
                    baseline = float(np.median(window_slice))
                last_baseline = baseline
            else:
                baseline = last_baseline

        deviation = proc_t - baseline
        eval_dev = abs(deviation) if cfg.bidirectional_drift else deviation

        # Rate of change R(t) = [T(t) - T(t - dt)] / dt (converted to °C/min)
        if len(valid_history) > cfg.roc_window_samples:
            t_prev = valid_history[-(cfg.roc_window_samples + 1)]
            ts_prev = valid_times[-(cfg.roc_window_samples + 1)]
            dt_min = (ts - ts_prev).total_seconds() / 60.0
            rate_c_per_min = (proc_t - t_prev) / dt_min if dt_min > 0 else 0.0
        else:
            rate_c_per_min = 0.0

        # --- Layer 5: Decision & Warning Engine (Persistence + Hysteresis) ---
        if eval_dev >= cfg.warning_threshold_c:
            warning_count += 1
            caution_count += 1
            recovery_count = 0
        elif eval_dev >= cfg.caution_threshold_c:
            caution_count += 1
            warning_count = 0
            recovery_count = 0
        else:
            caution_count = 0
            warning_count = 0
            if eval_dev <= cfg.recovery_threshold_c:
                recovery_count += 1
            else:
                recovery_count = 0

        new_status = current_status
        reason = "Operating within normal baseline band"

        if proc_t >= cfg.critical_limit_c:
            new_status = "CRITICAL"
            reason = f"Temperature ({proc_t:.2f} °C) reached critical limit ({cfg.critical_limit_c:.2f} °C)"
        elif warning_count >= cfg.warning_persistence:
            new_status = "WARNING"
            reason = (
                f"Deviation ({deviation:+.2f} °C) exceeded warning threshold "
                f"({cfg.warning_threshold_c:.1f} °C) for {warning_count} consecutive samples"
            )
        elif caution_count >= cfg.caution_persistence:
            if current_status != "WARNING":
                new_status = "CAUTION"
                reason = (
                    f"Deviation ({deviation:+.2f} °C) exceeded caution threshold "
                    f"({cfg.caution_threshold_c:.1f} °C) for {caution_count} consecutive samples"
                )
            else:
                reason = "Warning remains active (hysteresis awaiting recovery band)"
        else:
            if current_status == "SENSOR/DATA FAULT":
                new_status = "NORMAL"
                reason = "Valid sensor stream restored within normal band"
            elif current_status in ("CAUTION", "WARNING", "CRITICAL"):
                if recovery_count >= cfg.recovery_persistence:
                    new_status = "NORMAL"
                    reason = (
                        f"Recovered within {cfg.recovery_threshold_c:.1f} °C of baseline "
                        f"for {recovery_count} consecutive samples"
                    )
                else:
                    reason = (
                        f"{current_status} held by hysteresis "
                        f"({recovery_count}/{cfg.recovery_persistence} recovery samples)"
                    )
            else:
                new_status = "NORMAL"

        if new_status != current_status:
            events.append(
                {
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "from_status": current_status,
                    "to_status": new_status,
                    "temperature_c": round(proc_t, 2),
                    "baseline_c": round(baseline, 2),
                    "deviation_c": round(deviation, 2),
                    "reason": reason,
                }
            )
            current_status = new_status

        records.append(
            {
                "timestamp": ts,
                "sensor_id": sensor_id,
                "operating_mode": mode,
                "temperature_raw": round(raw_t, 2),
                "temperature_proc": round(proc_t, 2),
                "baseline_c": round(baseline, 2),
                "deviation_c": round(deviation, 2),
                "rate_c_per_min": round(rate_c_per_min, 3),
                "status": current_status,
                "reason": reason,
            }
        )

    results_df = pd.DataFrame(records)
    for col in ("temperature_raw", "temperature_proc", "baseline_c", "deviation_c", "rate_c_per_min"):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    if "ground_truth_drift" in df.columns:
        results_df["ground_truth_drift"] = df["ground_truth_drift"].values
    events_df = pd.DataFrame(events)
    return results_df, events_df


# =============================================================================
# 4. EVALUATION METRICS (Section 10.3)
# =============================================================================
def compute_evaluation_metrics(results_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if "ground_truth_drift" not in results_df.columns:
        return None
    y_true = results_df["ground_truth_drift"].astype(int).values
    y_pred = results_df["status"].isin(["CAUTION", "WARNING", "CRITICAL"]).astype(int).values

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    drift_indices = np.where(y_true == 1)[0]
    delay_samples = np.nan
    if len(drift_indices) > 0:
        first_drift = drift_indices[0]
        alert_after = np.where((y_pred == 1) & (np.arange(len(y_pred)) >= first_drift))[0]
        if len(alert_after) > 0:
            delay_samples = int(alert_after[0] - first_drift)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "Detection Delay (samples)": delay_samples,
    }


# =============================================================================
# 5. INTERACTIVE ALTAIR CHART BUILDER (Built into Streamlit - No Plotly Needed)
# =============================================================================
def build_trend_chart(results_df: pd.DataFrame, cfg: DriftConfig) -> alt.Chart:
    chart_df = results_df.dropna(subset=["timestamp"]).copy()
    chart_df["caution_upper"] = chart_df["baseline_c"] + cfg.caution_threshold_c
    chart_df["warning_upper"] = chart_df["baseline_c"] + cfg.warning_threshold_c

    series_cols = {
        "temperature_raw": "Measured Temp T(t)",
        "baseline_c": "Baseline B(t)",
        "caution_upper": "Caution Upper Band",
        "warning_upper": "Warning Upper Band",
    }
    if cfg.bidirectional_drift:
        chart_df["caution_lower"] = chart_df["baseline_c"] - cfg.caution_threshold_c
        chart_df["warning_lower"] = chart_df["baseline_c"] - cfg.warning_threshold_c
        series_cols["caution_lower"] = "Caution Lower Band"
        series_cols["warning_lower"] = "Warning Lower Band"

    long_df = chart_df[["timestamp"] + list(series_cols.keys())].melt(
        id_vars=["timestamp"], var_name="series_key", value_name="temperature_c"
    )
    long_df["Series"] = long_df["series_key"].map(series_cols)

    color_domain = [
        "Measured Temp T(t)",
        "Baseline B(t)",
        "Caution Upper Band",
        "Warning Upper Band",
        "Caution Lower Band",
        "Warning Lower Band",
    ]
    color_range = ["#2563eb", "#16a34a", "#d97706", "#ea580c", "#d97706", "#ea580c"]

    lines = (
        alt.Chart(long_df.dropna(subset=["temperature_c"]))
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title="Timestamp"),
            y=alt.Y("temperature_c:Q", title="Temperature (°C)", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=color_domain, range=color_range)),
            strokeDash=alt.condition(
                alt.datum.Series == "Measured Temp T(t)",
                alt.value([1, 0]),
                alt.value([5, 4]),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M:%S"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("temperature_c:Q", title="Temp (°C)", format=".2f"),
            ],
        )
    )

    # Critical limit horizontal line
    crit_df = pd.DataFrame({"critical": [cfg.critical_limit_c], "label": [f"Critical ({cfg.critical_limit_c} °C)"]})
    crit_rule = (
        alt.Chart(crit_df)
        .mark_rule(color="#dc2626", strokeDash=[6, 4], strokeWidth=2)
        .encode(y="critical:Q")
    )

    # Highlight non-normal status points
    alert_df = chart_df[chart_df["status"] != "NORMAL"].dropna(subset=["temperature_raw"])
    status_domain = ["CAUTION", "WARNING", "CRITICAL", "SENSOR/DATA FAULT"]
    status_range = ["#d97706", "#ea580c", "#dc2626", "#7c3aed"]

    layers = [lines, crit_rule]
    if not alert_df.empty:
        alert_pts = (
            alt.Chart(alert_df)
            .mark_point(size=70, filled=True)
            .encode(
                x="timestamp:T",
                y="temperature_raw:Q",
                color=alt.Color("status:N", title="Alert State", scale=alt.Scale(domain=status_domain, range=status_range)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M:%S"),
                    alt.Tooltip("temperature_raw:Q", title="Measured (°C)", format=".2f"),
                    alt.Tooltip("status:N", title="Status"),
                    alt.Tooltip("reason:N", title="Reason"),
                ],
            )
        )
        layers.append(alert_pts)

    return alt.layer(*layers).resolve_scale(color="independent").properties(height=420).interactive()


# =============================================================================
# 6. STREAMLIT DASHBOARD UI (Section 5.4: User Interface Concept)
# =============================================================================
def main() -> None:
    st.set_page_config(
        page_title="Industrial Temperature Drift Detection",
        page_icon="🌡️",
        layout="wide",
    )

    st.title("🌡️ Industrial Temperature Drift Detection & Early Warning")
    st.caption(
        "Decision-support prototype for trend-based thermal monitoring | "
        "Ashutosh Saxena (Roll No: 2300320210019), EEE, ABES Engineering College"
    )

    # --- Sidebar Configuration ---
    st.sidebar.header("1. Data Source")
    data_mode = st.sidebar.radio("Input Mode", ["Built-in Test Scenarios", "Upload CSV File"])

    scenario = "Custom CSV"
    if data_mode == "Built-in Test Scenarios":
        scenario = st.sidebar.selectbox(
            "Select Scenario (Section 10.2)",
            [
                "Slow Upward Drift + Recovery",
                "Slow Upward Drift (To Critical)",
                "Slow Downward Drift",
                "Stable Operation",
                "Single Spike (Noise Resilience)",
                "Missing Data & Sensor Fault",
                "Repeated Noisy Fluctuations",
            ],
        )
        raw_df = generate_scenario_data(scenario)
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV (timestamp, temperature_c)", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
        else:
            st.info(
                "Upload a CSV file with columns `timestamp`, `sensor_id`, `temperature_c`, `operating_mode` "
                "(Appendix A format), or switch to **Built-in Test Scenarios**."
            )
            st.stop()

    st.sidebar.header("2. Detection Configuration")
    baseline_label_map = {
        "Verified Initial Baseline (Fixed Median)": "fixed_initial",
        "Rolling Median": "rolling_median",
        "Rolling Mean": "rolling_mean",
    }
    selected_baseline_label = st.sidebar.selectbox(
        "Baseline Method", list(baseline_label_map.keys()), index=0
    )

    default_bidirectional = scenario == "Slow Downward Drift"

    cfg = DriftConfig(
        sensor_min_c=st.sidebar.number_input("Sensor Min (°C)", value=0.0, step=5.0),
        sensor_max_c=st.sidebar.number_input("Sensor Max (°C)", value=150.0, step=10.0),
        max_gap_seconds=st.sidebar.number_input("Max Sampling Gap (s)", value=30.0, step=5.0),
        baseline_method=baseline_label_map[selected_baseline_label],
        baseline_window=st.sidebar.slider("Baseline Window (samples)", 5, 40, 20),
        freeze_baseline_on_alert=st.sidebar.checkbox("Freeze Rolling Baseline During Alerts", value=True),
        smoothing_window=st.sidebar.slider("Smoothing Window (1 = Raw)", 1, 7, 1),
        bidirectional_drift=st.sidebar.checkbox(
            "Detect Downward Drift Too (|D(t)|)",
            value=default_bidirectional,
            key=f"bidir_{scenario}",
        ),
        caution_threshold_c=st.sidebar.number_input("Caution Threshold (°C)", value=3.0, step=0.5),
        caution_persistence=st.sidebar.slider("Caution Persistence (samples)", 1, 15, 5),
        warning_threshold_c=st.sidebar.number_input("Warning Threshold (°C)", value=6.0, step=0.5),
        warning_persistence=st.sidebar.slider("Warning Persistence (samples)", 1, 15, 3),
        critical_limit_c=st.sidebar.number_input("Critical Hard Limit (°C)", value=75.0, step=1.0),
        recovery_threshold_c=st.sidebar.number_input("Recovery Hysteresis Band (°C)", value=1.5, step=0.5),
        recovery_persistence=st.sidebar.slider("Recovery Persistence (samples)", 1, 15, 4),
    )

    try:
        results_df, events_df = run_drift_pipeline(raw_df, cfg)
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        st.stop()

    # --- Latest Status Banner & KPI Cards ---
    latest = results_df.iloc[-1]
    status_colors = {
        "NORMAL": "#16a34a",
        "CAUTION": "#d97706",
        "WARNING": "#ea580c",
        "CRITICAL": "#dc2626",
        "SENSOR/DATA FAULT": "#7c3aed",
    }
    badge_color = status_colors.get(latest["status"], "#475569")

    st.markdown(
        f"""
        <div style="padding: 14px 18px; border-radius: 8px; border-left: 6px solid {badge_color};
                    background-color: rgba(148, 163, 184, 0.12); margin-bottom: 16px;">
            <strong style="color: {badge_color}; font-size: 1.15rem;">STATUS: {latest["status"]}</strong><br/>
            <span>Trigger / Reason: {latest["reason"]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Latest Temperature",
        f"{latest['temperature_raw']:.2f} °C" if pd.notna(latest["temperature_raw"]) else "FAULT",
    )
    c2.metric(
        "Baseline B(t)",
        f"{latest['baseline_c']:.2f} °C" if pd.notna(latest["baseline_c"]) else "N/A",
    )
    c3.metric(
        "Deviation D(t)",
        f"{latest['deviation_c']:+.2f} °C" if pd.notna(latest["deviation_c"]) else "N/A",
    )
    c4.metric(
        "Rate of Change R(t)",
        f"{latest['rate_c_per_min']:+.2f} °C/min" if pd.notna(latest["rate_c_per_min"]) else "N/A",
    )

    # --- Interactive Trend Chart ---
    st.subheader("Temperature Trend & Drift Thresholds")
    st.altair_chart(build_trend_chart(results_df, cfg), use_container_width=True)

    # --- Event Log & Evaluation Tabs ---
    tab1, tab2, tab3 = st.tabs(["📋 Event Log", "📊 Evaluation Metrics", "🗂️ Processed Data"])

    with tab1:
        if events_df.empty:
            st.success("No state transitions or warnings were triggered during this run.")
        else:
            st.dataframe(events_df, use_container_width=True)
            st.download_button(
                "Download Event Log (CSV)",
                data=events_df.to_csv(index=False).encode("utf-8"),
                file_name="temperature_drift_events.csv",
                mime="text/csv",
            )

    with tab2:
        metrics = compute_evaluation_metrics(results_df)
        if metrics is None:
            st.info("Evaluation metrics require a `ground_truth_drift` column (included in built-in scenarios).")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{metrics['Precision']:.2%}")
            m2.metric("Recall", f"{metrics['Recall']:.2%}")
            delay_val = metrics["Detection Delay (samples)"]
            m3.metric("Detection Delay", f"{delay_val} samples" if pd.notna(delay_val) else "N/A")
            m4.metric("False Positives (FP)", metrics["FP"])
            st.caption(
                f"Confusion Matrix — TP: {metrics['TP']} | FP: {metrics['FP']} | "
                f"TN: {metrics['TN']} | FN: {metrics['FN']}"
            )

    with tab3:
        st.dataframe(results_df, use_container_width=True)
        st.download_button(
            "Download Processed Data (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="processed_temperature_data.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(
        "⚠️ Safety Disclaimer: This software is an academic decision-support prototype. "
        "It does not replace certified over-temperature trips, interlocks, or safety instrumented systems."
    )


if __name__ == "__main__":
    main()
