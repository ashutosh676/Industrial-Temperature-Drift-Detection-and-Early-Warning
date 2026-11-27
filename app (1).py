import time
import random
from datetime import datetime
from collections import deque

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Industrial Temperature Drift Detection",
    page_icon="🌡️",
    layout="wide",
)

# -------------------- SESSION STATE --------------------
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=500)
if "alerts" not in st.session_state:
    st.session_state.alerts = deque(maxlen=100)
if "temperature" not in st.session_state:
    st.session_state.temperature = 55.0
if "running" not in st.session_state:
    st.session_state.running = True
if "last_alert_key" not in st.session_state:
    st.session_state.last_alert_key = None

# -------------------- SIMULATED SENSOR --------------------
def get_simulated_temperature():
    """Generate a slowly varying temperature with noise and occasional drift."""
    previous = st.session_state.temperature
    elapsed = len(st.session_state.history)

    # Gradual operating variation plus small random noise.
    drift = 0.035
    if elapsed > 25 and elapsed % 90 > 55:
        drift = 0.16

    new_value = previous + drift + random.uniform(-0.65, 0.65)

    # Keep simulated values in a plausible demonstration range.
    new_value = max(25.0, min(new_value, 115.0))
    st.session_state.temperature = new_value
    return round(new_value, 2)


def classify_temperature(value, warning_limit, critical_limit):
    if value >= critical_limit:
        return "CRITICAL", "🔴"
    if value >= warning_limit:
        return "WARNING", "🟠"
    return "NORMAL", "🟢"


def detect_drift(data, baseline, drift_limit, window_size):
    """Detect sustained deviation using the mean of the latest window."""
    if len(data) < window_size:
        return False, 0.0, "Collecting readings"

    recent = [row["Temperature (°C)"] for row in list(data)[-window_size:]]
    average = sum(recent) / len(recent)
    deviation = average - baseline
    detected = abs(deviation) >= drift_limit
    message = "Sustained temperature drift detected" if detected else "No sustained drift"
    return detected, round(deviation, 2), message


# -------------------- SIDEBAR SETTINGS --------------------
st.sidebar.title("⚙️ Monitoring Settings")
st.sidebar.caption("Simulation mode · MAX6675-ready project")

warning_limit = st.sidebar.slider(
    "Warning temperature (°C)", min_value=30, max_value=110, value=75, step=1
)
critical_limit = st.sidebar.slider(
    "Critical temperature (°C)", min_value=40, max_value=120, value=90, step=1
)
if critical_limit <= warning_limit:
    st.sidebar.error("Critical limit must be higher than warning limit.")
    st.stop()

baseline = st.sidebar.number_input(
    "Reference baseline (°C)", min_value=0.0, max_value=200.0, value=55.0, step=0.5
)
drift_limit = st.sidebar.slider(
    "Drift deviation limit (°C)", min_value=1.0, max_value=30.0, value=5.0, step=0.5
)
window_size = st.sidebar.slider(
    "Drift detection window (readings)", min_value=3, max_value=30, value=8
)
refresh_seconds = st.sidebar.slider(
    "Refresh interval (seconds)", min_value=1, max_value=10, value=2
)

st.sidebar.divider()
st.sidebar.subheader("Demo controls")
st.session_state.running = st.sidebar.toggle("Live monitoring", value=st.session_state.running)
if st.sidebar.button("Reset readings", use_container_width=True):
    st.session_state.history.clear()
    st.session_state.alerts.clear()
    st.session_state.temperature = 55.0
    st.session_state.last_alert_key = None
    st.rerun()

# -------------------- HEADER --------------------
st.title("🌡️ Industrial Temperature Drift Detection & Early Warning")
st.caption("Live industrial temperature monitoring dashboard | Python simulation")

now = datetime.now()
st.write(f"**Machine:** Industrial Motor Unit A  ·  **Sensor:** MAX6675 / K-type thermocouple (simulated)  ·  **Updated:** {now:%d %b %Y, %H:%M:%S}")

# Add one sample per app run while monitoring is enabled.
if st.session_state.running:
    current_temp = get_simulated_temperature()
    status, emoji = classify_temperature(current_temp, warning_limit, critical_limit)
    timestamp = datetime.now()
    st.session_state.history.append({
        "Timestamp": timestamp,
        "Temperature (°C)": current_temp,
        "Status": status,
    })

    drift_detected, drift_value, drift_message = detect_drift(
        st.session_state.history, baseline, drift_limit, window_size
    )

    alert_key = (status, drift_detected)
    if status != "NORMAL" or drift_detected:
        if alert_key != st.session_state.last_alert_key:
            messages = []
            if status != "NORMAL":
                messages.append(f"{status}: temperature is {current_temp:.2f} °C")
            if drift_detected:
                messages.append(f"DRIFT: deviation from baseline is {drift_value:+.2f} °C")
            st.session_state.alerts.append({
                "Timestamp": timestamp,
                "Alert": " | ".join(messages),
                "Temperature (°C)": current_temp,
            })
    st.session_state.last_alert_key = alert_key
else:
    if st.session_state.history:
        current_temp = st.session_state.history[-1]["Temperature (°C)"]
        status, emoji = classify_temperature(current_temp, warning_limit, critical_limit)
        drift_detected, drift_value, drift_message = detect_drift(
            st.session_state.history, baseline, drift_limit, window_size
        )
    else:
        current_temp, status, emoji = 55.0, "WAITING", "⚪"
        drift_detected, drift_value, drift_message = False, 0.0, "Waiting for readings"

# -------------------- KPI CARDS --------------------
history_df = pd.DataFrame(list(st.session_state.history))
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current temperature", f"{current_temp:.2f} °C")
col2.metric("Operating status", f"{emoji} {status}")
col3.metric("Drift deviation", f"{drift_value:+.2f} °C", help="Recent-window average minus reference baseline")
col4.metric("Total readings", len(history_df))

# -------------------- ALERT BANNERS --------------------
if status == "CRITICAL":
    st.error("🚨 CRITICAL ALERT — Temperature has reached or exceeded the critical limit. Inspect the equipment immediately.")
elif status == "WARNING":
    st.warning("⚠️ WARNING — Temperature has crossed the configured warning limit.")
elif status == "NORMAL":
    st.success("✅ NORMAL — Temperature is within the configured operating limits.")
if drift_detected:
    st.warning(f"📈 DRIFT DETECTED — {drift_message}. Recent-window deviation: {drift_value:+.2f} °C.")

# -------------------- CHART --------------------
st.subheader("📊 Live Temperature Trend")
if not history_df.empty:
    fig = px.line(
        history_df,
        x="Timestamp",
        y="Temperature (°C)",
        title="Temperature vs Time",
        markers=True,
    )
    fig.add_hline(y=warning_limit, line_dash="dash", annotation_text="Warning limit")
    fig.add_hline(y=critical_limit, line_dash="dash", annotation_text="Critical limit")
    fig.add_hline(y=baseline, line_dash="dot", annotation_text="Reference baseline")
    fig.update_layout(
        height=420,
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No readings yet. Turn on Live monitoring in the sidebar.")

# -------------------- DRIFT DETAILS --------------------
left, right = st.columns(2)
with left:
    st.subheader("🔎 Drift Detection")
    st.write(f"**Reference baseline:** {baseline:.2f} °C")
    st.write(f"**Drift limit:** ±{drift_limit:.2f} °C")
    st.write(f"**Detection window:** {window_size} readings")
    st.write(f"**Result:** {'Drift detected' if drift_detected else 'No sustained drift detected'}")
    st.caption("Method: moving-window average compared with a configured baseline. This is a demonstration algorithm, not a certified industrial safety system.")

with right:
    st.subheader("🚦 Threshold Configuration")
    st.write(f"🟢 Normal: below {warning_limit} °C")
    st.write(f"🟠 Warning: {warning_limit} °C to below {critical_limit} °C")
    st.write(f"🔴 Critical: {critical_limit} °C and above")

# -------------------- ALERT LOG --------------------
st.subheader("🔔 Alert History")
alerts_df = pd.DataFrame(list(st.session_state.alerts))
if alerts_df.empty:
    st.info("No alerts generated in this session.")
else:
    st.dataframe(alerts_df.iloc[::-1], use_container_width=True, hide_index=True)

# -------------------- DATA TABLE + CSV --------------------
st.subheader("📁 Temperature Data")
if not history_df.empty:
    st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download temperature history (CSV)",
        data=csv,
        file_name="temperature_history.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Academic prototype only. Sensor values are simulated. Do not use this dashboard as the sole protection or control system for real industrial equipment.")

# Streamlit rerun loop for live updates.
if st.session_state.running:
    time.sleep(refresh_seconds)
    st.rerun()
