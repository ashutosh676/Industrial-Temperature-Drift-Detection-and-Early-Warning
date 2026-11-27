# Industrial Temperature Drift Detection and Early Warning

A Python-only dashboard prototype using Streamlit. It generates simulated temperature readings, plots trends, detects sustained drift using a moving-window average, raises threshold alerts, stores session history, and exports readings to CSV.

## 1. Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10 or newer
- Internet connection for the initial package installation

## 2. Setup on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Setup on macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The browser normally opens automatically at `http://localhost:8501`.

## 4. How to use

1. Keep **Live monitoring** switched on in the sidebar.
2. Set the warning and critical temperature limits.
3. Set the reference baseline and drift deviation limit.
4. Watch the temperature KPI, trend chart, status banners, drift result, and alert history.
5. Download the session's readings using the CSV button.
6. Use **Reset readings** to clear the demo session.

## 5. Drift detection logic

The app calculates the average of the latest N readings:

`window_average = sum(latest_N_readings) / N`

It then calculates:

`deviation = window_average - reference_baseline`

If `abs(deviation) >= drift_limit`, sustained drift is flagged. This is a simple academic demonstration algorithm. A production system should use a calibrated baseline, sensor validation, filtering, persistence, and a site-approved alarm strategy.

## 6. MAX6675 + K-type thermocouple notes

The dashboard currently runs in **simulation mode** and does not require a MAX6675 board.

For later physical testing, a MAX6675 module is commonly connected to a Raspberry Pi using SPI:
- MAX6675 VCC -> 3.3V (confirm your module's specifications)
- MAX6675 GND -> GND
- MAX6675 SCK -> SPI SCLK
- MAX6675 CS -> a chosen GPIO chip-select pin
- MAX6675 SO -> SPI MISO
- K-type thermocouple -> module's thermocouple terminals

Enable SPI on the Raspberry Pi and use a suitable Python MAX6675 library or a reviewed SPI driver. Check the exact module's voltage requirements and Raspberry Pi pinout before wiring. Do not connect a module's 5V logic output directly to Raspberry Pi GPIO.

To add hardware input later, replace `get_simulated_temperature()` in `app.py` with a function that reads the MAX6675 driver and returns a temperature in degrees Celsius. Keep exception handling and sensor-disconnection detection.

## 7. Project structure

```text
industrial_temperature_dashboard/
├── app.py
├── requirements.txt
└── README.md
```

## Safety notice

This is an educational prototype, not a certified industrial alarm, safety interlock, or machine protection system. Do not rely on it as the sole means of protecting people or equipment.
