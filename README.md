# CarCareAI: Predictive Maintenance Dashboard for Vehicle Health

## Overview

Modern vehicles generate a continuous stream of data from sensors embedded in the powertrain, chassis, safety systems and infotainment units.  This data includes engine temperature, oil pressure, vibration, speed, mileage and hundreds of other signals that describe the current operating state【371604006037961†L188-L215】.  When captured and analysed correctly, these signals support concrete outcomes such as fewer roadside failures, faster diagnostics and improved fuel efficiency【371604006037961†L188-L196】.

**CarCareAI** is an end‑to‑end example project that demonstrates how an automotive manufacturer could use this telemetry to move from reactive maintenance to data‑driven predictive maintenance.  It simulates vehicle sensor data, trains a machine‑learning model to forecast whether maintenance is required and visualises the results in an interactive web dashboard.  The project is intended for developers and data scientists who are interested in automotive analytics and want a portfolio project that shows skills in data simulation, feature engineering, machine learning and web app deployment.

![CarCareAI Dashboard Screenshot](./images/dashboard_screenshot.png)

## Project Features

- **Synthetic telemetry generator** – A script generates a realistic multivariate dataset of vehicle sensor readings.  Features include engine temperature, oil pressure, vibration, brake wear, battery voltage and more.  Maintenance labels are assigned based on domain‑driven rules (e.g. high temperature and low oil pressure indicate potential failure).
- **Machine learning pipeline** – A training script loads the synthetic data, performs train/test splitting, and fits a random forest classifier to predict whether maintenance is needed.  It evaluates the model with standard metrics and saves the trained model to disk for later inference.
- **Streamlit dashboard** – An interactive web application allows you to explore the dataset and obtain predictions.  Users can adjust sensor values with sliders or text boxes and see whether the model predicts maintenance is required.  The dashboard also displays feature importance and a correlation heatmap.
- **Modular design** – The project is organised into separate modules (`data_generator.py`, `model_training.py`, `dashboard.py`) so that you can swap the synthetic data generator for real OBD‑II data sources in the future.  The `requirements.txt` file lists all dependencies.

## Motivation and Background

Automotive manufacturers are embracing artificial intelligence and predictive maintenance to reduce unplanned downtime, improve safety and optimise production.  According to an article on AI‑driven predictive maintenance in automotive manufacturing, predictive systems collect real‑time sensor data, apply machine‑learning algorithms to predict equipment failures and enable just‑in‑time maintenance【731405785906734†L14-L66】.  The architecture typically comprises a sensor layer (IoT devices and ECU sensors), a data aggregation layer, an analytics layer for machine learning and a decision layer that generates alerts【731405785906734†L128-L149】.  CarCareAI is a simplified example of such a system.

Vehicle telemetry is multi‑layered.  At the lowest layer, physical sensors publish raw signals on the in‑vehicle network.  Standard protocols such as OBD‑II, J1979 and CAN‑FD expose diagnostic and operational information, and applications correlate sensor readings with context to derive meaningful events【371604006037961†L210-L218】.  Categories of vehicle data include engine and performance metrics (RPM, temperature, torque), fuel and energy data (fuel level, energy use), location and GPS data, safety and diagnostic data (DTCs, ABS performance) and driving behaviour【371604006037961†L230-L249】.  By combining signals, you can detect patterns such as early cooling system degradation or excessive idling【371604006037961†L255-L260】.  CarCareAI therefore includes multiple synthetic signals to approximate this richness.

## Getting Started

### Prerequisites

Make sure you have **Python 3.8+** installed.  Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Generate Synthetic Telemetry

The `data_generator.py` script creates a CSV file of synthetic vehicle sensor readings and assigns a `maintenance_needed` label.  You can specify the number of samples and the output path.  For example:

```bash
python data_generator.py --num_samples 10000 --output data/vehicle_telemetry.csv
```

The script uses random distributions to approximate realistic ranges of engine temperature, oil pressure, vibration, mileage, brake wear and other signals.  Maintenance is flagged when thresholds are exceeded (e.g. engine temperature > 105 °C, oil pressure < 2 bar, brake wear > 70 %).

### Train the Machine‑Learning Model

Run the training script to build a predictive model:

```bash
python model_training.py --data_path data/vehicle_telemetry.csv --model_path models/carcare_model.pkl
```

The script loads the CSV, splits it into training and testing sets, fits a random forest classifier, evaluates accuracy, precision, recall and F1 score and saves the trained model for inference.  It also outputs a confusion matrix and feature importance plot.

### Launch the Dashboard

Use Streamlit to launch the interactive dashboard:

```bash
streamlit run dashboard.py -- --model_path models/carcare_model.pkl
```

Open the provided localhost URL (default `http://localhost:8501`) in your browser.  The dashboard lets you:

- Explore a sample of the dataset and view summary statistics.
- Visualise correlations between features using a heatmap.
- Adjust sensor values using sliders and receive an immediate prediction on whether maintenance is required.
- Inspect feature importance to understand which signals drive the model’s decisions.

### Extending the Project

This repository uses synthetic data to illustrate the workflow.  For a more advanced project:

1. **Integrate real OBD‑II/CAN data** – Use an ELM327 or AutoPi device to collect real‑time vehicle data.  Libraries such as `python‑obd` or `cantools` can decode OBD‑II PIDs and CAN messages.  Replace the synthetic generator with a module that logs signals directly from your vehicle.
2. **Experiment with algorithms** – Try gradient boosting, XGBoost or deep learning models (LSTM, CNN) on time‑series sensor data as described in research on predictive maintenance【731405785906734†L80-L98】.
3. **Deploy to the cloud** – Package the dashboard in a Docker container and deploy to Heroku, Azure Web Apps or AWS.  Use message queues (MQTT, Kafka) for streaming data ingestion.
4. **Apply to manufacturing equipment** – Adapt the sensor set to equipment used in car factories (robotic arms, conveyor belts, CNC machines) and apply the same predictive maintenance principles【468205953509044†L60-L77】.

## Repository Structure

```
carcareai/
├── data/                         # Folder for generated or collected datasets
├── models/                       # Saved machine learning models
├── images/                       # Screenshots and figures for README
├── data_generator.py             # Script to simulate vehicle telemetry
├── model_training.py             # ML pipeline for predictive maintenance
├── dashboard.py                  # Streamlit dashboard for visualization and inference
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

## References

- Nithin Subba Rao, *AI‑Driven Predictive Maintenance Using IoT in Automotive Manufacturing* (2025) – describes how IoT sensors and AI algorithms collect real‑time data to predict equipment failures in automotive factories【731405785906734†L14-L66】.  It outlines the sensor, data aggregation and analytics layers used in predictive maintenance frameworks【731405785906734†L128-L149】.
- **AutoPi Blog** – Explains the layers of vehicle data (sensors, OBD, context) and lists categories such as engine/performance, fuel/energy, location, safety and driving behaviour【371604006037961†L210-L249】.  It notes that combining signals (e.g. coolant temperature, ambient temperature and road grade) helps detect early problems【371604006037961†L255-L260】.
- **Automotive‑Technology.com** – Highlights how AI and digital twins transform automotive production, reducing downtime and enabling predictive maintenance【468205953509044†L60-L77】.  It notes that major OEMs like BMW use AI to achieve 95 % accuracy in predicting failures【468205953509044†L72-L77】.

---

*This project was created as a showcase for automotive analytics and predictive maintenance.  Feel free to fork, modify and improve it for your own portfolio.*