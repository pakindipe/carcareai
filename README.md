# CarCareAI: Predictive Maintenance Dashboard for Vehicle Health

## Overview

Modern vehicles generate continuous streams of telemetry from powertrain, chassis, safety, and infotainment systems. This data includes engine temperature, oil pressure, vibration, mileage, and hundreds of other signals that describe the real time operating state of a vehicle. When analysed correctly, it enables outcomes such as reduced roadside failures, faster diagnostics, and improved fuel efficiency.

CarCareAI is an end to end example project that demonstrates how an automotive manufacturer could transform this raw telemetry into a data driven predictive maintenance system. The project simulates vehicle sensor data, trains a machine learning model to predict maintenance needs, and presents the results in an interactive web dashboard.

This repository is designed as a portfolio ready project for developers and data scientists interested in automotive analytics, machine learning, and applied AI.

## Project Features

- **Synthetic Vehicle Telemetry Generator** – A script generates a realistic multivariate dataset of vehicle sensor readings.  Features include engine temperature, oil pressure, vibration, brake wear, battery voltage and more.  Maintenance labels are assigned based on domain‑driven rules (e.g. high temperature and low oil pressure indicate potential failure).
- **Machine learning pipeline** – A training script loads the synthetic data, performs train/test splitting, and fits a random forest classifier to predict whether maintenance is needed.  It evaluates the model with standard metrics and saves the trained model to disk for later inference.
- **Streamlit dashboard** – An interactive web application allows you to explore the dataset and obtain predictions.  Users can adjust sensor values with sliders or text boxes and see whether the model predicts maintenance is required.  The dashboard also displays feature importance and a correlation heatmap.
- **Modular design** – Clear separation of components makes it easy to extend the project or integrate real vehicle data sources.

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

Possible extensions include integrating real OBD II or CAN data, experimenting with alternative models such as gradient boosting or deep learning, deploying the application to the cloud, or adapting the workflow to predictive maintenance for manufacturing equipment.

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

---

*This project was created as a showcase for automotive analytics and predictive maintenance.  Feel free to fork, modify and improve it for your own portfolio.*
