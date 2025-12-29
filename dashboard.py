"""
dashboard.py
-------------

Interactive Streamlit dashboard for the CarCareAI predictive maintenance project.  The
dashboard allows users to explore a sample of the dataset, view correlations between
features, adjust sensor values and obtain a prediction from the trained model.  To
launch the app, run:

    streamlit run dashboard.py -- --model_path models/carcare_model.pkl

The `--model_path` argument is optional; if omitted, the script attempts to load
`models/carcare_model.pkl` by default.  Use the dataset generated via
`data_generator.py` and the model saved by `model_training.py` for best results.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def load_model(model_path: str):
    """Load a trained RandomForestClassifier model from disk."""
    return joblib.load(model_path)


def load_dataset(sample_size: int = 5000) -> pd.DataFrame:
    """Load or generate a small synthetic dataset on the fly for exploration.

    This function calls the `generate_vehicle_data` function from `data_generator` to
    create a DataFrame if no CSV is available.  Because Streamlit apps often run
    statelessly, we avoid loading large files by generating a small sample.
    """
    from data_generator import generate_vehicle_data
    df = generate_vehicle_data(num_samples=sample_size)
    return df


def display_dataset(df: pd.DataFrame):
    """Show a preview of the dataset and summary statistics."""
    st.subheader("Dataset Sample")
    st.dataframe(df.head())
    st.markdown("**Summary Statistics**")
    st.write(df.describe())


def display_correlation_heatmap(df: pd.DataFrame):
    """Plot and display a correlation heatmap for the dataset."""
    corr = df.drop(columns=['maintenance_needed'], errors='ignore').corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)


def get_user_input(feature_ranges: dict) -> pd.DataFrame:
    """Create interactive sliders/inputs for each feature and return a DataFrame with one row."""
    st.sidebar.subheader("Enter Sensor Values")
    user_data = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        default_val = (min_val + max_val) / 2
        if feature in ['engine_temperature', 'oil_pressure', 'vibration', 'battery_voltage', 'coolant_temp', 'fuel_consumption']:
            user_data[feature] = st.sidebar.slider(
                label=feature.replace('_', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=(max_val - min_val) / 100.0
            )
        else:
            # Use numeric input for broader ranges
            user_data[feature] = st.sidebar.number_input(
                label=feature.replace('_', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val)
            )
    return pd.DataFrame([user_data])


def display_feature_importance(model, feature_names):
    """Plot feature importances of the trained model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    fig, ax = plt.subplots()
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette='viridis')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    st.pyplot(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the CarCareAI Streamlit dashboard.")
    parser.add_argument('--model_path', type=str, default='models/carcare_model.pkl',
                        help='Path to the trained model pickle file')
    return parser.parse_args()


def main():
    args = parse_args()
    st.set_page_config(page_title="CarCareAI Dashboard", layout="wide")
    st.title("CarCareAI Predictive Maintenance Dashboard")

    # Load model
    model_path = args.model_path
    if not Path(model_path).exists():
        st.warning(f"Model file not found at {model_path}. Please train the model first.")
        return
    model = load_model(model_path)

    # Load or generate data for exploration
    df = load_dataset(sample_size=1000)

    # Feature ranges for user input (min, max) based on synthetic generator ranges
    feature_ranges = {
        'engine_temperature': (60.0, 120.0),
        'oil_pressure': (1.0, 5.0),
        'vibration': (0.0, 0.1),
        'speed': (0.0, 200.0),
        'mileage': (0.0, 300000.0),
        'brake_wear': (0.0, 100.0),
        'battery_voltage': (9.0, 15.0),
        'coolant_temp': (40.0, 120.0),
        'engine_load': (0.0, 100.0),
        'fuel_consumption': (2.0, 15.0)
    }

    # Sidebar for user input
    user_input_df = get_user_input(feature_ranges)

    # Display dataset and correlations
    with st.expander("Explore Dataset"):
        display_dataset(df)
        display_correlation_heatmap(df)

    # Prediction
    st.subheader("Predict Maintenance Requirement")
    if st.button("Predict"):
        prediction = model.predict(user_input_df)[0]
        probability = model.predict_proba(user_input_df)[0][prediction]
        if prediction == 1:
            st.error(f"Maintenance Needed! Confidence: {probability:.2f}")
        else:
            st.success(f"No Immediate Maintenance Needed. Confidence: {probability:.2f}")
    else:
        st.write("Adjust the sliders and click predict to see results.")

    # Feature importance
    with st.expander("Model Feature Importance"):
        display_feature_importance(model, list(feature_ranges.keys()))


if __name__ == '__main__':
    main()