"""
data_generator.py
------------------

Generate a synthetic dataset of vehicle telemetry signals for predictive maintenance.  The
script produces a CSV file containing continuous features (e.g., engine temperature, oil
pressure) and a binary target indicating whether maintenance is required.  Synthetic
labels are assigned based on domain rules: high engine temperature, low oil pressure,
excessive vibration, worn brakes, low battery voltage, high coolant temperature or high
engine load all contribute to maintenance needs.

Usage:
    python data_generator.py --num_samples 10000 --output data/vehicle_telemetry.csv

The default number of samples is 10,000.  The output directory will be created if it
doesn't exist.

"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_vehicle_data(num_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Create a DataFrame containing synthetic vehicle sensor readings and a maintenance label.

    Parameters
    ----------
    num_samples : int
        Number of rows (vehicles/time points) to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic sensor values and a binary `maintenance_needed` column.
    """
    rng = np.random.default_rng(seed=random_state)

    # Generate continuous features based on typical automotive sensor ranges.
    engine_temperature = rng.normal(loc=90.0, scale=15.0, size=num_samples)
    oil_pressure = rng.normal(loc=3.0, scale=0.5, size=num_samples)
    vibration = rng.normal(loc=0.02, scale=0.01, size=num_samples)
    speed = rng.uniform(low=0.0, high=150.0, size=num_samples)
    mileage = rng.normal(loc=120_000.0, scale=40_000.0, size=num_samples)
    brake_wear = rng.uniform(low=0.0, high=100.0, size=num_samples)
    battery_voltage = rng.normal(loc=12.5, scale=1.0, size=num_samples)
    coolant_temp = rng.normal(loc=80.0, scale=10.0, size=num_samples)
    engine_load = rng.uniform(low=0.0, high=100.0, size=num_samples)
    fuel_consumption = rng.normal(loc=8.0, scale=2.0, size=num_samples)

    # Ensure mileage, brake wear and other features are within realistic bounds.
    mileage = np.clip(mileage, a_min=0, a_max=None)
    brake_wear = np.clip(brake_wear, a_min=0, a_max=100)
    fuel_consumption = np.clip(fuel_consumption, a_min=0, a_max=None)

    # Determine maintenance labels based on thresholds.
    # A vehicle needs maintenance if any of the conditions below is true.
    maintenance_conditions = [
        engine_temperature > 105.0,      # overheating engine
        oil_pressure < 2.0,              # low oil pressure
        vibration > 0.05,                # excessive vibration
        brake_wear > 70.0,               # worn brake pads/discs
        battery_voltage < 11.0,          # weak battery
        coolant_temp > 95.0,             # coolant overheating
        engine_load > 90.0               # sustained high load
    ]

    # Combine conditions to compute maintenance flag.
    maintenance_needed = np.any(maintenance_conditions, axis=0).astype(int)

    # Assemble DataFrame.
    df = pd.DataFrame({
        'engine_temperature': engine_temperature,
        'oil_pressure': oil_pressure,
        'vibration': vibration,
        'speed': speed,
        'mileage': mileage,
        'brake_wear': brake_wear,
        'battery_voltage': battery_voltage,
        'coolant_temp': coolant_temp,
        'engine_load': engine_load,
        'fuel_consumption': fuel_consumption,
        'maintenance_needed': maintenance_needed
    })

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save the DataFrame to a CSV file.  Creates parent directories if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to save.
    output_path : str
        Path to the output CSV file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle telemetry data.")
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of data samples to generate (default: 10000)')
    parser.add_argument('--output', type=str, default='data/vehicle_telemetry.csv',
                        help='Output CSV file path (default: data/vehicle_telemetry.csv)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_vehicle_data(num_samples=args.num_samples)
    save_dataset(df, args.output)


if __name__ == '__main__':
    main()
