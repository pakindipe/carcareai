import traceback
import joblib
import pandas as pd
from pathlib import Path

model_path = Path('models/carcare_model.pkl')
if not model_path.exists():
    print('ERROR: model file not found at', model_path)
    raise SystemExit(1)

try:
    model = joblib.load(model_path)
    sample = pd.DataFrame([{ 
        'engine_temperature': 90.0,
        'oil_pressure': 3.0,
        'vibration': 0.02,
        'speed': 60.0,
        'mileage': 100000.0,
        'brake_wear': 10.0,
        'battery_voltage': 12.5,
        'coolant_temp': 80.0,
        'engine_load': 30.0,
        'fuel_consumption': 8.0
    }])
    pred = model.predict(sample)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(sample)[0]
        proba = probs.max()
    print(f'PRED:{pred};PROB:{proba}')
except Exception as e:
    print('EXCEPTION during smoke test:')
    traceback.print_exc()
    raise
