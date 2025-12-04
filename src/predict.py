# src/predict.py
import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.environ.get('MODEL_PATH', 'artifacts/lgbm.pkl')
SCALER_PATH = os.environ.get('SCALER_PATH', 'artifacts/scaler.pkl')
THRESHOLD_PATH = os.environ.get('THRESHOLD_PATH', 'artifacts/threshold.json')

def load_components():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def preprocess_input(record, scaler):
    df = pd.DataFrame([record])
    # scale Amount if present
    if 'Amount' in df.columns:
        df['Amount'] = scaler.transform(df[['Amount']])
    return df

def predict(record):
    model, scaler = load_components()
    X = preprocess_input(record, scaler)
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X)[:,1][0]
    else:
        prob = float(model.predict(X)[0])
    return {'score': float(prob), 'label': int(prob > 0.5)}
