# src/features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib, os

def scale_amount(df, scaler_path='artifacts/scaler.pkl'):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    if not os.path.exists(scaler_path):
        scaler = StandardScaler()
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df['Amount_scaled'] = scaler.transform(df[['Amount']])
    df = df.drop(columns=['Amount'])
    return df
