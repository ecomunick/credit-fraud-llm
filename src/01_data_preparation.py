import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler, StandardScaler
from data_utils import load_raw, train_test_split_save

def prepare_data():
    print("Loading data...")
    raw_path = os.path.join("data", "raw", "creditcard.csv")
    df = load_raw(raw_path)

    print("Cleaning duplicates...")
    # Keep fraud duplicates, remove non-fraud ones
    duplicates = df[df.duplicated(keep=False)]
    duplicates_0 = duplicates[duplicates['Class'] == 0]
    df = df.drop(duplicates_0.index)

    print("Engineering features...")
    # Extract hour
    df['Hour'] = (df['Time'] // 3600) % 24

    print("Scaling features...")
    robust_scaler = RobustScaler()
    std_scaler = StandardScaler()

    df['amount_scaled'] = robust_scaler.fit_transform(df[['Amount']])
    df['hour_scaled'] = std_scaler.fit_transform(df[['Hour']])
    
    # Save scalers for deployment
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(robust_scaler, 'models/robust_scaler.joblib')
    joblib.dump(std_scaler, 'models/std_scaler.joblib')

    print("Selecting features for modeling...")
    # Selecting V1-V28 and the two new scaled features + Class
    model_features = ['V' + str(i) for i in range(1, 29)] + ['amount_scaled', 'hour_scaled', 'Class']
    df_model = df[model_features]

    print("Splitting and saving data...")
    train, test = train_test_split_save(df_model, out_dir='data/processed')
    
    print(f"Data preparation complete. Shape of train: {train.shape}, Shape of test: {test.shape}")

if __name__ == "__main__":
    prepare_data()
