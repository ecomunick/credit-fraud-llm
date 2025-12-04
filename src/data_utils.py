# src/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os

def load_raw(path):
    return pd.read_csv(path)

def train_test_split_save(df, target='Class', test_size=0.2, random_state=42, out_dir='data/processed'):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=random_state)
    os.makedirs(out_dir, exist_ok=True)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(os.path.join(out_dir,'train.csv'), index=False)
    test.to_csv(os.path.join(out_dir,'test.csv'), index=False)
    return train, test
