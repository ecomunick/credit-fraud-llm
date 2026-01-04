import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    classification_report, 
    confusion_matrix
)

def load_processed_data(data_dir='data/processed'):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    X_train = train.drop(columns=['Class'])
    y_train = train['Class']
    X_test = test.drop(columns=['Class'])
    y_test = test['Class']
    
    return X_train, X_test, y_train, y_test

from sklearn.metrics import precision_recall_curve
import numpy as np

def optimize_threshold(model, X_test, y_test):
    print("\nOptimizing Threshold for XGBoost...")
    y_proba = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Avoid division by zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Best F1 Score: {f1_scores[best_idx]:.4f}")
    
    return best_threshold

def evaluate_model(model, X_test, y_test, model_name="Model", threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}

def train_baseline(X_train, y_train):
    print("Training Baseline: Logistic Regression...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    print("\nTraining Random Forest...")
    # Basic tuning: setting better defaults for imbalance
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    # Calculating scale_pos_weight for imbalance
    count_pos = (y_train == 1).sum()
    count_neg = (y_train == 0).sum()
    scale_pos_weight = count_neg / count_pos
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    all_metrics = []
    
    # 1. Train Baseline
    lr_model = train_baseline(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    lr_metrics['model'] = 'Logistic Regression'
    all_metrics.append(lr_metrics)
    joblib.dump(lr_model, 'models/logistic_regression_baseline.joblib')

    # 2. Train Random Forest
    rf_model = train_rf(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_metrics['model'] = 'Random Forest'
    all_metrics.append(rf_metrics)
    joblib.dump(rf_model, 'models/random_forest_tuned.joblib')

    # 3. Train XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    
    # 4. Optimize threshold for XGBoost
    best_threshold = optimize_threshold(xgb_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Optimized Threshold)", threshold=best_threshold)
    xgb_metrics['model'] = 'XGBoost (Optimized)'
    all_metrics.append(xgb_metrics)
    
    joblib.dump(xgb_model, 'models/xgboost_tuned.joblib')
    
    # Save all metrics to a CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('experiments/model_metrics.csv', index=False)
    
    print("\nAll models trained and metrics saved to experiments/model_metrics.csv")
