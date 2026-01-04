import pandas as pd
import joblib
import shap
import os
import matplotlib.pyplot as plt

def load_data_and_model():
    model = joblib.load('models/xgboost_tuned.joblib')
    test = pd.read_csv('data/processed/test.csv')
    X_test = test.drop(columns=['Class'])
    y_test = test['Class']
    return model, X_test, y_test

def get_shap_explanation(model, X_sample):
    """
    Computes SHAP values for a single sample.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

def format_explanation_for_llm(X_sample, shap_values, feature_names, top_n=5):
    """
    Identifies top N features and formats them into a structured string for an LLM.
    """
    # For XGBoost TreeExplainer, shap_values is often an array [1, num_features]
    # We take the first row.
    if len(shap_values.shape) > 1:
        s_values = shap_values[0]
    else:
        s_values = shap_values
    
    # Create a Series for easy sorting
    shap_series = pd.Series(s_values, index=feature_names)
    top_features = shap_series.abs().sort_values(ascending=False).head(top_n)
    
    explanation_parts = []
    for feature, val in top_features.items():
        direction = "increased" if shap_series[feature] > 0 else "decreased"
        actual_value = X_sample[feature].values[0]
        explanation_parts.append(f"- {feature} (value: {actual_value:.4f}) {direction} fraud risk.")
        
    return "\n".join(explanation_parts)

if __name__ == "__main__":
    # Standard CLI execution logic
    os.makedirs('experiments', exist_ok=True)
    model, X_test, y_test = load_data_and_model()
    
    fraud_indices = y_test[y_test == 1].index
    sample_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
    X_sample = X_test.loc[[sample_idx]]
    
    print(f"Explaining Transaction at Index {sample_idx}")
    explainer, shap_values = get_shap_explanation(model, X_sample)
    
    structured_explanation = format_explanation_for_llm(X_sample, shap_values, X_test.columns)
    print("\n--- Structured Explanation for LLM ---")
    print(structured_explanation)
