import os
import sys
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

# Add root directory to sys.path to import our custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.explain import get_shap_explanation, format_explanation_for_llm
from src.llm_utils import get_fraud_explanation

app = FastAPI(title="Credit Fraud Detection API", description="AI-powered fraud detection with LLM explanations.")

# Load models and scalers at startup
try:
    model = joblib.load('models/xgboost_tuned.joblib')
    robust_scaler = joblib.load('models/robust_scaler.joblib')
    std_scaler = joblib.load('models/std_scaler.joblib')
    print("Models and scalers loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

class Transaction(BaseModel):
    # Map raw transaction features
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

def preprocess(tx: Transaction):
    # 1. Engineering: Extract Hour from Time
    hour = (tx.Time // 3600) % 24
    
    # 2. Scaling
    amount_scaled = robust_scaler.transform([[tx.Amount]])[0][0]
    hour_scaled = std_scaler.transform([[hour]])[0][0]
    
    # 3. Form input dataframe for XGBoost
    # Note: Order must match training (V1-V28, amount_scaled, hour_scaled)
    data = {f'V{i}': [getattr(tx, f'V{i}')] for i in range(1, 29)}
    data['amount_scaled'] = [amount_scaled]
    data['hour_scaled'] = [hour_scaled]
    
    return pd.DataFrame(data)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Credit Fraud Detection API",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost"}

@app.post("/predict")
def predict(tx: Transaction):
    try:
        df_input = preprocess(tx)
        proba = model.predict_proba(df_input)[:, 1][0]
        
        # Using the optimized threshold from our training
        threshold = 0.8791
        label = 1 if proba >= threshold else 0
        
        return {
            "fraud_probability": round(float(proba), 4),
            "is_fraud": bool(label),
            "threshold": threshold
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(tx: Transaction):
    try:
        # 1. Get Prediction
        df_input = preprocess(tx)
        proba = model.predict_proba(df_input)[:, 1][0]
        threshold = 0.8791
        is_fraud = bool(proba >= threshold)
        
        # 2. Get SHAP factors
        _, shap_values = get_shap_explanation(model, df_input)
        structured_info = format_explanation_for_llm(df_input, shap_values, df_input.columns)
        
        # 3. Get LLM Natural Language Explanation
        ai_explanation = get_fraud_explanation(structured_info, proba, is_fraud)
        
        return {
            "fraud_probability": round(float(proba), 4),
            "is_fraud": is_fraud,
            "technical_factors": structured_info,
            "ai_explanation": ai_explanation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
