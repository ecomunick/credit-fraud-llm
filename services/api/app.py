# services/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict
import uvicorn
import os

app = FastAPI(title="Credit Fraud POC API")

class Transaction(BaseModel):
    # include relevant fields; for demo we allow extra fields
    Time: float = None
    V1: float = None
    V2: float = None
    # ... add V3..V28 or accept a dict
    Amount: float

@app.post("/predict")
def predict_transaction(tx: dict):
    """
    Expects a dict of features; returns score and label
    """
    res = predict(tx)
    return res

@app.post("/explain")
def explain_transaction(tx: dict):
    """
    Returns model prediction + LLM-style explanation stub or actual LLM call
    """
    # run model predict and top features extraction (shap) here
    # For demo return structure:
    model_out = predict(tx)
    explanation = {
        "model_score": model_out['score'],
        "reasons": [
            "Top feature: V14 high value",
            "High amount relative to norm",
            "Unusual transaction time"
        ],
        "recommended_action": "Flag for manual review"
    }
    return {"prediction": model_out, "explanation": explanation}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
