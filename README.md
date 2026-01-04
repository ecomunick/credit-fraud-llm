# 💳 Credit Fraud Detection + LLM Explainability

## 📌 Problem Description
In the financial industry, identifying fraudulent credit card transactions is a critical challenge. Modern machine learning models provide high accuracy but often act as "black boxes," leaving risk analysts and fraud investigation teams without clear reasoning for a flagged transaction.

**The Goal:** Build a high-precision fraud detection system that not only classifies transactions but also provides **Natural Language Explanations** for its decisions using Large Language Models (LLMs). This bridge between technical metrics and human-friendly interpretation is the core value of this project.

## 🛠️ Project Roadmap
1. **EDA & Preprocessing**: Cleaned data, engineered temporal features (`Hour`), and handled heavy class imbalance (~0.17%).
2. **Modeling**: Trained and tuned multiple models (Logistic Regression, Random Forest, XGBoost).
3. **Interpretability**: Leveraged SHAP values to extract global and local feature importance.
4. **LLM Integration**: Translated technical SHAP outputs into plain-English reports for analysts.
5. **Deployment**: Flask/FastAPI service containerized with Docker.

## 🤖 Model Performance
Given the heavy class imbalance, we prioritize **PR-AUC** (Precision-Recall AUC) as our primary metric.

| Model | ROC-AUC | PR-AUC | Best F1-Score |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** (Baseline) | 0.9833 | 0.7535 | 0.11 |
| **Random Forest** (Tuned) | **0.9936** | 0.8630 | 0.84 |
| **XGBoost (Optimized Threshold)** | 0.9868 | **0.8689** | **0.87** |

*The XGBoost model, with a decision threshold of **0.879**, achieved the best balance of Precision (0.91) and Recall (0.83).*

## 🔍 Explainability (SHAP + LLM)
We use SHAP (SHapley Additive exPlanations) to identify the top 5 contributing factors for every flagging event.
- **SHAP Summary Plot**: Found in `experiments/shap_summary_plot.png`.
- **LLM Report**: The model passes feature weights (e.g., *“V14 value -4.68 increased risk”*) to OpenAI's GPT-4o-mini to generate an intuitive summary.


## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- OpenAI API Key (for explanations)

### 2. Dataset Download
We use the **Credit Card Fraud Detection Dataset** from ULB.
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Create a folder `data/raw` in the project root.
3. Unzip and place `creditcard.csv` inside `data/raw/`.

Alternatively, use the Kaggle CLI:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

### 3. Setup
```bash
# Clone the repository
git clone <repo_url>
cd fraud_detection

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# 1. Copy the template to create a real .env file
cp .env.template .env

# 2. Open .env and replace 'your_openai_api_key_here' with your actual key.
# The .env file is git-ignored to keep your secrets safe.
```

### 3. Execution Pipeline
```bash
# Prepare and clean data
python src/01_data_preparation.py

# Perform Feature Importance Analysis
python src/feature_importance.py

# Train and Evaluate Models
python src/train_model.py

# Generate Sample SHAP Explanation
python src/explain.py
```

## 📂 Project Structure
- `data/`: Raw data (git-ignored) and processed training/test sets.
- `notebooks/`: EDA and experimental data preparation.
- `src/`: core logic (01_data_preparation, train_model, explain, llm_utils).
- `models/`: Saved joblib model artifacts (Models and Scalers).
- `services/`: FastAPI implementation for real-time inference.
- `experiments/`: Saved metrics, correlation plots, and SHAP visualizations.

## 🚀 Deployment (Containerization)
The application is containerized using **Docker** for easy deployment to cloud platforms like Render or AWS.

### 1. Build the Image
```bash
docker build -t fraud-detection-api .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 --env-file .env fraud-detection-api
```

### 3. API Endpoints
- **GET `/health`**: Check system status.
- **POST `/predict`**: Returns fraud probability and classification.
- **POST `/explain`**: Returns prediction + technical SHAP factors + LLM natural language summary.

## 📺 Project Demo
![API Interaction Demo](experiments/api_demo.gif)

## 🧪 Testing the API
Once the server is running, you can test the **Explainability Pipeline** using the interactive documentation:

1. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
2. Expand **`POST /explain`**.
3. Click **Try it out**.
4. Paste the following **High-Risk (Fraudulent) JSON** into the request body:

```json
{
  "Time": 406,
  "Amount": 1000.0,
  "V1": -2.3122,
  "V2": 1.9519,
  "V3": -1.6098,
  "V4": 3.9979,
  "V5": -0.5221,
  "V6": -1.4265,
  "V7": -2.5373,
  "V8": 1.3916,
  "V9": -2.7700,
  "V10": -2.7722,
  "V11": 3.2020,
  "V12": -2.8999,
  "V13": -0.5952,
  "V14": -4.2892,
  "V15": 0.3897,
  "V16": -1.1407,
  "V17": -2.8300,
  "V18": -0.0168,
  "V19": 0.4169,
  "V20": 0.1269,
  "V21": 0.5172,
  "V22": -0.0350,
  "V23": -0.4652,
  "V24": 0.3201,
  "V25": 0.0445,
  "V26": 0.1778,
  "V27": 0.2611,
  "V28": -0.1432
}

or other dummy example data:
{
  "Time": 1000,
  "Amount": 45.99,
  "V1": 1.23,
  "V2": -0.54,
  "V3": 0.89,
  "V4": 0.12,
  "V5": -0.34,
  "V6": -0.12,
  "V7": 0.05,
  "V8": -0.01,
  "V9": 0.78,
  "V10": -0.23,
  "V11": 0.45,
  "V12": 0.67,
  "V13": -0.89,
  "V14": -0.12,
  "V15": 1.10,
  "V16": -0.45,
  "V17": 0.23,
  "V18": -0.67,
  "V19": 0.12,
  "V20": -0.05,
  "V21": -0.18,
  "V22": -0.45,
  "V23": 0.15,
  "V24": -0.32,
  "V25": 0.25,
  "V26": 0.89,
  "V27": -0.07,
  "V28": 0.01
}
```

5. Click the blue **Execute** button.
6. Scroll down to see the **`fraud_probability`** and the **`ai_explanation`** generated by the LLM!
