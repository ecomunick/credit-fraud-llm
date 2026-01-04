ML Zoomcamp Capstone Project
Evaluation Criteria
The project will be evaluated using these criteria:

Problem description
0 points: Problem is not described
1 point: Problem is described in README birefly without much details
2 points: Problem is described in README with enough context, so it's clear what the problem is and how the solution will be used
EDA
0 points: No EDA
1 point: Basic EDA (looking at min-max values, checking for missing values)
2 points: Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis) For images: analyzing the content of the images. For texts: frequent words, word clouds, etc
Model training
0 points: No model training
1 point: Trained only one model, no parameter tuning
2 points: Trained multiple models (linear and tree-based). For neural networks: tried multiple variations - with dropout or without, with extra inner layers or without
3 points: Trained multiple models and tuned their parameters. For neural networks: same as previous, but also with tuning: adjusting learning rate, dropout rate, size of the inner layer, etc.
Exporting notebook to script
0 points: No script for training a model
1 point: The logic for training the model is exported to a separate script
Reproducibility
0 points: Not possitble to execute the notebook and the training script. Data is missing or it's not easiliy accessible
1 point: It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the data
Model deployment
0 points: Model is not deployed
1 point: Model is deployed (with Flask, BentoML or a similar framework)
Dependency and enviroment management
0 points: No dependency management
1 point: Provided a file with dependencies (requirements.txt, pipfile, bentofile.yaml with dependencies, etc)
2 points: Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to activate the env
Containerization
0 points: No containerization
1 point: Dockerfile is provided or a tool that creates a docker image is used (e.g. BentoML)
2 points: The application is containerized and the README describes how to build a container and how to run it
Cloud deployment
0 points: No deployment to the cloud
1 point: Docs describe clearly (with code) how to deploy the service to cloud or kubernetes cluster (local or remote)
2 points: There's code for deployment to cloud or kubernetes cluster (local or remote). There's a URL for testing - or video/screenshot of testing it
Total max 16 points

# My project, or goal is: Capstone 1: Credit Fraud Detection + LLM Explainability
Capstone 1 – Detailed proposal (high-score + portfolio)
📌 Problem

Predict fraudulent credit card transactions and explain model decisions in natural language.

Target audience:

Risk analysts

Fraud investigation teams

Models

Train at least 3:

Logistic Regression (baseline)

Random Forest

XGBoost / LightGBM

Tuning:

class weights

max depth / learning rate

threshold optimization for recall vs precision

Metrics:

ROC-AUC

Precision-Recall AUC

Confusion matrix at different thresholds

🤖 LLM Explainability (your differentiator)

Pipeline:

Use SHAP to extract top contributing features

Pass structured explanation to an LLM

Generate analyst-friendly text:

“This transaction was flagged because the amount is unusually high compared to recent behavior and occurred at an atypical time.”

This is excellent recruiter signal:

ML + explainability

LLM used responsibly, not as hype

🚀 Deployment

Flask or FastAPI

/predict endpoint

/explain endpoint (LLM)

Docker:

single Dockerfile

clean README instructions

Optional cloud:

Render / Fly.io / AWS EC2 (screenshot is enough)

# Fraud Detection Project - Setup Guidelines

### Phase 1: Project Setup

## 1. Project Structure
Create the following project directories:

```
fraud_detection/
├── data/
│ └── raw/ # Raw datasets
├── notebooks/ # Jupyter notebooks for EDA / experimentation
├── src/ # Scripts for preprocessing, training, etc.
├── models/ # Saved trained models
└── docs/ # Plots, reports, README, etc.
```


You can create them in one command:

```bash
mkdir -p fraud_detection/{data/raw,notebooks,src,models,docs}
```

## 2. Virtual Environment
Create and activate a virtual environment:

```bash
cd fraud_detection
python3 -m venv fraude_venv
source fraude_venv/bin/activate   # macOS/Linux
# OR Windows PowerShell: .\fraude_venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap fastapi "uvicorn[standard]" joblib python-dotenv streamlit kaggle matplotlib seaborn
```

## 3. Kaggle API Setup

1. Go to Kaggle Account → API
2. Click Create New API Token → download `kaggle.json`
3. Place the file in `~/.kaggle/`:
```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
``` 
4. Test authentication:
```bash
kaggle datasets list -s creditcard
``` 

## 4. Dataset Download

We use the **Credit Card Fraud Detection Dataset** from ULB (Machine Learning Group):

Dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download the dataset via Kaggle CLI:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

Check that the CSV exists:
```bash
ls -lh data/raw
# creditcard.csv should be ~150 MB
```

## 5. Folder for EDA Notebooks
- Place exploratory notebooks in notebooks/
- Scripts for reproducible preprocessing in src/

Example:
```bash
notebooks/01_eda_creditcard.ipynb
src/01_data_preparation.py
```

## 6. Save preprocessed data as data/processed/train.csv for modeling
- Tip: keep it reproducible — the script should be runnable standalone.

### check if all the steps in phase 1 was acomplished for a criteria of success in Capstone Project
The project will be evaluated using these criteria:

Problem description
0 points: Problem is not described
1 point: Problem is described in README birefly without much details
2 points: Problem is described in README with enough context, so it's clear what the problem is and how the solution will be used
EDA
0 points: No EDA
1 point: Basic EDA (looking at min-max values, checking for missing values)
2 points: Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis) For images: analyzing the content of the images. For texts: frequent words, word clouds, etc

### Phase 2: Model Building
## Building the models
- Let's build at least 3 models:
    - Logistic Regression (baseline)
    - Random Forest
    - XGBoost / LightGBM
    - (Optional) simple NN for embeddings fusion
    - tuning:
        - class weights
        - max depth / learning rate
        - threshold optimization for recall vs precision
    
Also include:

class imbalance handling

threshold optimization

precision–recall curves for risk tiers

metrics:

ROC-AUC

Precision-Recall AUC

Confusion matrix at different thresholds


Interpretability (huge plus)

You should include:

SHAP for tabular model

similarity explanations for images

LLM-generated analyst explanations (optional but powerful)

Example output:

“This receipt was flagged due to unusually high amount, visual similarity to previously submitted receipts, and mismatch between OCR-extracted total and transaction amount.”

This aligns exactly with:

“Create interpretable scoring frameworks for manual review teams.”

9️⃣ Deployment (keep it clean, not fancy)

API endpoints:

/predict

/score

/explain

Tech:

FastAPI or Flask

Docker

Optional cloud (Render is enough)

Do not overengineer streaming systems — just describe them in README.