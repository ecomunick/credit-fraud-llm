# Fraud Detection Project - Setup Guidelines

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