# src/models.py
import joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def train_logreg(X, y, out_path='artifacts/logreg.pkl'):
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    return model

def train_rf(X, y, out_path='artifacts/rf.pkl'):
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    model.fit(X, y)
    joblib.dump(model, out_path)
    return model

def train_lgbm(X, y, out_path='artifacts/lgbm.pkl'):
    dtrain = lgb.Dataset(X, label=y)
    params = {'objective':'binary', 'metric':'auc', 'is_unbalance': True, 'verbosity': -1}
    model = lgb.train(params, dtrain, num_boost_round=200)
    joblib.dump(model, out_path)
    return model

def load_model(path):
    return joblib.load(path)
