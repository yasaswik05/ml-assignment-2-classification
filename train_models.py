"""
Training script for ML Assignment 2.

Run this file once before starting the Streamlit application:
    python train_models.py

It creates:
- model/*.joblib
- test_data.csv
- metrics.csv
- metadata.json

Dataset:
UCI Breast Cancer Wisconsin (Diagnostic), represented here by the
same dataset available through scikit-learn's built-in copy.
"""

import json
import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series((data.target == 0).astype(int), name="diagnosis")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
    ),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),
}

filename_map = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

rows = []

for name, model in models.items():
    print(f"Training: {name}")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    joblib.dump(model, os.path.join(MODEL_DIR, filename_map[name]))

test_data = X_test.copy()
test_data["diagnosis"] = y_test.values
test_data.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)

metrics = pd.DataFrame(rows)
metrics.to_csv(os.path.join(BASE_DIR, "metrics.csv"), index=False)

metadata = {
    "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
    "source": "UCI Machine Learning Repository",
    "target_column": "diagnosis",
    "positive_class": 1,
    "negative_class": 0,
    "feature_count": len(X.columns),
    "instance_count": len(X),
    "test_size": 0.20,
    "random_state": 42,
    "features": list(X.columns),
}

with open(os.path.join(BASE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining completed successfully.")
print("\nEvaluation results:")
print(metrics.to_string(index=False))
print("\nCreated model files in:", MODEL_DIR)
print("Created:", os.path.join(BASE_DIR, "test_data.csv"))
