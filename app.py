import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

with open(os.path.join(BASE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

st.title("📊 Machine Learning Classification Model Comparison")
st.write(
    "Interactive evaluation of five classification models on the "
    "Breast Cancer Wisconsin (Diagnostic) test dataset."
)

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload test data CSV",
    type=["csv"],
    help="Upload a CSV containing the 30 model features and the diagnosis column.",
)

if uploaded_file is None:
    data_path = os.path.join(BASE_DIR, "test_data.csv")
    df = pd.read_csv(data_path)
    st.sidebar.info("No file uploaded. The supplied test_data.csv is being used.")
else:
    df = pd.read_csv(uploaded_file)

target_col = metadata["target_column"]
required_features = metadata["features"]

missing = [col for col in required_features + [target_col] if col not in df.columns]

if missing:
    st.error(
        "The uploaded CSV is missing required columns: "
        + ", ".join(missing)
    )
    st.stop()

X = df[required_features].copy()
y = df[target_col].copy()

# Accept either 0/1 labels or common UCI B/M labels.
if y.dtype == object:
    mapping = {"B": 0, "M": 1, "benign": 0, "malignant": 1}
    y = y.astype(str).str.strip().map(mapping)

if y.isna().any():
    st.error(
        "The target column must contain 0/1 or UCI-style B/M labels."
    )
    st.stop()

y = y.astype(int)

st.subheader("Uploaded Test Data")
st.write(f"Rows: **{len(df)}** | Features: **{len(required_features)}**")
st.dataframe(df.head(10), use_container_width=True)

selected_model = st.sidebar.selectbox(
    "Select a model",
    list(MODEL_FILES.keys()),
)

model_path = os.path.join(MODEL_DIR, MODEL_FILES[selected_model])

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)

pred = model.predict(X)
proba = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, pred)
auc = roc_auc_score(y, proba)
precision = precision_score(y, pred, zero_division=0)
recall = recall_score(y, pred, zero_division=0)
f1 = f1_score(y, pred, zero_division=0)
mcc = matthews_corrcoef(y, pred)

st.subheader(f"Evaluation Metrics — {selected_model}")

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")

c4, c5, c6 = st.columns(3)
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, pred)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)
ax.set_title(selected_model)
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Benign (0)", "Malignant (1)"])
ax.set_yticklabels(["Benign (0)", "Malignant (1)"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Classification Report")
report = classification_report(
    y,
    pred,
    target_names=["Benign (0)", "Malignant (1)"],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("All Model Results on the Uploaded Test Data")
all_rows = []

for name, filename in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        continue

    mdl = joblib.load(path)
    p = mdl.predict(X)
    pr = mdl.predict_proba(X)[:, 1]

    all_rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y, p),
        "AUC": roc_auc_score(y, pr),
        "Precision": precision_score(y, p, zero_division=0),
        "Recall": recall_score(y, p, zero_division=0),
        "F1": f1_score(y, p, zero_division=0),
        "MCC": matthews_corrcoef(y, p),
    })

results_df = pd.DataFrame(all_rows)
st.dataframe(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}",
    }),
    use_container_width=True,
)

winner = results_df.sort_values(
    by=["F1", "AUC", "MCC"], ascending=False
).iloc[0]

st.success(
    f"Current overall winner based on F1, with AUC and MCC as tie-breakers: "
    f"**{winner['ML Model Name']}**"
)

st.caption(
    "Dataset source: UCI Machine Learning Repository — "
    "Breast Cancer Wisconsin (Diagnostic)."
)
