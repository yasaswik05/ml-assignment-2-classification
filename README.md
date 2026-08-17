# Machine Learning Assignment 2 - Classification Model Comparison

## a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on the same public classification dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application is then used to demonstrate the trained models and their results on test data.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

The dataset contains 569 instances and 30 real-valued predictive features. The target is binary: benign versus malignant. The UCI repository identifies this as a classification dataset and reports 569 instances and 30 features, satisfying the assignment's minimum requirements of 500 instances and 12 features.

UCI dataset page:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

For this project, the target is encoded as:
- 0 = Benign
- 1 = Malignant

The supplied `test_data.csv` contains the held-out 20% test set used for the evaluation.

## c. GitHub Repository Link
https://github.com/yasaswik05/ml-assignment-2-classification

## d. Models Used

The assignment PDF says "6 ML models" in one sentence, but the actual numbered list and comparison-table rows specify five models. Therefore, this implementation follows the five models explicitly named in the assignment:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | **0.9960** | 0.9750 | **0.9286** | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | **1.0000** | 0.8333 | 0.9091 | 0.8715 |
| **Random Forest (Ensemble)** | **0.9737** | 0.9944 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |


### Observations

Based on the model evaluation results:

- **Logistic Regression:** Achieved strong overall performance with an accuracy of **96.49%**, AUC of **99.60%**, and F1-score of **95.12%**. It provides a strong linear baseline with high precision and recall.
- **Decision Tree:** Achieved the lowest overall performance among the evaluated models, with an accuracy of **92.11%** and F1-score of **88.61%**. Its performance may be affected by the selected tree structure and splitting decisions.
- **kNN:** Delivered good performance with an accuracy of **95.61%** and F1-score of **93.83%**. Its distance-based approach benefits from appropriately scaled or standardized features.
- **Naive Bayes:** Achieved an accuracy of **93.86%** and perfect precision (**100%**), but its recall was relatively lower at **83.33%**. This suggests that while its positive predictions were highly reliable, it missed some positive instances.
- **Random Forest (Ensemble):** Achieved the best overall performance, with the highest accuracy (**97.37%**), F1-score (**96.30%**), and MCC (**94.42%**). It also achieved **100% precision** and **92.86% recall**, demonstrating strong and balanced classification performance.


- **Overall Winner:** **Random Forest (Ensemble)** performed best across the primary evaluation metrics, while **Logistic Regression** provided the strongest linear baseline. The results indicate that the ensemble approach was particularly effective for capturing the patterns in the dataset.


## Streamlit App

The application provides:

1. Test-data CSV upload.
2. Model selection dropdown.
3. Accuracy, AUC, Precision, Recall, F1, and MCC.
4. Confusion matrix.
5. Classification report.
6. A comparison table showing all implemented models on the uploaded test data.

## Project Structure

ml_assignment_2_project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── metadata.json
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest_ensemble.joblib

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the models
python train_models.py

### 3. Start the Streamlit application
streamlit run app.py

## Streamlit Community Cloud Link

https://ml-assignment-2-classification-abkwsuappk2rfazsbgos5ma.streamlit.app/

## Academic Integrity Note

This project is intended as a learning implementation. Before submission, review the code, run it yourself in the BITS Virtual Lab, understand the model choices and metrics, customize the README observations in your own words, and make your own GitHub commits as required by the assignment.

## BITS Virtual Lab Screenshot

The assignment requires one screenshot showing execution on BITS Virtual Lab. Take this screenshot yourself after running the project in the BITS Virtual Lab and include it in the final PDF.

## Submission Checklist

- GitHub repository link works - Yes
- app.py is present - Yes
- train_models.py is present - Yes
- requirements.txt is present - Yes
- README.md is present - Yes
- test_data.csv is present - Yes
- All five saved model files are present in model/ - Yes
- Streamlit application opens without errors - Yes
- CSV upload works - Yes
- Model selection works - Yes
- All five evaluation metrics are displayed - Yes
- Confusion matrix/classification report is displayed - Yes
- One BITS Virtual Lab execution screenshot is captured - Yes
- GitHub URL is updated in this README - Yes
- Streamlit URL is updated in this README - Yes
- README content is included in the final PDF - Yes
