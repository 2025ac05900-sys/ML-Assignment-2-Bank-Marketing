# 📊 Bank Marketing Classification Dashboard


## 🎯 Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit using machine learning classification algorithms. Multiple classification models were implemented and compared using various evaluation metrics to identify the best-performing model.

---

## 📂 Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

### Dataset Characteristics

- Total Records: **4521**
- Number of Features: **16**
- Target Variable: **y**
- Problem Type: **Binary Classification**

### Target Variable

| Value | Meaning |
|---------|---------|
| yes | Customer subscribed to term deposit |
| no | Customer did not subscribe to term deposit |

The dataset contains customer demographic information, financial details, and marketing campaign related attributes used for predicting subscription decisions.

---

## 🤖 Machine Learning Models Implemented

The following classification models were implemented and evaluated:

✅ Logistic Regression

✅ Decision Tree Classifier

✅ K-Nearest Neighbor (KNN)

✅ Naive Bayes Classifier

✅ Random Forest Classifier

---

## 📈 Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.8232 | 0.8908 | 0.3716 | 0.7788 | 0.5031 | 0.4533 |
| Decision Tree | 0.8011 | 0.8054 | 0.3403 | 0.7788 | 0.4737 | 0.4222 |
| KNN | 0.8884 | 0.7997 | 0.5517 | 0.1538 | 0.2406 | 0.2492 |
| Naive Bayes | 0.8210 | 0.7881 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| Random Forest | 0.8862 | 0.9079 | 0.5051 | 0.4808 | 0.4926 | 0.4287 |

---

## 🔍 Model Observations

### Logistic Regression

- Strong baseline classifier.
- Highest MCC score among all models.
- Excellent recall performance.
- Provides interpretable predictions.

### Decision Tree

- Captures non-linear patterns effectively.
- Good recall performance.
- Slightly lower overall predictive capability compared to Logistic Regression and Random Forest.

### K-Nearest Neighbor (KNN)

- Highest Accuracy.
- Very low Recall and F1 Score.
- Misses a large number of positive class instances.

### Naive Bayes

- Computationally efficient.
- Moderate overall performance.
- Independence assumption limits predictive capability on this dataset.

### Random Forest

- Highest AUC Score.
- High Accuracy.
- Strong balance between Precision and Recall.
- Robust ensemble-based classification performance.

---

## 🏆 Overall Winner

### Random Forest Classifier

Random Forest achieved the best overall balance of predictive performance.

**Reasons:**

✅ Highest AUC Score (0.9079)

✅ High Accuracy (0.8862)

✅ Strong F1 Score

✅ Robust Ensemble Learning

✅ Good Generalization Capability

Therefore, Random Forest was selected as the best-performing model for the Bank Marketing Dataset.

---

## 🌐 Streamlit Dashboard Features

The deployed Streamlit application includes:

- 📁 CSV File Upload
- 🤖 Model Selection Dropdown
- 📈 Evaluation Metrics Dashboard
- 🔷 Confusion Matrix Visualization
- 📋 Classification Report
- 🔍 Prediction Results Viewer
- 🏆 All Model Comparison Dashboard
- ⬇️ CSV Export Functionality

---

## 📸 Application Screenshot

> Add screenshot after deployment

<img width="1780" height="970" alt="image" src="https://github.com/user-attachments/assets/66211048-923c-4a15-9c7a-213fdd58c406" />


```text
screenshots/dashboard.png
```

---

## 🚀 Streamlit Application

Live Application:

```
https://ml-assignment-2-bank-marketing-rhqshacrvswzrojqj2zhq8.streamlit.app/
---

## 🔗 GitHub Repository

Repository URL:

```
https://github.com/2025ac05900-sys/ML-Assignment-2-Bank-Marketing/

---

## 📁 Project Structure

```
ML_Assignment_2/

│── app.py
│── requirements.txt
│── README.md
│── test_data.csv

│── model/
│     ├── ML_Assignment_2.ipynb
│     ├── model_comparison_results.csv
│     ├── decision_tree.pkl
│     ├── knn.pkl
│     ├── logistic_regression.pkl
│     ├── naive_bayes.pkl
│     └── random_forest.pkl
```
Note: The Jupyter Notebook (ML_Assignment_2.ipynb) contains complete model development, training, evaluation, and model serialization code. The .pkl files are the trained models used by the deployed Streamlit application.
---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## 👨‍💻 Author

**Shubham Vishwakarma**

M.Tech (Artificial Intelligence & Machine Learning)

BITS Pilani WILP

Student ID : 2025AC05900

---
