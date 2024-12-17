# Overview
This project uses machine learning to predict employee attrition by analyzing demographic, job-related, and satisfaction metrics. It provides insights into the likelihood of employees leaving and explores class imbalance handling, feature engineering, and hyperparameter tuning for optimal performance.
# Features
Exploratory Data Analysis (EDA): Includes visualizations of class imbalance, correlation matrices, and feature importance.

Feature Engineering: Created interaction terms, binned continuous variables, and applied improved categorical encoding.

Class Imbalance Handling: Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
# Model Training:
Baseline model using XGBoost.
Fine-tuned model through GridSearchCV for hyperparameter optimization.
# Evaluation Metrics:
Accuracy, Precision, Recall, F1-Score, PR-AUC, and ROC-AUC.
# Visualization:
Precision-Recall Curve.
ROC Curve.
Confusion Matrix.
Feature Importances.
# Models Used
Baseline Model: XGBoost Classifier with default parameters.
Optimized Model: XGBoost Classifier tuned using GridSearchCV with parameters such as:
  Number of estimators (n_estimators)
  Maximum depth (max_depth)
  Learning rate (learning_rate)
  Subsampling ratio (subsample)
  Class imbalance weight (scale_pos_weight)
# Tools & Technologies
Programming Language: Python
# Libraries:
Data Manipulation: pandas, numpy
Visualization: matplotlib, seaborn
Machine Learning: scikit-learn, xgboost
Imbalance Handling: imblearn
# Model Deployment:
Included code to integrate the trained XGBoost model into an application (app.py).
# Dataset
The dataset contains anonymized employee information and includes features such as:
Demographics: Age, Gender, Marital Status
Job-Related Metrics: Monthly Income, Job Satisfaction, Years at Company
Performance Metrics: Performance Rating, Training Times Last Year
Target Variable: Attrition (binary: 0 = No, 1 = Yes)
# Project Workflow
Data Cleaning: Removed irrelevant columns and handled missing values.

Feature Engineering: Created new interaction terms and binned categorical variables.

Data Preprocessing: Applied scaling and encoding techniques.

Model Training:
Split the data into training and testing sets.
Balanced the training data using SMOTE.
Trained and fine-tuned the XGBoost model.

Evaluation: Compared baseline and optimized models on various performance metrics.

Visualization: Generated insights through plots and metrics visualization.
# Results
Optimized Model Metrics (with threshold tuning at 0.40):
Accuracy: ~85%
Precision: ~54%
Recall: ~53%
F1-Score: ~53%
PR-AUC: ~0.58
ROC-AUC: ~0.82
# Feature Importance:
Key influential features include Monthly Income, OverTime, and Job Satisfaction.
