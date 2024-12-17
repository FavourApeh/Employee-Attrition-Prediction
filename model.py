# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import joblib

#Loading dataset
dataset = pd.read_csv('HR-Employee-Attrition.csv')
dataset.head()

dataset.columns

dataset.shape

dataset['Attrition'].unique().sum()

dataset['Attrition'].value_counts()

dataset.isnull().sum()

# Drop irrelevant or constant columns
columns_to_drop = ['EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours']
cleaned_data = dataset.drop(columns=columns_to_drop)

# Encode categorical variables
label_encodable_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField',
                        'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

label_encoders = {col: LabelEncoder() for col in label_encodable_cols}
for col in label_encodable_cols:
  cleaned_data[col] = label_encoders[col].fit_transform(cleaned_data[col])

# Verify the cleaned data
data_cleaned_info = {
    "Shape": cleaned_data.shape,
    "Sample Data": cleaned_data.head(),
    "Encoded Columns": label_encodable_cols
}

data_cleaned_info


# Visualize class imbalance in the target variable
plt.figure(figsize=(6, 4))
sns.countplot(data=cleaned_data, x='Attrition', palette='viridis')
plt.title('Class Imbalance in Attrition')
plt.xlabel('Attrition (0:No, 1: Yes)')
plt.ylabel('Count')
plt.show()


# Compute correlations and display the top 10 features related to Attrition
attrition_corr = cleaned_data.corr()['Attrition'].sort_values(ascending=False)
print("Top 10 features most correlated with Attrition:")
print(attrition_corr.head(10))

# Generated a correlation matrix
correlation_matrix = cleaned_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()


# Creating a copy of the cleaned data for feature engineering
fe_data = cleaned_data.copy()

# 1. Interaction Features
# -------------------------
# Interaction between 'Age' and 'JobSatisfaction' to capture how satisfaction varies with age
fe_data['Age_JobSatisfaction'] = fe_data['Age'] * fe_data['JobSatisfaction']

# Interaction between 'MonthlyIncome' and 'JobLevel' to understand income at different levels
fe_data['Income_JobLevel'] = fe_data['MonthlyIncome'] * fe_data['JobLevel']

# Interaction between 'DistanceFromHome' and 'OverTime' to check the effect of commuting and overtime
fe_data['Distance_OverTime'] = fe_data['DistanceFromHome'] * fe_data['OverTime']

# 2. Binning Continuous Variables
# --------------------------------
# Binning 'Age' into categories: Young, Mid-Career, Experienced
fe_data['Age_Group'] = pd.cut(
    fe_data['Age'],
    bins=[18, 30, 45, 60],  # Defining age ranges
    labels=['Young', 'Mid-Career', 'Experienced']
)

# Binning 'YearsAtCompany' into categories: Newcomer, Settled, Veteran
fe_data['YearsAtCompany_Group'] = pd.cut(
    fe_data['YearsAtCompany'],
    bins=[0, 5, 10, 40],  # Defining ranges for years
    labels=['Newcomer', 'Settled', 'Veteran']
)

# Encoding the new categorical variables
binned_cols = ['Age_Group', 'YearsAtCompany_Group']
for col in binned_cols:
    fe_data[col] = LabelEncoder().fit_transform(fe_data[col])

# 3. Scaling Numerical Features
# ------------------------------
# Apply StandardScaler to numerical features for scaling
scaler = StandardScaler()
numerical_features = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Age_JobSatisfaction',
    'Income_JobLevel', 'Distance_OverTime'
]

fe_data[numerical_features] = scaler.fit_transform(fe_data[numerical_features])

# Check the enhanced dataset
fe_data_info = {
    "Shape": fe_data.shape,
    "Sample Data": fe_data.head(),
    "New Features": [
        'Age_JobSatisfaction', 'Income_JobLevel', 'Distance_OverTime',
        'Age_Group', 'YearsAtCompany_Group'
    ]
}

print("Feature Engineering Completed!")
print(fe_data_info)


#Improved Categorical Encoding
# ---------------------------------
# Use One-Hot Encoding for nominal features and Ordinal Encoding for ordinal features
# Define nominal and ordinal columns
nominal_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
ordinal_cols = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating',
                'RelationshipSatisfaction', 'WorkLifeBalance']

# One-Hot Encoding for nominal columns
fe_data = pd.get_dummies(fe_data, columns=nominal_cols, drop_first=True)

# Ordinal Encoding for ordinal columns
ordinal_mappings = {
    'Education': [1, 2, 3, 4, 5],  # 1: Below College, 5: Doctor
    'EnvironmentSatisfaction': [1, 2, 3, 4],  # 1: Low, 4: Very High
    'JobInvolvement': [1, 2, 3, 4],  # 1: Low, 4: Very High
    'JobSatisfaction': [1, 2, 3, 4],  # 1: Low, 4: Very High
    'PerformanceRating': [1, 2, 3, 4],  # 1: Low, 4: Outstanding
    'RelationshipSatisfaction': [1, 2, 3, 4],  # 1: Low, 4: Very High
    'WorkLifeBalance': [1, 2, 3, 4]  # 1: Bad, 4: Best
}

for col, mapping in ordinal_mappings.items():
    fe_data[col] = fe_data[col].map(lambda x: mapping.index(x) + 1 if x in mapping else x)

#Scaling Numerical Features
# ------------------------------
# Apply StandardScaler to numerical features for scaling
scaler = StandardScaler()
numerical_features = [
    'Age', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'Age_JobSatisfaction',
    'Income_JobLevel', 'Distance_OverTime'
]

fe_data[numerical_features] = scaler.fit_transform(fe_data[numerical_features])

# Check the enhanced dataset
fe_data_info = {
    "Shape": fe_data.shape,
    "Sample Data": fe_data.head(),
    "New Features": [
        'Age_JobSatisfaction', 'Income_JobLevel', 'Distance_OverTime',
        'Age_Group', 'YearsAtCompany_Group'
    ]
}

print("Feature Engineering Completed!")
print(fe_data_info)



# Data splitting
X = fe_data.drop('Attrition', axis=1)
y = fe_data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Applying SMOTE to balance the training data
smote = SMOTE(random_state = 42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Calculate scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Train baseline XGBoost model
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"  # Avoids a warning
)
xgb_model.fit(X_train, y_train)

# Evaluate on test data
y_pred_xgb = xgb_model.predict(X_test)

# Compute metrics
xgb_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_xgb),
    "Precision": precision_score(y_test, y_pred_xgb),
    "Recall": recall_score(y_test, y_pred_xgb),
    "F1-Score": f1_score(y_test, y_pred_xgb),
}
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)

# Print results
print("XGBoost Metrics:", xgb_metrics)
print("Confusion Matrix:\n", conf_matrix_xgb)

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "scale_pos_weight": [scale_pos_weight],
}

# Create XGBoost classifier
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="f1",  # Focus on F1-Score due to class imbalance
    cv=3,          # 3-fold cross-validation
    verbose=1,
    n_jobs=-1,
)

# Perform GridSearch
grid_search.fit(X_train, y_train)

# Get best parameters and model
best_xgb = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the best model from GridSearchCV
y_pred_best_xgb = best_xgb.predict(X_test)
y_proba_best_xgb = best_xgb.predict_proba(X_test)[:, 1]

# Predict probabilities for the test data
y_proba_best_xgb = best_xgb.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Set optimal threshold
optimal_threshold = 0.40  # Based on threshold analysis
y_pred_final = (y_proba_best_xgb >= optimal_threshold).astype(int)

# Compute evaluation metrics with the chosen threshold
final_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_final),
    "Precision": precision_score(y_test, y_pred_final),
    "Recall": recall_score(y_test, y_pred_final),
    "F1-Score": f1_score(y_test, y_pred_final),
    "ROC-AUC": roc_auc_score(y_test, y_proba_best_xgb),
}
conf_matrix_final = confusion_matrix(y_test, y_pred_final)



# Print final evaluation results
print("Final Metrics with Threshold = {:.2f}:".format(optimal_threshold))
print(final_metrics)
print("Final Confusion Matrix:\n", conf_matrix_final)

# Calculate PR-AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_best_xgb)
pr_auc = auc(recall, precision)
print(f"PR-AUC: {pr_auc:.4f}")

# Visualize Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'Precision-Recall Curve (PR-AUC = {pr_auc:.4f})')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid()
plt.show()

# Visualize ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba_best_xgb)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Classifier')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()

# Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_final, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attrition', 'Attrition'], yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix (Threshold = {:.2f})'.format(optimal_threshold))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Get feature importances from the trained model
feature_importances = best_xgb.feature_importances_

# Create a DataFrame for better readability
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Print top features
print("Top 10 Features by Importance:")
print(importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"].iloc[:10], importance_df["Importance"].iloc[:10], color="skyblue")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.xlabel("Feature Importance")
plt.title("Top 10 Most Influential Features")
plt.show()

# Save the trained model
model_filename = 'xgboost_employee_attrition_model.pkl'
joblib.dump(best_xgb, model_filename)

print(f"Model saved to {model_filename}")

# Load the saved model
loaded_model = joblib.load('xgboost_employee_attrition_model.pkl')

# Use it for predictions
y_pred = loaded_model.predict(X_test)

