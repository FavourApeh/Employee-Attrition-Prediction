import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('xgboost_employee_attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the expected feature names for the model
expected_features = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
    'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'Age_JobSatisfaction', 'Income_JobLevel', 'Distance_OverTime',
    'Age_Group', 'YearsAtCompany_Group', 'BusinessTravel_1',
    'BusinessTravel_2', 'Department_1', 'Department_2', 'EducationField_1',
    'EducationField_2', 'EducationField_3', 'EducationField_4',
    'EducationField_5', 'Gender_1', 'JobRole_1', 'JobRole_2', 'JobRole_3',
    'JobRole_4', 'JobRole_5', 'JobRole_6', 'JobRole_7', 'JobRole_8',
    'MaritalStatus_1', 'MaritalStatus_2', 'OverTime_1'
]

# Streamlit app
def main():
    st.title("Employee Attrition Prediction App")

    # Input fields for user input
    st.sidebar.header("Input Employee Data")

    # User input for each feature
    user_inputs = {}
    user_inputs['Age'] = st.sidebar.number_input("Age", min_value=18, max_value=65, step=1, value=30)
    user_inputs['DailyRate'] = st.sidebar.number_input("Daily Rate", min_value=100, max_value=2000, step=1, value=800)
    user_inputs['DistanceFromHome'] = st.sidebar.number_input("Distance From Home", min_value=1, max_value=100, step=1, value=10)
    user_inputs['Education'] = st.sidebar.slider("Education (1-5)", min_value=1, max_value=5, step=1, value=3)
    user_inputs['EnvironmentSatisfaction'] = st.sidebar.slider("Environment Satisfaction (1-4)", min_value=1, max_value=4, step=1, value=3)
    user_inputs['HourlyRate'] = st.sidebar.number_input("Hourly Rate", min_value=20, max_value=100, step=1, value=50)
    user_inputs['JobInvolvement'] = st.sidebar.slider("Job Involvement (1-4)", min_value=1, max_value=4, step=1, value=3)
    user_inputs['JobLevel'] = st.sidebar.slider("Job Level (1-5)", min_value=1, max_value=5, step=1, value=2)
    user_inputs['JobSatisfaction'] = st.sidebar.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, step=1, value=3)
    user_inputs['MonthlyIncome'] = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, step=100, value=5000)
    user_inputs['MonthlyRate'] = st.sidebar.number_input("Monthly Rate", min_value=1000, max_value=50000, step=100, value=15000)
    user_inputs['NumCompaniesWorked'] = st.sidebar.number_input("Number of Companies Worked", min_value=0, max_value=10, step=1, value=1)
    user_inputs['PercentSalaryHike'] = st.sidebar.number_input("Percent Salary Hike", min_value=5, max_value=30, step=1, value=15)
    user_inputs['PerformanceRating'] = st.sidebar.slider("Performance Rating (1-4)", min_value=1, max_value=4, step=1, value=3)
    user_inputs['RelationshipSatisfaction'] = st.sidebar.slider("Relationship Satisfaction (1-4)", min_value=1, max_value=4, step=1, value=3)
    user_inputs['StockOptionLevel'] = st.sidebar.slider("Stock Option Level (0-3)", min_value=0, max_value=3, step=1, value=1)
    user_inputs['TotalWorkingYears'] = st.sidebar.number_input("Total Working Years", min_value=0, max_value=40, step=1, value=10)
    user_inputs['TrainingTimesLastYear'] = st.sidebar.number_input("Training Times Last Year", min_value=0, max_value=10, step=1, value=2)
    user_inputs['WorkLifeBalance'] = st.sidebar.slider("Work Life Balance (1-4)", min_value=1, max_value=4, step=1, value=3)
    user_inputs['YearsAtCompany'] = st.sidebar.number_input("Years At Company", min_value=0, max_value=40, step=1, value=5)
    user_inputs['YearsInCurrentRole'] = st.sidebar.number_input("Years In Current Role", min_value=0, max_value=20, step=1, value=2)
    user_inputs['YearsSinceLastPromotion'] = st.sidebar.number_input("Years Since Last Promotion", min_value=0, max_value=20, step=1, value=1)
    user_inputs['YearsWithCurrManager'] = st.sidebar.number_input("Years With Current Manager", min_value=0, max_value=20, step=1, value=3)
    user_inputs['BusinessTravel_2'] = st.sidebar.number_input("Business Travel Two", min_value=0, max_value=20, step=1, value=3)
    user_inputs['Department_1'] = st.sidebar.number_input("Department One", min_value=0, max_value=20, step=1, value=3)
    user_inputs['Department_2'] = st.sidebar.number_input("Department Two", min_value=0, max_value=20, step=1, value=3)
    user_inputs['JobRole_2'] = st.sidebar.number_input("Job Two", min_value=0, max_value=20, step=1, value=3)
    user_inputs['JobRole_8'] = st.sidebar.number_input("JobRole Eight", min_value=0, max_value=20, step=1, value=3)
    user_inputs['MaritalStatus_1'] = st.sidebar.number_input("Marital Status one", min_value=0, max_value=20, step=1, value=3)
    user_inputs['MaritalStatus_2'] = st.sidebar.number_input("Marital Status Two", min_value=0, max_value=20, step=1, value=3)
    user_inputs['OverTime_1'] = st.sidebar.number_input("Over Time One", min_value=0, max_value=20, step=1, value=3)



    # Converting user input to a DataFrame
    input_data = pd.DataFrame(user_inputs, index=[0])

    # Align with model's expected feature names
    input_data = input_data.reindex(columns=expected_features, fill_value=0)

    # Display input data for debugging
    st.subheader("Input Data")
    st.write(input_data)

    # Perform prediction
    if st.button("Predict"):
        try:
            prediction_proba = model.predict_proba(input_data)[:, 1]  # Probability of attrition
            st.success(f"The probability of attrition is {prediction_proba[0]:.2%}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
