# create environment for windows
# python -m venv myenv
# activate environment
# myenv\Scripts\activate
# pip install streamlit scikit-learn pandas seaborn numpy
# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scalerr.pkl', 'rb'))

st.title("Employee Attrition Prediction App")

# -------------------------
# Numerical Inputs
# -------------------------
age = st.number_input('Age', 18, 60, 30)
daily_rate = st.number_input('Daily Rate', 100, 1500, 500)
distance = st.number_input('Distance From Home', 1, 50, 10)
education = st.slider('Education (1-5)', 1, 5, 3)
env_sat = st.slider('Environment Satisfaction (1-4)', 1, 4, 3)
hourly_rate = st.number_input('Hourly Rate', 30, 150, 70)
job_involvement = st.slider('Job Involvement (1-4)', 1, 4, 3)
job_level = st.slider('Job Level (1-5)', 1, 5, 2)
job_sat = st.slider('Job Satisfaction (1-4)', 1, 4, 3)
monthly_income = st.number_input('Monthly Income', 1000, 20000, 5000)
monthly_rate = st.number_input('Monthly Rate', 2000, 30000, 10000)
num_companies = st.number_input('Num Companies Worked', 0, 10, 2)
percent_hike = st.slider('Percent Salary Hike', 10, 30, 15)
performance = st.slider('Performance Rating', 1, 4, 3)
relationship = st.slider('Relationship Satisfaction', 1, 4, 3)
stock_option = st.slider('Stock Option Level', 0, 3, 1)
total_years = st.number_input('Total Working Years', 0, 40, 10)
training = st.slider('Training Times Last Year', 0, 10, 3)
work_life = st.slider('Work Life Balance', 1, 4, 3)
years_company = st.number_input('Years At Company', 0, 40, 5)
years_role = st.number_input('Years In Current Role', 0, 20, 3)
years_promo = st.number_input('Years Since Last Promotion', 0, 15, 1)
years_manager = st.number_input('Years With Current Manager', 0, 20, 3)

# -------------------------
# Categorical Inputs
# -------------------------
business_travel = st.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
department = st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
education_field = st.selectbox('Education Field', ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other'])
gender = st.selectbox('Gender', ['Male', 'Female'])
job_role = st.selectbox('Job Role', [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative',
    'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
overtime = st.selectbox('OverTime', ['Yes', 'No'])


model_columns = pickle.load(open('columns.pkl','rb'))
# -------------------------
# Create DataFrame
# -------------------------
input_features = pd.DataFrame({
    'Age':[age],
    'BusinessTravel':[business_travel],
    'DailyRate':[daily_rate],
    'Department':[department],
    'DistanceFromHome':[distance],
    'Education':[education],
    'EducationField':[education_field],
    'EnvironmentSatisfaction':[env_sat],
    'Gender':[gender],
    'HourlyRate':[hourly_rate],
    'JobInvolvement':[job_involvement],
    'JobLevel':[job_level],
    'JobRole':[job_role],
    'JobSatisfaction':[job_sat],
    'MaritalStatus':[marital_status],
    'MonthlyIncome':[monthly_income],
    'MonthlyRate':[monthly_rate],
    'NumCompaniesWorked':[num_companies],
    'OverTime':[overtime],
    'PercentSalaryHike':[percent_hike],
    'PerformanceRating':[performance],
    'RelationshipSatisfaction':[relationship],
    'StockOptionLevel':[stock_option],
    'TotalWorkingYears':[total_years],
    'TrainingTimesLastYear':[training],
    'WorkLifeBalance':[work_life],
    'YearsAtCompany':[years_company],
    'YearsInCurrentRole':[years_role],
    'YearsSinceLastPromotion':[years_promo],
    'YearsWithCurrManager':[years_manager]
})
#apply Same encoding
input_encoded = pd.get_dummies(input_features)

if 'Attrition' in model_columns:
    model_columns = model_columns.drop('Attrition')
#match training columns
input_encoded = input_encoded.reindex(columns=model_columns,fill_value=0)
# Scale
input_scaled = scaler.transform(input_encoded)

# -------------------------
# Prediction
# -------------------------
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error(f'Employee is likely to leave  (Probability: {prob[0][1]:.2f})')
    else:
        st.success(f'Employee is likely to stay  (Probability: {prob[0][0]:.2f})')