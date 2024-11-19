import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load your trained model and feature order
model = joblib.load('stroke_model.pkl')  # Replace with your actual model path
fit_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

# Preprocess the input data
def preprocess_input(age, gender, ever_married, work_type, residence_type, smoking_status, 
                     heart_disease, hypertension, avg_glucose_level, bmi):
    # Create DataFrame for input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'smoking_status': [smoking_status],
        'heart_disease': [heart_disease],
        'hypertension': [hypertension],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi]
    })

    # Label encoding for categorical features
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoder = LabelEncoder()
    for feature in categorical_features:
        input_data[feature] = encoder.fit_transform(input_data[feature])

    # Scaling numerical features
    numerical_features = ['age', 'heart_disease', 'hypertension', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])

    # Reorder the input data to match the training order
    input_data = input_data[fit_order]

    # Return the processed input
    return input_data

# Define custom CSS for centering the inputs and adding styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 30px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 10px;
        }
        .input-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            width: 80%;
            margin: auto;
        }
        .input-container > div {
            width: 45%;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            margin: 10px;
        }
        .result-box {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            width: 50%;
            margin: auto;
        }
        .high-risk {
            background-color: #f44336;
            color: white;
        }
        .low-risk {
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title at the top of the page
st.markdown("<div class='title'>Stroke Risk Prediction</div>", unsafe_allow_html=True)

# Create the form with inputs in two columns
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)

    # First Column
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', min_value=18, max_value=120, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
        heart_disease = st.number_input('Heart Disease (0 = No, 1 = Yes)', min_value=0, max_value=1, value=0)
        hypertension = st.number_input('Hypertension (0 = No, 1 = Yes)', min_value=0, max_value=1, value=0)

    with col2:
        work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'Children'])
        residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
        smoking_status = st.selectbox('Smoking Status', ['Never smoked', 'Formerly smoked', 'Smokes', 'Unknown'])
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=90.0)
        bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)

    st.markdown("</div>", unsafe_allow_html=True)

# Button to predict
if st.button('Predict'):
    # Preprocess the input data
    user_input = preprocess_input(age, gender, ever_married, work_type, residence_type, smoking_status, 
                                  heart_disease, hypertension, avg_glucose_level, bmi)

    # Make a prediction
    prediction = model.predict(user_input)

    # Display the prediction result in a styled box
    if prediction[0] == 1:
        st.markdown(f"<div class='result-box high-risk'>Prediction: High Risk of Stroke!</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box low-risk'>Prediction: Low Risk of Stroke!</div>", unsafe_allow_html=True)
