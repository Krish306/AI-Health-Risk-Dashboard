import streamlit as st
import numpy as np
import pickle

st.title("AI Health Risk Dashboard")
st.write("Enter your health information below to estimate risk levels.")

# Load models
with open("heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)

with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

with open("stroke_model.pkl", "rb") as f:
    stroke_model = pickle.load(f)

st.header("Enter Your Information")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=25)
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=110)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=150)
max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=180)

bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=90)

hypertension = st.selectbox("Hypertension (1 = Yes, 0 = No)", [0, 1])
heart_disease_existing = st.selectbox("Existing Heart Disease (1 = Yes, 0 = No)", [0, 1])
smoking = st.selectbox("Smoking (1 = Yes, 0 = No)", [0, 1])

if st.button("Calculate Risk"):

    # HEART (matches training exactly)
    heart_input = np.array([[age, blood_pressure, cholesterol, max_heart_rate]])
    heart_risk = heart_model.predict_proba(heart_input)[0][1]

    # DIABETES (make sure this matches your diabetes training file)
    diabetes_input = np.array([[glucose, bmi, age]])
    diabetes_risk = diabetes_model.predict_proba(diabetes_input)[0][1]

    # STROKE (matches retrained stroke model)
    stroke_input = np.array([[age, hypertension, heart_disease_existing, glucose]])
    stroke_risk = stroke_model.predict_proba(stroke_input)[0][1]

    st.header("Results")
    st.write(f"Heart Disease Risk: {heart_risk * 100:.2f}%")
    st.write(f"Diabetes Risk: {diabetes_risk * 100:.2f}%")
    st.write(f"Stroke Risk: {stroke_risk * 100:.2f}%")
