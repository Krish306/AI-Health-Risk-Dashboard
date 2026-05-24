import streamlit as st
import numpy as np
import pickle

# Page Config (Makes it look wider and more professional)
st.set_page_config(page_title="AI Health Dashboard", layout="wide")

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

# --- IMPROVED INPUT LAYOUT ---
col_in1, col_in2 = st.columns(2)

with col_in1:
    st.subheader("Physical Vitals")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=110)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=150)
    max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=180)

with col_in2:
    st.subheader("Lifestyle")
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=90)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease_existing = st.selectbox("Existing Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

if st.button("Calculate Risk Score", type="primary"): # Made the button blue/primary
    
    # HEART Calculation
    heart_input = np.array([[age, blood_pressure, cholesterol, max_heart_rate]])
    heart_risk = heart_model.predict_proba(heart_input)[0][1]
    if smoking == 1: heart_risk += 0.03 
    if hypertension == 1: heart_risk += 0.05
    heart_risk = min(heart_risk, 1.0)

    # DIABETES Calculation
    diabetes_input = np.array([[glucose, bmi, age]])
    diabetes_risk = diabetes_model.predict_proba(diabetes_input)[0][1]
    if hypertension == 1: diabetes_risk += 0.03
    diabetes_risk = min(diabetes_risk, 1.0)

    # STROKE Calculation
    stroke_input = np.array([[age, hypertension, heart_disease_existing, glucose]])
    stroke_risk = stroke_model.predict_proba(stroke_input)[0][1]
    if smoking == 1: stroke_risk += 0.05
    stroke_risk = min(stroke_risk, 1.0)
    
    st.header("Results")
    
    # Metrics display
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric(label="Heart Risk", value=f"{heart_risk * 100:.1f}%")
    with res_col2:
        st.metric(label="Diabetes Risk", value=f"{diabetes_risk * 100:.1f}%")
    with res_col3:
        st.metric(label="Stroke Risk", value=f"{stroke_risk * 100:.1f}%")    
   
    # Alert Banner
    if heart_risk > 0.20 or stroke_risk > 0.15:
        st.error(" **High Risk Detected:** Please consult a physician for a formal evaluation as soon as possible.")
    elif heart_risk > 0.10:
        st.warning("**Moderate Risk:** Consider lifestyle changes and regular monitoring.")
    else:
        st.success("**Low Risk:** Maintain your healthy lifestyle habits!")
       
st.divider()
st.caption("**Disclaimer:** This AI Health Risk Dashboard is a prototype developed for a TKS focus project. "
           "The predictions are based on credible public health datasets (like the Cleveland Heart Disease dataset) "
           "and are for informational purposes only. This tool does not provide medical advice "
           "and should not be used for self-diagnosis. Please consult a qualified healthcare professional.")

# --- SIDEBAR POLISH ---
with st.sidebar:
    st.title("User Profile")
    st.markdown(f"""
    **Name:** Krish Kaliraj  
    **Focus:** AI integrated Healthcare  
    
    I am a TKS Innovate Student passionate about maximizing clinical efficiency and improving health outcomes through AI.
    """)
    
    st.divider()
    
    st.subheader("Project Technicals")
    st.info("""
    - **Models:** Logistic Regression / Random Forest
    - **Data Sources:** Cleveland Clinic, Kaggle, NHANES
    - **Logic:** Multi-modal Heuristic Layer 
    """)
    
    st.subheader(" Roadmap")
    st.write("""
    1. **Diverse Data:** Adding more diverse demographic data to improve accuracy.
    2. **Explainability:** Implementing SHAP values to explain risk factors, which features (like Glucose vs BMI) are influencing the risk scores more.
    3. **Doctor Portal:**
        - Automated PDF Reports for patient-doctor consultations.
        - Risk Factor Attribution: showing the doctor exactly *why* a risk is high.
        - Longitudinal Tracking: monitoring patient risk trends over time.
    """)

 
