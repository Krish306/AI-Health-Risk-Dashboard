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
    age = st.number_input("Age", min_value=1, max_value=110, value=25)
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=110)
    cholesterol = st.number_input("Cholesterol", min_value=80, max_value=400, value=110)
    max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=180)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=300, value=90)

with col_in2:
    st.subheader("Lifestyle")
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease_existing = st.selectbox("Existing Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    activity_level = st.selectbox("Physical Activity", ["Active (3+ times/week)", "Moderate", "Sedentary"], index=1)
    diet_quality = st.selectbox("Diet Quality", ["High (Whole foods)", "Average", "Low (Processed/High Sugar)"], index=1)
    stress_level = st.select_slider(
        "Chronic Stress Level", 
        options=["Low", "Moderate", "High"], 
    )

    # DYNAMIC COLOR FEEDBACK
    if stress_level == "Low":
        st.markdown(":green[**Status: Low Impact on Heart/Stroke**]")
    elif stress_level == "Moderate":
        st.markdown(":orange[**Status: Moderate Physiological Strain**]")
    else:
        st.markdown(":red[**Status: High Cortisol/Arterial Tension**]")

st.divider()

if st.button("Calculate Risk Score", type="primary"): # Made the button blue/primary
    
    # HEART Calculation
    heart_input = np.array([[age, blood_pressure, cholesterol, max_heart_rate]])
    heart_risk = heart_model.predict_proba(heart_input)[0][1]
    if smoking == 1: heart_risk += 0.03 
    if hypertension == 1: heart_risk += 0.04
    heart_risk = min(heart_risk, 1.0)

    # DIABETES Calculation
    diabetes_input = np.array([[glucose, bmi, age]])
    diabetes_risk = diabetes_model.predict_proba(diabetes_input)[0][1]
    if hypertension == 1: diabetes_risk += 0.04
    diabetes_risk = min(diabetes_risk, 1.0)

    # STROKE Calculation
    stroke_input = np.array([[age, hypertension, heart_disease_existing, glucose]])
    stroke_risk = stroke_model.predict_proba(stroke_input)[0][1]
    if smoking == 1: stroke_risk += 0.03
    stroke_risk = min(stroke_risk, 1.0)
    
    if activity_level == "Sedentary":
        heart_risk += 0.03
        diabetes_risk += 0.03
        stroke_risk += 0.03
    elif activity_level == "Active (3+ times/week)":
        heart_risk -= 0.02 # Benefit for being active
        diabetes_risk -= 0.02

    # 2. Diet Quality Adjustment
    if diet_quality == "Low (Processed/High Sugar)":
        diabetes_risk += 0.03
        heart_risk += 0.03
    elif diet_quality == "High (Whole foods)":
        diabetes_risk -= 0.02
        heart_risk -= 0.01

    # Apply caps again at the very end
    heart_risk = max(0.0, min(heart_risk, 1.0))
    diabetes_risk = max(0.0, min(diabetes_risk, 1.0))
    stroke_risk = max(0.0, min(stroke_risk, 1.0))
    
    if stress_level == "High":
        heart_risk += 0.05
        stroke_risk += 0.04
        diabetes_risk += 0.02
    elif stress_level == "Moderate":
        heart_risk += 0.02
        
    # Final normalization (Keep between 0 and 1)
    heart_risk = max(0.0, min(heart_risk, 1.0))
    diabetes_risk = max(0.0, min(diabetes_risk, 1.0))
    stroke_risk = max(0.0, min(stroke_risk, 1.0))
    st.header("Results")
    
    # Metrics display
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric(label="Heart Risk", value=f"{heart_risk * 100:.1f}%")
    with res_col2:
        st.metric(label="Diabetes Risk", value=f"{diabetes_risk * 100:.1f}%")
    with res_col3:
        st.metric(label="Stroke Risk", value=f"{stroke_risk * 100:.1f}%")    

    # We check Heart (>25%), Stroke (>20%), and Diabetes (>25%) for High Risk
    if (heart_risk > 0.20 or stroke_risk > 0.15 or diabetes_risk > 0.20 or 
        blood_pressure >= 180 or glucose >= 250 or bmi >= 45):
        st.error("**High Risk / Critical Value Detected:** Please consult a physician immediately. Some of your vitals are in a range that requires professional evaluation.")
    
    # 2. MODERATE OVERRIDE (Yellow Alert)
    # Triggered by moderate AI risk OR serious lifestyle habits
    elif (heart_risk > 0.10 or diabetes_risk > 0.10 or 
          smoking == 1 or blood_pressure >= 140 or glucose >= 125):
        
        # Specific message if it's just the smoking/vitals causing the alert
        if smoking == 1 and heart_risk <= 0.10:
            st.warning("**Lifestyle Warning:** While your current risk score is low, **smoking** and/or your current vitals are significant long-term threats to your health.")
        else:
            st.warning("**Moderate Risk:** Lifestyle modifications and clinical monitoring are advised.")
    
    # 3. THE "CLEAN BILL" (Green Alert)
    else:
        st.success("**Low Risk:** Maintain your healthy lifestyle habits!")
        
    st.divider()
    st.subheader("Personalized Risk Mitigation Plan")
    st.caption("This framework provides evidence-based suggestions for risk reduction. Consult a healthcare professional before implementation.")
    
    plan_col1, plan_col2 = st.columns(2)
    
    with plan_col1:
        st.markdown("### Clinical Management")
        # Specific Triggers
        if blood_pressure > 130 or hypertension == 1:
            st.write("- **Hypertension Monitoring:** Regular BP tracking is vital to prevent long-term arterial damage.")
        if stress_level == "High":
            st.write("- **Cortisol Regulation:** Consider MBSR to lower systemic inflammation caused by chronic stress.")
        if age > 50 or glucose > 100:
            st.write("- **Diagnostic Screenings:** Maintain regular lipid panels and HbA1c checks to monitor metabolic trends.")
        
        # ALWAYS SHOW THIS (General Clinical Advice)
        st.write("- **Baseline Vitals:** Even with low risk scores, an annual physical is recommended to remain in good health.")
        st.write("- **Physician Review:** Share these AI-stratified results with your doctor to discuss long-term preventive care.")

    with plan_col2:
        st.markdown("### Lifestyle Interventions")
        # Specific Triggers
        if activity_level == "Sedentary":
            st.write("- **Physical Activity:** Aim for 150 min/week of moderate aerobic exercise to enhance cardiovascular resilience.")
        elif activity_level == "Active (3+ times/week)":
            st.write("- **Performance Maintenance:** Continue your current activity level; consider adding resistance training for metabolic health.")
            
        if diet_quality == "Low (Processed/High Sugar)":
            st.write("- **Nutritional Optimization:** Shift toward a Mediterranean or DASH-style diet to improve lipid profiles.")
        else:
            st.write("- **Nutritional Consistency:** Maintain a high-fiber, whole-food diet to support long-term heart and gut health.")

        if smoking == 1:
            st.write("- **Smoking Cessation:** This is the #1 modifiable risk factor. Seek professional support quit smoking and/or tobacco use.")
        
        # ALWAYS SHOW THIS (General Lifestyle Advice)
        st.write("- **Hydration & Sleep:** Prioritize 7-9 hours of quality sleep and consistent hydration to support systemic recovery.")
            
    with st.expander("See Risk Calculation Logic"):
        st.write("""
        This dashboard uses a **Hybrid Intelligence Model**:
        - **Base Prediction:** Trained on clinical datasets (Cleveland, NHANES) using Logistic Regression.
        - **Heuristic Layer:** Adjusts scores based on lifestyle factors (Smoking, Stress, Physical Activity) that are often underrepresented in instantaneous vitals.
        - **Risk Stratification Levels:** Risk alerts are triggered based on standard epidemiological cut-offs.
        """)    
       
st.divider()
st.caption("**Disclaimer:** This AI Health Risk Dashboard is a prototype developed for a TKS focus project. "
           "The predictions are based on credible public health datasets (like the Cleveland Heart Disease dataset) "
           "and are for informational purposes only. This tool does not provide medical advice "
           "and should not be used for self-diagnosis. Please consult a qualified healthcare professional.")

# --- SIDEBAR POLISH ---
with st.sidebar:
    st.title("About Me")
    st.markdown(f"""
    **Name:** Krish Kaliraj  
    **Focus:** AI Integrated Healthcare  
    
    I am a TKS Innovate Student passionate about maximizing clinical efficiency and improving health outcomes through AI.
    """)
    
    st.divider()
    
    st.subheader("Project Technicals")
    st.info("""
    - **Models:** Logistic Regression / Random Forest
    - **Datasets Used:** Cleveland Clinic, Kaggle, NHANES
    - **Logic:** Multi-modal Heuristic Layer 
    """)
    
    st.subheader("Roadmap")
    st.write("""
    1. **Diverse Data:** Adding more diverse demographic data to improve accuracy.
    2. **Explainability:** Implementing SHAP values to explain risk factors, which features (like Glucose vs BMI) are influencing the risk scores more.
    3. **Doctor Portal:**
        - Automated PDF Reports for patient-doctor consultations.
        - Risk Factor Attribution: showing the doctor exactly *why* a risk is high.
        - Longitudinal Tracking: monitoring patient risk trends over time.
    """)

 
