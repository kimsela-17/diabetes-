
import streamlit as st
import joblib
import numpy as np

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Diabestie", page_icon="🩺", layout="centered")
st.title("Diabestie — Clinical Diabetes Assistant")
st.markdown("**Professional clinical guidance (educational use only).**")
st.markdown("---")

# ------------------------------
# Load Diabetes Prediction Model
# ------------------------------
@st.cache_resource
def load_diabetes_model():
    model = joblib.load("diabetes_model.pkl")
    return model

diabetes_model = load_diabetes_model()

# ------------------------------
# Dummy Med-Gemma simulator (professional clinical tone)
# ------------------------------
def medgemma_simulator(risk_label, features_dict):
    """
    Simulated clinical advice generator.
    risk_label: one of 'No diabetes risk', 'Pre-diabetes condition', 'Diabetes'
    features_dict: dict with keys like glucose, bmi, bp, age, insulin
    """
    glucose = features_dict.get("glucose")
    bmi = features_dict.get("bmi")
    age = features_dict.get("age")

    if risk_label == "No diabetes risk":
        advice = (
            "Assessment: Current input parameters do not indicate elevated diabetes risk. "
            "Recommendation: Continue routine preventive measures including a balanced diet, "
            "regular physical activity, and periodic monitoring of fasting glucose as per annual check-ups. "
            "Consider maintaining BMI within recommended ranges and consult a clinician if new symptoms arise."
        )
    elif risk_label == "Pre-diabetes condition":
        advice = (
            "Assessment: The provided data are consistent with a pre-diabetic state or elevated risk. "
            "Recommendation: Initiate lifestyle interventions — structured dietary modification to reduce simple carbohydrates, "
            "increase intake of vegetables and whole grains, and implement at least 150 minutes per week of moderate-intensity exercise. "
            "Arrange follow-up testing (HbA1c and fasting plasma glucose) and consult a healthcare provider for evaluation and consideration of pharmacologic prophylaxis if indicated."
        )
    else:  # Diabetes
        advice = (
            "Assessment: The parameters suggest a likelihood of diabetes. "
            "Recommendation: Expedite evaluation with confirmatory laboratory testing (fasting plasma glucose, HbA1c) and arrange clinical assessment. "
            "Initiate glycemic control measures as advised by a treating clinician, including diet, exercise, and consideration of pharmacotherapy. "
            "Advise close follow-up and assessment for diabetes-related complications."
        )

    # Add a brief tailored note
    tailored = f" (Clinical note: age={age}, BMI={bmi}, glucose={glucose})."
    return advice + tailored

# ------------------------------
# Prediction helper
# ------------------------------
def predict_diabetes(features):
    """
    features: numpy array or list in order [glucose, bmi, bp, age, insulin]
    returns: risk label string
    """
    prediction = diabetes_model.predict([features])
    if prediction[0] == 0:
        return "No diabetes risk"
    elif prediction[0] == 1:
        return "Pre-diabetes condition"
    else:
        return "Diabetes"

# ------------------------------
# User Input
# ------------------------------
st.subheader("Patient Data (for demonstration)")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 1, 120, 45)
    glucose = st.number_input("Glucose (mg/dL)", 0, 500, 110)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 26.0)
with col2:
    bp = st.number_input("Systolic Blood Pressure (mmHg)", 60, 250, 120)
    insulin = st.number_input("Insulin Level (µU/mL)", 0.0, 1000.0, 85.0)

if st.button("Evaluate"):
    with st.spinner("Performing clinical assessment..."):
        features = np.array([glucose, bmi, bp, age, insulin])
        risk_label = predict_diabetes(features)
        features_dict = {"glucose": glucose, "bmi": bmi, "bp": bp, "age": age, "insulin": insulin}
        advice = medgemma_simulator(risk_label, features_dict)

    st.markdown("### Clinical Assessment Result")
    st.write(f"**Prediction:** {risk_label}")
    st.markdown("---")
    st.markdown("### Clinical Recommendation")
    st.write(advice)
    st.markdown("---")
    st.caption("⚠️ Clinical disclaimer: This application is for educational and demonstration purposes only. It is not a substitute for professional medical evaluation, diagnosis, or treatment.")
