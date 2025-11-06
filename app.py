import streamlit as st
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------
# 🔹 Page Configuration
# ------------------------------
st.set_page_config(page_title="Diabetes AI Assistant", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes AI Assistant")
st.markdown("### Predict your diabetes risk and get educational advice powered by AI.")
st.markdown("---")

# ------------------------------
# 🔹 Load Diabetes Prediction Model
# ------------------------------
@st.cache_resource
def load_diabetes_model():
    # Load your trained model (make sure diabetes_model.pkl is in the same folder)
    model = joblib.load("diabetes_model.pkl")
    return model

diabetes_model = load_diabetes_model()

# ------------------------------
# 🔹 Load Med-Gemma Model
# ------------------------------
@st.cache_resource
def load_medgemma_model():
    model_name = "google/med-gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, med_model = load_medgemma_model()

# ------------------------------
# 🔹 Define Functions
# ------------------------------
def predict_diabetes(features):
    prediction = diabetes_model.predict([features])
    if prediction[0] == 0:
        return "No diabetes risk"
    elif prediction[0] == 1:
        return "Pre-diabetes condition"
    else:
        return "Diabetes"

def generate_advice(risk_level):
    prompt = (
        f"The patient has {risk_level}. "
        "Provide a short and friendly health explanation with advice "
        "about healthy habits, exercise, and diet in simple language."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = med_model.generate(**inputs, max_new_tokens=120)
    advice = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return advice

# ------------------------------
# 🔹 User Input Section
# ------------------------------
st.subheader("Enter your health data")

age = st.number_input("Age", 1, 100, 30)
glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 110)
bmi = st.number_input("BMI (Body Mass Index)", 0.0, 60.0, 22.5)
bp = st.number_input("Blood Pressure (mmHg)", 0, 200, 80)
insulin = st.number_input("Insulin Level", 0, 300, 90)

if st.button("🔍 Check Risk"):
    with st.spinner("Analyzing your data..."):
        features = np.array([glucose, bmi, bp, age, insulin])
        risk_result = predict_diabetes(features)
        advice_text = generate_advice(risk_result)

    st.success(f"**Prediction:** {risk_result}")
    st.markdown("---")
    st.markdown("### 🧠 AI Health Advice")
    st.write(advice_text)
    st.markdown("---")
    st.caption("⚠️ This chatbot is for educational and research use only. It does not replace professional medical diagnosis or treatment.")

