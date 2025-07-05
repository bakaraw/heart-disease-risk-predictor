import streamlit as st
import os
import numpy as np
import joblib
import gdown
import time

# Google Drive model file ID
file_id = "1uAUgca7bg42dFFrIJta1PnlShHgqykYS"
url = f"https://drive.google.com/uc?id={file_id}"

# Download only if not already downloaded
model_path = "heart_disease_model.joblib"
if not os.path.exists(model_path):
    with st.spinner("🔄 Downloading model from Google Drive..."):
        progress = st.progress(0)
        for i in range(0, 100, 10):
            time.sleep(0.1)  # Simulate progress
            progress.progress(i)
        gdown.download(url, model_path, quiet=False)
        progress.progress(100)

# Load model
model = joblib.load(model_path)

# Mapping for age category
age_order = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3,
    '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7,
    '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11,
    '80 or older': 12
}

# Mapping helpers
def binary_input(label): return st.selectbox(label, ["No", "Yes"]) == "Yes"
def encode_ordinal(value, categories): return categories.index(value)

# UI
st.title("💓 Heart Disease Risk Predictor")
st.write("Provide the patient’s information below:")

with st.form("input_form"):
    # Continuous
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    physical_health = st.slider("Physical Health (days of bad health in past 30 days)", 0.0, 30.0, 0.0)
    mental_health = st.slider("Mental Health (days of bad health in past 30 days)", 0.0, 30.0, 0.0)
    sleep_time = st.slider("Sleep Time (hours per night)", 0.0, 24.0, 7.0)

    # Binary (Yes/No)
    smoking = binary_input("Do you smoke?")
    alcohol = binary_input("Do you drink alcohol?")
    stroke = binary_input("Have you ever had a stroke?")
    diff_walking = binary_input("Do you have difficulty walking?")
    physical_activity = binary_input("Are you physically active?")
    asthma = binary_input("Do you have asthma?")
    kidney = binary_input("Do you have kidney disease?")
    skin_cancer = binary_input("Do you have skin cancer?")

    # Categorical
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 1 if sex == "Male" else 0

    age_cat = st.selectbox("Age Category", list(age_order.keys()))
    age = age_order[age_cat]

    diabetic = st.selectbox("Diabetic?", ["No", "Yes", "During Pregnancy"])
    diabetic_encoded = {"No": 0, "Yes": 1, "During Pregnancy": 2}[diabetic]

    gen_health = st.selectbox("General Health", ["Poor", "Fair", "Good", "Very good", "Excellent"])
    gen_health_encoded = encode_ordinal(gen_health, ["Poor", "Fair", "Good", "Very good", "Excellent"])

    # One-hot encoded Race
    race = st.selectbox("Race", ["Asian", "Black", "Hispanic", "Other", "White"])
    race_encoded = [0, 0, 0, 0, 0]
    race_map = {"Asian": 0, "Black": 1, "Hispanic": 2, "Other": 3, "White": 4}
    race_encoded[race_map[race]] = 1

    submitted = st.form_submit_button("Predict")

if submitted:
    input_array = np.array([[ 
        bmi,
        int(smoking),
        int(alcohol),
        int(stroke),
        physical_health,
        mental_health,
        int(diff_walking),
        sex,
        age,
        diabetic_encoded,
        int(physical_activity),
        gen_health_encoded,
        sleep_time,
        int(asthma),
        int(kidney),
        int(skin_cancer),
        *race_encoded
    ]])

    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array)[0][1] * 100

    st.subheader("🩺 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ Patient is AT RISK of Heart Disease ({confidence:.2f}% confidence)")
    else:
        st.success(f"✅ Patient is NOT at Risk of Heart Disease ({100 - confidence:.2f}% confidence)")
