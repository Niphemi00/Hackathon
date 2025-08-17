import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Load model and scaler ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model/best_model.joblib"
SCALER_PATH = BASE_DIR / "model/scaler.joblib"


model, scaler = None, None
if MODEL_PATH.exists() and SCALER_PATH.exists():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.error("Model files not found. Please ensure best_model.joblib and scaler.joblib are in /model folder.")

st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓")

st.title("🎓 Student Dropout Risk Predictor")
st.write("Enter student information to predict dropout risk.")

# --- Input Form ---
with st.form("student_form"):
    age = st.number_input("Age", min_value=15, max_value=25, value=16)
    sex = st.selectbox("Sex", ["F", "M"])
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], index=1)
    failures = st.number_input("Past Failures", min_value=0, max_value=5, value=0)
    absences = st.number_input("Absences", min_value=0, max_value=75, value=5)
    G1 = st.slider("Grade Period 1 (0-20)", 0, 20, 12)
    G2 = st.slider("Grade Period 2 (0-20)", 0, 20, 12)
    G3 = st.slider("Final Grade (0-20)", 0, 20, 12)
    submit = st.form_submit_button("Predict")

if submit:
  if model is None or scaler is None:
    st.warning("Model is not available, cannot make predictions.")
  else:
    # Construct input dataframe
    input_dict = {
        "age": [age],
        "sex": [sex],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "G1": [G1],
        "G2": [G2],
        "G3": [G3],
    }
    input_df = pd.DataFrame(input_dict)

    # Feature engineering
    input_df["avg_grade"] = input_df[["G1","G2","G3"]].mean(axis=1)
    input_df["grade_trend"] = input_df["G3"] - input_df["G1"]
    input_df["passed_final"] = (input_df["G3"] >= 10).astype(int)

    # Encoding: match training columns
    # NOTE: In hackathon you may want to store the full column set
    # For now, we keep it simple (sex → binary)
    input_df = pd.get_dummies(input_df, columns=["sex"], drop_first=True)

    # Scale numeric values
    input_scaled = scaler.transform(input_df)

    # Prediction
    prob = model.predict_proba(input_scaled)[:,1][0]
    prediction = "🚨 At Risk" if prob >= 0.5 else "✅ Not at Risk"

    st.subheader(f"Prediction: {prediction}")
    st.write(f"Risk Probability: **{prob:.2f}**")
