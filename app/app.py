import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_class_weight.joblib"
)

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
threshold = bundle["threshold"]

# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🏭 Smart Industrial Predictive Maintenance")
st.subheader("Machine Failure Prediction System")

st.write(
    "Enter machine operating parameters "
    "to estimate the probability of machine failure."
)

st.divider()

# --------------------------------
# MACHINE INPUTS
# --------------------------------

st.header("Machine Operating Parameters")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        value=310.0
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        value=1500
    )

    torque = st.number_input(
        "Torque [Nm]",
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        value=100
    )

temperature_difference = (
    process_temp - air_temp
)

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("Predict Machine Failure", type="primary"):

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature difference": temperature_difference
    }])

    probabilities = model.predict_proba(input_data)

    failure_index = list(model.classes_).index(1)

    failure_probability = probabilities[0, failure_index]

    prediction = int(failure_probability >= threshold)

    st.divider()
    st.header("Prediction Results")

    st.metric(
        "Failure Probability",
        f"{failure_probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠️ Potential Machine Failure Detected")
    else:
        st.success("✅ No Failure Predicted")

    st.write(
        f"Classification threshold: {threshold:.2f}"
    )

    st.progress(float(failure_probability))