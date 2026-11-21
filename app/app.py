import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# ==========================================
# 2. LOAD MODEL
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "random_forest_class_weight.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()

model = bundle["model"]
threshold = bundle["threshold"]

# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🏭 Smart Industrial Predictive Maintenance")

st.caption(
    "AI-powered machine monitoring and failure intelligence system"
)

st.divider()

# ==========================================
# 4. MACHINE INPUT SECTION
# ==========================================

st.header("⚙️ Machine Operating Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=200.0,
        max_value=400.0,
        value=300.0
    )

with col2:
    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=200.0,
        max_value=450.0,
        value=310.0
    )

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=0,
        max_value=10000,
        value=1500
    )

with col3:
    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=200.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=1000,
        value=100
    )

temperature_difference = (
    process_temp - air_temp
)

# ==========================================
# 5. PREPARE MACHINE DATA
# ==========================================

input_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Temperature difference": temperature_difference
}])

# ==========================================
# 6. MACHINE STATUS DASHBOARD
# ==========================================

st.divider()

st.header("📊 Live Machine Monitoring")

st.caption(
    "Current parameter readings based on the values entered above."
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Machine Type", machine_type)
m2.metric("Air Temperature", f"{air_temp:.1f} K")
m3.metric("Process Temperature", f"{process_temp:.1f} K")
m4.metric("Temperature Difference", f"{temperature_difference:.1f} K")

m5, m6, m7 = st.columns(3)

m5.metric("Rotational Speed", f"{rotational_speed} rpm")
m6.metric("Torque", f"{torque:.1f} Nm")
m7.metric("Tool Wear", f"{tool_wear} min")

# ==========================================
# 7. FEATURE MONITORING DATA
# ==========================================

feature_data = pd.DataFrame({
    "Feature": [
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear",
        "Temperature Difference"
    ],
    "Current Value": [
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear,
        temperature_difference
    ],
    "Unit": [
        "K", "K", "rpm", "Nm", "min", "K"
    ]
})

# ==========================================
# 8. LINE GRAPH
# ==========================================

st.divider()

st.subheader("📈 Machine Parameter Line Graph")

line_fig = px.line(
    feature_data,
    x="Feature",
    y="Current Value",
    markers=True,
    hover_data=["Unit"],
    title="Current Machine Parameter Readings"
)

line_fig.update_layout(
    xaxis_title="Machine Features",
    yaxis_title="Current Reading (native units)",
    xaxis_tickangle=-30,
    height=450
)

st.plotly_chart(line_fig, use_container_width=True)

# ==========================================
# 9. BAR CHART
# ==========================================

st.subheader("📊 Machine Feature Bar Chart")

bar_fig = px.bar(
    feature_data,
    x="Feature",
    y="Current Value",
    color="Feature",
    hover_data=["Unit"],
    title="Current Values of Machine Features"
)

bar_fig.update_layout(
    xaxis_title="Machine Features",
    yaxis_title="Current Reading (native units)",
    xaxis_tickangle=-30,
    showlegend=False,
    height=450
)

st.plotly_chart(bar_fig, use_container_width=True)

st.caption(
    "Note: These charts show current readings, not historical trends. "
    "The features have different units and scales, so compare each "
    "reading against its own engineering reference range."
)

# ==========================================
# 10. CURRENT FEATURE STATUS TABLE
# ==========================================

st.subheader("🔍 Current Feature Status")

status_data = feature_data.copy()

status_data["Reading Status"] = "Current input captured"

st.dataframe(
    status_data,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 11. MACHINE FAILURE PREDICTION
# ==========================================

st.divider()

st.header("🤖 AI Machine Failure Prediction")

if st.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    if temperature_difference <= 0:
        st.error(
            "Process temperature must be higher than air temperature."
        )

    else:
        probabilities = model.predict_proba(input_data)

        failure_index = list(model.classes_).index(1)

        failure_probability = probabilities[
            0, failure_index
        ]

        prediction = int(
            failure_probability >= threshold
        )

        st.subheader("Prediction Results")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Failure Probability",
                f"{failure_probability:.2%}"
            )

        with c2:
            st.metric(
                "Model Threshold",
                f"{threshold:.2f}"
            )

        st.progress(float(failure_probability))

        if prediction == 1:
            st.error(
                "⚠️ Potential Machine Failure Detected"
            )
            st.warning(
                "Inspect the machine and investigate the operating conditions."
            )
        else:
            st.success(
                "✅ No Machine Failure Predicted"
            )

        st.caption(
            "This is a model-based prediction, not a confirmed "
            "physical machine condition."
        )

# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Smart Industrial Predictive Maintenance | "
    "Random Forest Class Weight | AI4I 2020 Dataset"
)