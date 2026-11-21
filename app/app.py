import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

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
# LOAD TRAINED MODEL AND SHAP
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_class_weight.joblib"
)


@st.cache_resource
def load_model_and_explainer(model_path):

    bundle = joblib.load(model_path)

    trained_model = bundle["model"]
    trained_threshold = bundle["threshold"]

    # Extract preprocessing and Random Forest
    trained_preprocessor = (
        trained_model.named_steps["preprocessor"]
    )

    trained_classifier = (
        trained_model.named_steps["model"]
    )

    # Create SHAP explainer
    trained_explainer = shap.TreeExplainer(
        trained_classifier
    )

    return (
        trained_model,
        trained_threshold,
        trained_preprocessor,
        trained_classifier,
        trained_explainer
    )


(
    model,
    threshold,
    preprocessor,
    classifier,
    explainer
) = load_model_and_explainer(str(MODEL_PATH))


# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🏭 Smart Industrial Predictive Maintenance")

st.subheader(
    "Machine Failure Prediction & Explainable AI System"
)

st.write(
    "Enter machine operating parameters to estimate "
    "the probability of machine failure and understand "
    "which features influence the prediction."
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

if st.button(
    "Predict Machine Failure",
    type="primary"
):

    if process_temp <= air_temp:

        st.error(
            "Process temperature must be greater "
            "than air temperature."
        )

    else:

        # ----------------------------
        # PREPARE INPUT DATA
        # ----------------------------

        input_data = pd.DataFrame([{
            "Type": machine_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Temperature difference": temperature_difference
        }])


        # ----------------------------
        # MODEL PREDICTION
        # ----------------------------

        probabilities = model.predict_proba(
            input_data
        )

        failure_index = list(
            classifier.classes_
        ).index(1)

        failure_probability = probabilities[
            0, failure_index
        ]

        prediction = int(
            failure_probability >= threshold
        )


        # ----------------------------
        # DISPLAY PREDICTION
        # ----------------------------

        st.divider()

        st.header("Prediction Results")

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                "Failure Probability",
                f"{failure_probability:.2%}"
            )

        with metric_col2:

            st.metric(
                "Classification Threshold",
                f"{threshold:.2f}"
            )


        if prediction == 1:

            st.error(
                "⚠️ Potential Machine Failure Detected"
            )

        else:

            st.success(
                "✅ No Failure Predicted"
            )


        st.progress(
            float(failure_probability)
        )


        # ====================================
        # SHAP EXPLAINABILITY
        # ====================================

        st.divider()

        st.header(
            "🔍 Explainable AI - SHAP Analysis"
        )

        st.write(
            "SHAP explains how individual machine "
            "operating parameters influence the "
            "Random Forest prediction."
        )


        # ----------------------------
        # TRANSFORM INPUT DATA
        # ----------------------------

        transformed_input = preprocessor.transform(
            input_data
        )

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        shap_input = pd.DataFrame(
            transformed_input,
            columns=feature_names
        )


        # ----------------------------
        # CALCULATE SHAP VALUES
        # ----------------------------

        raw_shap_values = explainer.shap_values(
            shap_input
        )


        # Extract SHAP values for class 1
        # Class 1 = Machine Failure

        if isinstance(raw_shap_values, list):

            failure_shap_values = (
                raw_shap_values[1][0]
            )

        else:

            raw_shap_array = np.asarray(
                raw_shap_values
            )

            if raw_shap_array.ndim == 3:

                failure_shap_values = (
                    raw_shap_array[0, :, 1]
                )

            elif raw_shap_array.ndim == 2:

                failure_shap_values = (
                    raw_shap_array[0]
                )

            else:

                st.error(
                    "Unexpected SHAP output shape. "
                    "Please check your SHAP version."
                )

                st.stop()


        # ----------------------------
        # GET BASE VALUE FOR CLASS 1
        # ----------------------------

        expected_value = explainer.expected_value

        if isinstance(expected_value, (list, np.ndarray)):

            expected_array = np.asarray(
                expected_value
            )

            if expected_array.ndim > 0:

                base_value = float(
                    expected_array.flatten()[1]
                )

            else:

                base_value = float(expected_array)

        else:

            base_value = float(expected_value)


        # ----------------------------
        # CREATE SHAP EXPLANATION
        # ----------------------------

        shap_explanation = shap.Explanation(
            values=failure_shap_values,
            base_values=base_value,
            data=shap_input.iloc[0].values,
            feature_names=feature_names
        )


        # ====================================
        # FEATURE CONTRIBUTION TABLE
        # ====================================

        st.subheader(
            "Top Features Influencing Prediction"
        )

        shap_results = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": failure_shap_values

        })


        # Feature importance
        shap_results["Importance"] = (
            shap_results["SHAP Value"].abs()
        )


        # Sort by contribution magnitude
        shap_results = shap_results.sort_values(
            by="Importance",
            ascending=False
        ).reset_index(drop=True)


        # Display top five features
        top_features = shap_results.head(5).copy()


        # Make feature names readable
        top_features["Feature"] = (
            top_features["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
        )


        # Explain direction
        top_features["Influence"] = np.where(
            top_features["SHAP Value"] > 0,
            "Increases failure output",
            np.where(
                top_features["SHAP Value"] < 0,
                "Decreases failure output",
                "No meaningful contribution"
            )
        )


        # Display table
        display_table = top_features[
            [
                "Feature",
                "SHAP Value",
                "Influence"
            ]
        ].copy()


        display_table["SHAP Value"] = (
            display_table["SHAP Value"].round(5)
        )


        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------
        # INTERPRETATION
        # ----------------------------

        st.info(
            "Positive SHAP values push the model output "
            "towards machine failure. Negative SHAP "
            "values push it away from machine failure. "
            "Their magnitude indicates the strength "
            "of their contribution."
        )


        # ====================================
        # SHAP WATERFALL VISUALIZATION
        # ====================================

        st.subheader(
            "SHAP Waterfall Plot"
        )

        st.write(
            "This chart explains how each feature "
            "moves the model output from its baseline "
            "value towards the final prediction."
        )


        try:

            fig = plt.figure(figsize=(10, 6))

            shap.plots.waterfall(
                shap_explanation,
                max_display=9,
                show=False
            )

            st.pyplot(
                plt.gcf(),
                clear_figure=True
            )

            plt.close("all")

        except Exception as error:

            plt.close("all")

            st.warning(
                f"Could not generate SHAP waterfall plot: {error}"
            )


        # ====================================
        # FINAL EXPLANATION
        # ====================================

        st.divider()

        st.subheader(
            "Understanding the Prediction"
        )

        if prediction == 1:

            st.write(
                "The model has classified this machine "
                "input as a potential failure case. "
                "Review the SHAP contributions to "
                "identify which operating parameters "
                "most influenced this prediction."
            )

        else:

            st.write(
                "The model has not classified this "
                "machine input as a failure case. "
                "SHAP contributions still show which "
                "features increased or decreased "
                "the model's failure output."
            )


        st.caption(
            "Note: SHAP explains the model's prediction. "
            "It does not establish physical causation. "
            "This prototype currently uses manually "
            "entered machine parameters, not live sensor data."
        )