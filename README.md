🏭 Smart Industrial Predictive Maintenance & Failure Intelligence System

An End-to-End Machine Learning Project for Predicting Industrial Machine Failures

Project Type: Machine Learning | Predictive Maintenance | Artificial Intelligence | Explainable AI | Streamlit

Primary Model: Random Forest Classifier – Class Weight

Dataset: AI4I 2020 Predictive Maintenance Dataset

---

1. Project Overview

Industrial machines are essential components of manufacturing industries. Unexpected machine failures can result in production interruptions, equipment damage, increased maintenance costs, and safety risks.

Traditional maintenance approaches generally follow three strategies:

- Reactive Maintenance: Repairing machines after a failure occurs.
- Preventive Maintenance: Performing maintenance at scheduled intervals.
- Predictive Maintenance: Using machine data and machine learning to predict potential failures before they happen.

This project focuses on developing a Smart Industrial Predictive Maintenance and Failure Intelligence System using Machine Learning.

The system analyses machine operating parameters, identifies patterns associated with failures, and predicts whether a machine is likely to experience a failure.

The main objective is to detect as many potential machine failures as possible by prioritising Recall during model selection and threshold tuning.

The project also includes an interactive Streamlit dashboard with SHAP-based Explainable AI, allowing users to enter machine parameters, view failure predictions, understand the features influencing predictions, and visualise machine operating conditions.

---

2. Project Objectives

The major objectives are:

1. Analyse industrial machine operating data.
2. Perform data cleaning and preprocessing.
3. Explore relationships between machine parameters and failures.
4. Handle the class imbalance problem.
5. Train and compare multiple machine learning algorithms.
6. Select the best-performing model based on the project's failure-detection objective.
7. Prioritise failure detection using Recall.
8. Perform classification threshold tuning.
9. Explain model predictions using SHAP.
10. Identify features that increase or decrease the model's failure prediction.
11. Save the trained model for deployment.
12. Develop an interactive Streamlit dashboard.
13. Visualise machine operating parameters and SHAP feature contributions.
14. Demonstrate how the model could be integrated into a real industrial predictive maintenance system.

---

3. Dataset Information

AI4I 2020 Predictive Maintenance Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains simulated industrial machine operating data designed for predictive maintenance research.

It includes machine operating conditions, sensor-related measurements, and failure information.

Dataset characteristics:

- Approximately 10,000 records.
- Seven selected input features after feature engineering.
- Binary target variable: Machine failure.
- Multiple failure modes are available in the original dataset.

Dataset source:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Selected Input Features

Feature| Description| Unit
Type| Machine quality/type category| Categorical
Air temperature| Surrounding air temperature| K
Process temperature| Machine process temperature| K
Rotational speed| Machine rotational speed| rpm
Torque| Operating torque| Nm
Tool wear| Tool usage/wear measurement| min
Temperature difference| Process temperature minus air temperature| K

Target Variable

"Machine failure"

- "0" – No machine failure
- "1" – Machine failure

Important Feature Engineering

A new feature called Temperature difference was created:

"Temperature Difference = Process Temperature − Air Temperature"

This feature helps represent the difference between the surrounding temperature and the machine's process temperature.

Data Leakage Prevention

The original dataset contains individual failure-mode columns:

- TWF
- HDF
- PWF
- OSF
- RNF

These columns were excluded from the input features because they contain failure-related information that could reveal the target.

Using them to predict Machine failure would cause target leakage and produce misleading evaluation results.

---

4. Project Workflow – Notebook by Notebook

The project was developed through separate Jupyter notebooks to maintain an organised machine learning workflow.

Notebook 01 – Data Exploration

File: "01_data_exploration.ipynb"

Purpose: Understand the dataset before building machine learning models.

Activities performed:

- Imported the dataset using Pandas.
- Examined dataset dimensions.
- Inspected column names and data types.
- Viewed initial records.
- Analysed numerical and categorical features.
- Checked missing values.
- Checked duplicate records.
- Examined the target variable distribution.
- Visualised machine failure patterns.
- Used exploratory data analysis to understand the dataset.

Libraries: Pandas, NumPy, Matplotlib and Seaborn.

Outcome: Understood the dataset structure, input features, target variable, and class imbalance.

Notebook 02 – Data Preprocessing

File: "02_data_preprocessing.ipynb"

Purpose: Prepare clean and suitable data for machine learning.

Activities performed:

- Selected relevant input features.
- Separated independent variables (X) and target variable (y).
- Removed failure-mode columns to prevent target leakage.
- Created the Temperature Difference feature.
- Applied categorical encoding to the Type feature.
- Applied numerical feature scaling where required by the preprocessing workflow.
- Split the data into training and testing sets.
- Saved the processed training and testing datasets.

Preprocessing techniques:

- One-Hot Encoding
- StandardScaler
- Train-Test Split

Outcome: Prepared training and testing datasets containing 8,000 training records and 2,000 testing records.

Processed files:

- "X_train.csv"
- "X_test.csv"
- "y_train.csv"
- "y_test.csv"

Notebook 03 – Baseline Model Training

File: "03_baseline_model_training.ipynb"

Purpose: Build initial machine learning models and establish baseline performance.

Algorithms explored:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest

Activities performed:

- Imported machine learning algorithms.
- Trained baseline models.
- Generated predictions.
- Evaluated classification performance.
- Compared model behaviour.
- Examined the impact of class imbalance.

Evaluation metrics considered:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

Outcome: Established baseline model performance and identified the need for better handling of imbalanced failure data.

Notebook 04 – Imbalance Handling

File: "04_imbalance_handling.ipynb"

Purpose: Address the imbalance between normal machine observations and machine failure observations.

Industrial failure datasets often contain considerably fewer failure records than normal operating records.

A model trained directly on imbalanced data may achieve high accuracy while failing to identify a significant proportion of actual failures.

Two approaches were investigated.

1. Class Weight

Used:

"class_weight="balanced""

This technique adjusts the importance of different classes during model training.

It gives greater relative importance to the minority failure class based on class frequencies.

2. SMOTENC

Used Synthetic Minority Over-sampling Technique for Nominal and Continuous features.

SMOTENC generates synthetic minority-class samples while accounting for categorical features.

It was used to address the class imbalance problem while preserving the categorical nature of the machine Type feature.

Outcome: Developed alternative Random Forest models using class weighting and SMOTENC for comparison.

Notebook 05 – Model Evaluation and Threshold Tuning

File: "05_model_evaluation.ipynb"

Purpose: Evaluate candidate models and select a model suitable for failure detection.

This was one of the most important stages of the project.

Why Recall?

In industrial predictive maintenance, failing to detect a machine failure can be more costly than generating an unnecessary maintenance warning.

Therefore, this project prioritises Recall for the machine failure class.

Recall measures the proportion of actual failures correctly identified by the model.

"Recall = True Positives / (True Positives + False Negatives)"

A higher recall means fewer actual machine failures are missed.

Model Comparison

The two main Random Forest candidates were:

1. Random Forest with Class Weight.
2. Random Forest with SMOTENC.

Both were evaluated using classification metrics.

The evaluation considered:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The Class Weight Random Forest was selected as the preferred candidate based on the overall evaluation and recall-focused project objective.

Threshold Tuning

A standard binary classification model commonly uses a threshold of 0.50.

However, a lower threshold can identify more potential failures.

This project investigated different classification thresholds using cross-validation.

The selected threshold was:

"0.13"

Cross-validation threshold results:

Model| Selected Threshold| CV Recall
Random Forest – Class Weight| 0.13| 90.04%
Random Forest – SMOTENC| 0.22| 90.04%

The Class Weight model was selected.

Why 0.13?

The objective was to achieve approximately 90% recall for the failure class.

At the selected threshold, a machine is classified as a potential failure when its predicted failure probability is greater than or equal to 0.13.

This increases sensitivity to potential failures, although it can also increase false alarms.

Important: The threshold was selected using cross-validation rather than by repeatedly adjusting it against the final test set.

Outcome: Selected Random Forest with Class Weight and saved the trained model and threshold for further use.

Notebook 06 – Model Explainability

File: "06_model_explainability.ipynb"

Purpose: Understand how the trained machine learning model makes predictions.

A machine learning model should not only predict a result; it should also provide insights into the factors influencing that result.

This project uses SHAP (SHapley Additive exPlanations).

Activities performed:

- Loaded the saved Random Forest model.
- Loaded the processed training data.
- Extracted the preprocessing pipeline.
- Extracted the Random Forest classifier.
- Transformed the training data.
- Retrieved transformed feature names.
- Selected a sample of 500 observations.
- Created a SHAP TreeExplainer.
- Calculated SHAP values.
- Extracted SHAP values for the positive class (Machine Failure).

The transformed dataset contains nine features because the categorical Type feature is converted into three one-hot encoded features:

- Type_H
- Type_L
- Type_M

The remaining six features are numerical.

SHAP analysis helps identify:

- Which features have the greatest influence on model predictions.
- Whether higher or lower feature values push predictions towards failure.
- Which operating parameters influence individual predictions.

SHAP explains the model's learned behaviour. It does not establish physical causation.

---

5. Final Model Selection

Selected Model: Random Forest Classifier – Class Weight

The selected model uses:

- Random Forest Classifier
- 100 estimators
- Balanced class weights
- Random state 42
- Preprocessing pipeline with One-Hot Encoding and StandardScaler
- Classification threshold of 0.13

Reported Evaluation Results

Metric| Result
Accuracy| 94.3%
Precision| 36.63%
Recall| 92.65%
F1-score| 52.5%
ROC-AUC| Approximately 0.971

These are the recorded model evaluation figures from the project evaluation. The separate cross-validation threshold selection produced 90.04% recall at threshold 0.13.

The two results should not be treated as the same evaluation: the cross-validation result is used for threshold selection, while the recorded model evaluation metrics describe the evaluated model results.

Why Random Forest?

Random Forest was selected because:

1. It can model nonlinear relationships.
2. It combines multiple decision trees.
3. It works with numerical and encoded categorical features.
4. Class weighting helps address class imbalance.
5. It provides probability estimates for classification.
6. It supports SHAP-based model explainability.
7. It achieved strong failure-class recall in the project evaluation.

Model Limitations

The model has relatively low precision compared with its recall.

This means it can generate false-positive failure predictions.

In a real industrial environment, maintenance teams would need to investigate warnings rather than treating every warning as a confirmed failure.

---

6. Model Saving and Deployment Preparation

The selected model was saved using Joblib.

Model file:

"models/random_forest_class_weight.joblib"

The saved model bundle contains:

- Trained preprocessing and Random Forest pipeline.
- Selected classification threshold.
- Model name.
- Positive class information.

The saved bundle allows the Streamlit application to load the same trained model without retraining it every time.

The deployment verification process checks:

- Whether the model file exists.
- Whether the model loads successfully.
- Whether the threshold is 0.13.
- Whether the model supports probability prediction.
- Whether the model can accept new machine input data.

The prediction pipeline uses "predict_proba()" to obtain the probability of machine failure and applies the selected threshold.

---

7. Streamlit Dashboard – Prediction and Explainable AI

The project includes an interactive dashboard developed using Streamlit.

Application file:

"app/app.py"

The dashboard allows users to enter machine operating parameters and receive machine failure predictions.

The application uses the saved Random Forest model, preprocessing pipeline, classification threshold, and SHAP explainer.

7.1 Machine Input Section

Users can enter:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Temperature Difference is calculated automatically:

"Temperature Difference = Process Temperature − Air Temperature"

7.2 Machine Parameter Monitoring

The dashboard displays the numerical operating parameters entered by the user.

A machine readings table shows:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Temperature Difference

These readings help users inspect the current machine input.

7.3 Machine Feature Line Plot

The application includes a line plot showing the current numerical readings for the machine operating parameters.

It provides a visual representation of the values entered by the user.

Important: The features have different measurement units and scales. The line plot is intended for visual inspection of the readings, not for direct comparison of their magnitudes.

It does not represent historical machine readings or time-series sensor data.

7.4 Machine Feature Bar Chart

A feature reading table and bar-chart visualisation help users inspect the machine's current operating parameters.

The charts are based on the values entered in the application.

They are not engineering-limit or machine-health classifications.

7.5 Failure Prediction

The trained Random Forest model analyses the input data and predicts:

- No Machine Failure
- Potential Machine Failure Detected

The application calculates the probability of the positive class, where class 1 represents Machine Failure.

7.6 Failure Probability

The dashboard displays the model's predicted failure probability.

For example, a model probability of 0.78 is displayed as 78%.

This is a model estimate, not a guarantee that the machine will fail.

7.7 Threshold-Based Classification

The application uses the saved threshold of 0.13.

The classification logic is:

"Prediction = 1 if Failure Probability >= 0.13"

Otherwise:

"Prediction = 0"

This threshold supports the project's recall-focused failure-detection objective.

7.8 SHAP-Based Explainable AI

The application integrates SHAP into the Streamlit prediction workflow.

Whenever the user clicks Predict Machine Failure, the application calculates SHAP values for the current machine input.

The SHAP explainer uses the trained Random Forest classifier extracted from the saved preprocessing pipeline.

The application explains the positive class:

"Class 1 = Machine Failure"

This makes it possible to inspect the features that influenced the individual prediction.

7.9 Top Five Influencing Features

The dashboard displays the five features with the largest absolute SHAP contributions for the current prediction.

Each feature is displayed with:

- Feature name.
- SHAP contribution value.
- Direction of influence.

Interpretation:

- Positive SHAP value: Pushes the model output towards machine failure.
- Negative SHAP value: Pushes the model output away from machine failure.
- Near-zero SHAP value: Has little influence on this individual prediction.

The displayed feature ranking is based on absolute SHAP contribution, not a universal ranking of feature importance.

7.10 SHAP Contribution Bar Chart

The application includes a horizontal SHAP contribution bar chart.

The chart visually represents the strongest feature contributions.

Colour interpretation:

- Red: Positive SHAP contribution, pushing the model output towards failure.
- Green: Negative SHAP contribution, pushing the model output away from failure.

The zero reference line separates positive and negative contributions.

This chart helps users quickly understand which features influenced the model's current decision.

7.11 SHAP Waterfall Plot

The dashboard also includes an individual SHAP waterfall plot.

The waterfall plot explains how individual feature contributions move the model output from its baseline value towards the output for the current machine input.

It helps users understand how multiple operating parameters contribute to the prediction.

Important: The SHAP waterfall plot explains the Random Forest's model output scale. It should not be interpreted as a direct percentage-point breakdown of the displayed failure probability.

7.12 Final Prediction Interpretation

After displaying the SHAP results, the application provides a short interpretation of the prediction.

For a potential failure prediction, the user can review the strongest contributing features.

For a no-failure prediction, the user can still inspect which features pushed the model output towards or away from failure.

Dashboard Feature Summary

Dashboard Feature| Purpose
Machine Input Section| Enter operating parameters
Temperature Difference| Automatically calculate a derived feature
Machine Readings Table| Display current input values
Feature Line Plot| Visualise current readings
Feature Reading Bar Chart| Inspect machine parameter values
Failure Prediction| Predict potential machine failure
Failure Probability| Display estimated failure probability
Threshold Classification| Apply the selected 0.13 threshold
SHAP Contribution Table| Display top five influencing features
SHAP Bar Chart| Visualise positive and negative contributions
SHAP Waterfall Plot| Explain an individual prediction
Prediction Interpretation| Help users understand the model output

The dashboard provides a working prototype for manually entered machine parameters, failure prediction, and model explainability.

Important limitations:

- The application does not currently receive live sensor data.
- The line plot displays current readings, not historical trends.
- The charts do not independently establish whether a machine parameter is physically abnormal.
- SHAP explains model behaviour, not the physical cause of a machine failure.
- Automatic maintenance-team notifications are not implemented.

---

8. How the System Works – End-to-End Architecture

The current prototype works as follows:

1. User enters machine operating parameters.
2. Streamlit collects the input data.
3. The preprocessing pipeline transforms the input features.
4. The trained Random Forest model analyses the data.
5. The model calculates the probability of machine failure.
6. The classification threshold of 0.13 is applied.
7. The dashboard displays the predicted result and probability.
8. SHAP calculates feature contributions for the current input.
9. The dashboard displays the top contributing features and visual explanations.

Current Prototype Architecture

Machine Parameter Input
          |
          v
Streamlit Dashboard
          |
          v
Data Preparation and Preprocessing
          |
          v
Trained Random Forest Model
          |
          v
Failure Probability Prediction
          |
          v
Threshold-Based Classification
          |
          v
Prediction Results
          |
          v
SHAP Explainability
          |
          v
Feature Contribution Table
          |
          v
SHAP Bar Chart and Waterfall Plot
          |
          v
Machine Parameter Visualisations

---

9. Real-World Industrial Implementation

Important: The current project is a software prototype using a simulated predictive-maintenance dataset. It is not yet connected to actual industrial machines or live sensors.

However, the same machine learning concept could be extended into a real industrial predictive maintenance system.

How It Could Work in a Real Factory

Imagine a manufacturing plant containing several industrial machines.

Each machine can be equipped with suitable sensors to continuously monitor operating conditions.

Possible sensors include:

- Temperature Sensors
- RPM Sensors
- Torque Sensors
- Vibration Sensors
- Pressure Sensors
- Tool Wear Monitoring Systems

These sensors collect machine operating information.

Real-World System Architecture

Industrial Machines
          |
          v
Sensors Installed on Machines
          |
          v
IoT Gateway / Edge Device
          |
          v
Real-Time Data Collection
          |
          v
Data Validation and Preprocessing
          |
          v
Trained Machine Learning Model
          |
          v
Failure Probability Calculation
          |
          v
SHAP-Based Prediction Explanation
          |
          v
Threshold-Based Alert Generation
          |
          v
Maintenance Monitoring Dashboard
          |
          v
Maintenance Team Notification

Example Real-World Scenario

Consider a machine operating in a manufacturing plant.

The machine sensors continuously collect:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The sensor readings are transmitted to an edge device or central monitoring system.

The data is processed and passed to the trained machine learning model.

If the model predicts a potential machine failure and the probability crosses the configured threshold, the system can automatically generate an alert.

For example:

Machine ID: MACHINE-001

Predicted Status: Potential Machine Failure

Failure Probability: 78%

Alert: Immediate inspection recommended.

The system could then send notifications to the maintenance team through:

- Email
- SMS
- Mobile application notifications
- Industrial monitoring software
- Computerised Maintenance Management Systems (CMMS)

The maintenance team can inspect the machine before a possible failure causes an unexpected production interruption.

The alert percentage in this example is illustrative, not a measured result from the current prototype.

Automatic Maintenance Alerts

In a future production implementation, the system could be integrated with:

- MQTT for IoT communication.
- Apache Kafka for streaming data.
- Edge computing devices for local inference.
- Cloud infrastructure for data storage.
- Notification services for email and SMS.
- CMMS platforms for maintenance scheduling.

These integrations are proposed future capabilities, not currently implemented project features.

---

10. Technology Stack

Technology| Purpose
Python| Main programming language
Pandas| Data manipulation
NumPy| Numerical computation
Matplotlib| Data visualisation and SHAP charts
Seaborn| Exploratory data analysis
Scikit-learn| Machine learning
Imbalanced-learn| Class imbalance handling
SHAP| Model explainability
Joblib| Model saving and loading
Streamlit| Interactive dashboard
Plotly| Interactive charts
Jupyter Notebook| Model development and experimentation
Git| Version control
GitHub| Project hosting

---

11. Project Folder Structure

smart-industrial-predictive-maintenance/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   └── random_forest_class_weight.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_model_training.ipynb
│   ├── 04_imbalance_handling.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_model_explainability.ipynb
│
├── reports/
│   ├── final_model_evaluation.csv
│   ├── threshold_cv_results.csv
│   ├── shap_summary.png
│   └── shap_feature_importance.png
│
├── .gitignore
├── README.md
└── requirements.txt

Some report images and files are generated during the workflow. Their presence depends on whether the corresponding outputs have been saved.

The SHAP images in the reports directory are associated with the notebook analysis. The individual SHAP waterfall and contribution charts are also generated dynamically in the Streamlit application.

---

12. Project Requirements

Software Requirements

- Python 3.10 or a compatible version.
- Visual Studio Code or Google Antigravity.
- Jupyter Notebook.
- Git.
- Web browser.

Python Libraries

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
shap
joblib
streamlit
plotly
jupyter

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate to the project directory:

cd smart-industrial-predictive-maintenance

Create a virtual environment:

python -m venv .venv

Activate the environment.

Windows PowerShell:

.venv\Scripts\Activate.ps1

Git Bash:

source .venv/Scripts/activate

Install the dependencies:

pip install -r requirements.txt

Run the Streamlit Application

From the main project directory:

streamlit run app/app.py

The dashboard will open in your browser.

---

13. How to Use the Application

1. Start the Streamlit application.
2. Open the dashboard in your browser.
3. Select the machine type.
4. Enter the air temperature.
5. Enter the process temperature.
6. Enter the rotational speed.
7. Enter the torque.
8. Enter the tool wear.
9. Review the machine parameter readings and visualisations.
10. Click the Predict Machine Failure button.
11. Review the predicted status and failure probability.
12. Examine the top five SHAP feature contributions.
13. Review the SHAP contribution bar chart.
14. Examine the SHAP waterfall plot.
15. Understand which features pushed the model output towards or away from failure.

The application uses the saved Random Forest model, preprocessing pipeline, and selected threshold.

---

14. Limitations of the Current Project

The project is a prototype and has several limitations:

1. The dataset is simulated and does not represent live measurements from an actual manufacturing plant.
2. The application currently accepts manually entered machine parameters.
3. Real-time sensor integration has not been implemented.
4. Automatic email, SMS, or maintenance-team alerts have not been implemented.
5. Historical machine data storage is not currently implemented.
6. The model may produce false-positive predictions.
7. The selected threshold requires validation using real industrial operating data before production use.
8. The model should be monitored and retrained if real-world machine behaviour changes.
9. SHAP explains the model's learned relationships, not the physical causes of equipment failure.
10. The current feature visualisations display individual input readings rather than historical machine trends.

These limitations provide opportunities for future development.

---

15. Future Enhancements

Possible future improvements include:

- Integration with real industrial sensors.
- Real-time machine data collection.
- MQTT-based IoT communication.
- Historical machine monitoring.
- Multiple machine monitoring.
- Machine health scoring.
- Automated maintenance notifications.
- Email and SMS alert integration.
- CMMS integration.
- Cloud deployment.
- Model performance monitoring.
- Periodic model retraining using real operational data.
- Validation of SHAP explanations with industrial domain experts.

---

16. Expected Real-World Benefits

If properly validated and deployed in an industrial environment, a predictive maintenance system could help:

- Reduce unexpected machine downtime.
- Improve maintenance planning.
- Identify potential machine failures earlier.
- Reduce unnecessary maintenance activities.
- Improve equipment availability.
- Support data-driven maintenance decisions.
- Improve operational efficiency.

These are potential benefits of a properly implemented production system, not measured outcomes of this prototype.

---

17. Conclusion

The Smart Industrial Predictive Maintenance and Failure Intelligence System demonstrates how Machine Learning can be used to analyse industrial operating conditions and predict potential machine failures.

The project covers the complete machine learning development lifecycle, including data exploration, preprocessing, model training, class imbalance handling, model evaluation, threshold tuning, explainable AI, model saving, and application development.

A Random Forest Classifier with Class Weight was selected as the preferred model, with failure detection prioritised through recall-focused threshold tuning.

The Streamlit application demonstrates how machine operating parameters can be entered, visualised, and evaluated by a trained machine learning model.

The integration of SHAP-based Explainable AI provides additional transparency by identifying the features influencing individual machine failure predictions.

Although the current implementation is a prototype based on simulated data and manual inputs, it demonstrates the foundation for a future real-world industrial predictive maintenance system in which sensors continuously collect machine data, an AI model analyses the readings, and potential failure alerts are automatically delivered to the maintenance team.

The ultimate vision is simple:

Move from repairing machines after they fail to identifying potential failures before they interrupt production.

---

18. Author

Bilish B L

Data Science & AI Enthusiast

GitHub: https://github.com/bilishbl2004-commits

LinkedIn: https://linkedin.com/in/bilish-b-l

Project: Smart Industrial Predictive Maintenance & Failure Intelligence System

Domain: Artificial Intelligence | Machine Learning | Explainable AI | Industrial IoT | Predictive Maintenance

---

19. Disclaimer

This project is developed for educational, research, and demonstration purposes.

The predictions generated by the model are not guaranteed indicators of actual machine failure. The system should not be used as the sole basis for industrial safety decisions or maintenance actions without proper validation, testing, and integration with real industrial equipment.