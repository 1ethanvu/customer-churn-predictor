import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# 2. Load Model Artifacts
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "churn_model_pipeline.joblib")
    if not os.path.exists(model_path):
        st.error("Model file not found! Please run `train_model.py` first to generate model artifacts.")
        st.stop()
    return joblib.load(model_path)

artifacts = load_artifacts()
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
feature_names = artifacts["feature_names"]

# 3. Header Section
st.title("🔮 Real-Time Customer Churn Risk Predictor")
st.markdown("""
This interactive tool uses an **XGBoost machine learning model** to assess subscriber churn risk 
and leverages **SHAP (SHapley Additive exPlanations)** to explain key driving factors behind individual predictions.
""")

st.sidebar.header("Customer Profile Settings")

# 4. User Inputs via Sidebar
def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = tenure * monthly_charges  # Sensible estimate

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# 5. Prediction Pipeline Execution
processed_input = preprocessor.transform(input_df)
churn_probability = model.predict_proba(processed_input)[0][1]
churn_prediction = int(churn_probability >= 0.5)

# 6. Main Dashboard Interface Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Score Summary")
    
    # Display probability metric
    risk_percentage = round(churn_probability * 100, 1)
    
    if churn_probability >= 0.7:
        st.error(f"High Risk of Churn: {risk_percentage}%")
    elif churn_probability >= 0.4:
        st.warning(f"Moderate Risk of Churn: {risk_percentage}%")
    else:
        st.success(f"Low Risk (Retained): {risk_percentage}%")

    st.metric(label="Predicted Churn Probability", value=f"{risk_percentage}%")

    # Actionable Business Insight Text
    st.markdown("### Suggested Action Plan")
    if churn_probability >= 0.5:
        st.markdown("""
        * **Offer Contract Incentive:** Switch from Month-to-Month to a 1-year contract discount.
        * **Tech Support Follow-up:** Reach out if technical support onboarding was missed.
        * **Billing Adjustment:** Provide alternative payment options to electronic checks.
        """)
    else:
        st.markdown("""
        * **Cross-sell:** Recommend additional value-add services (e.g., streaming or backup features).
        * **Loyalty Perks:** Standard renewal communications apply.
        """)

with col2:
    st.subheader("Model Decision Explanation (SHAP Value Force Plot)")
    st.write(
        "Features pushing the prediction **higher (towards Churn)** appear in red, "
        "while features pushing it **lower (towards Retention)** appear in blue."
    )

    # Calculate SHAP Explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(processed_input)

    # Format SHAP visual output using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=processed_input[0],
            feature_names=feature_names
        ),
        max_display=8,
        show=False
    )
    st.pyplot(fig)

# 7. Customer Raw Feature Display
st.divider()
st.subheader("Current Customer Input Summary")
st.dataframe(input_df)