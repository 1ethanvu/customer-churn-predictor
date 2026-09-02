# 🔮 Real-Time Customer Churn Risk Predictor & Explainability App

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/XGBoost-Classifier-orange.svg)](https://xgboost.readthedocs.io/)
[![XAI](https://img.shields.io/badge/SHAP-Explainable_AI-brightgreen.svg)](https://shap.readthedocs.io/)

An end-to-end machine learning web application that predicts customer subscription churn and leverages **SHAP (SHapley Additive exPlanations)** to provide interpretable, individual-level business insights.

---

## 📌 Business Overview

Acquiring a new customer can cost **5x to 25x more** than retaining an existing one. For subscription-based business models, proactively identifying at-risk customers before they cancel allows retention teams to execute targeted campaigns (e.g., promotional discounts, contract extensions, or customer support outreach).

This project transitions machine learning from a "black-box" model into an actionable business decision support system:
1. **Predicts** probability of customer churn using an engineered **XGBoost Classifier**.
2. **Explains** *why* an individual customer is predicted to churn using **SHAP waterfall visualizers**.
3. **Recommends** specific retention strategies based on individual risk drivers.

---

## 🛠️ Tech Stack & Tools

* **Data Manipulation & Pipeline:** `pandas`, `numpy`, `scikit-learn`
* **Machine Learning:** `XGBoost`
* **Model Explainability (XAI):** `SHAP`
* **Web Interface:** `Streamlit`
* **Serialization & Environment:** `joblib`, `Python 3.10+`

---

## 📊 Key Findings & Model Performance

The dataset used is the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset. Because subscription churn exhibits significant class imbalance (fewer customers churn than stay), the model was optimized for **ROC-AUC** and **Recall** rather than raw accuracy.

* **Primary Algorithm:** XGBoost Classifier (weighted with `scale_pos_weight`)
* **ROC-AUC Score:** ~0.84
* **Top Churn Drivers (via SHAP):**
  1. **Contract Type:** Month-to-month contracts significantly increase churn probability.
  2. **Tenure:** Shorter customer lifecycle length directly correlates with higher churn risk.
  3. **Internet Service:** Fiber optic subscribers with high monthly charges experience higher friction.

---

## 📁 Repository Structure

```text
customer-churn-predictor/
├── data/                       # Dataset directory
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                     # Serialized pipeline artifacts
│   └── churn_model_pipeline.joblib
├── app.py                      # Interactive Streamlit Web Application
├── train_model.py              # Data preprocessing & model training script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git untracked patterns
└── README.md                   # Project documentation

## How To Run Locally
1. Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/customer-churn-predictor.git](https://github.com/YOUR_USERNAME/customer-churn-predictor.git)
cd customer-churn-predictor

2. Set Up Virtual Environment & Dependencies
Bash
# Create environment
python -m venv venv

# Activate environment (macOS/Linux)
source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

3. Train the Model
Make sure the dataset is placed inside the data/ directory, then run:

Bash
python train_model.py

4. Launch the Web Application
Bash
streamlit run app.py

Author & Contact
Ethan Vu

Data Science Student / Aspiring Data Scientist

LinkedIn: www.linkedin.com/in/ethan-vu-b7807b285

GitHub: @1ethanvu

Email: evu495848@gmail.com
