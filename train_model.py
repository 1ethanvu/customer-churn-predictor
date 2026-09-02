import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def train_and_save_churn_model():
    # 1. Load Data
    # Dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    # Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' into your /data folder
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Please download the dataset from Kaggle and place it at: {data_path}"
        )
        
    df = pd.read_csv(data_path)

    # 2. Data Cleaning
    df = df.drop(columns=["customerID"])
    
    # TotalCharges contains blank space strings for brand new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # Target encoding: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Separate Features and Target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Identify Column Types
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    # Transform Train & Test Sets
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Extract feature names after OneHotEncoding for SHAP visualizer compatibility
    ohe_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(ohe_cols)

    # 5. Model Training (Handling Class Imbalance with scale_pos_weight)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_train_prep, y_train)

    # 6. Model Evaluation
    y_pred = model.predict(X_test_prep)
    y_proba = model.predict_proba(X_test_prep)[:, 1]

    print("--- Model Performance Metrics ---")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # 7. Export Model Artifacts
    os.makedirs("models", exist_ok=True)
    
    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }
    
    model_save_path = os.path.join("models", "churn_model_pipeline.joblib")
    joblib.dump(artifacts, model_save_path)
    print(f"\nModel pipeline successfully saved to: {model_save_path}")

if __name__ == "__main__":
    train_and_save_churn_model()