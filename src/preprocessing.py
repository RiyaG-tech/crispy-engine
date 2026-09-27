"""
Data Preprocessing Pipeline for Customer Churn Prediction.

Handles data cleaning, type conversion, missing value imputation,
categorical encoding, and numerical scaling using Scikit-Learn pipelines.
Prevents data leakage by strictly separating train and test fitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Key Column Definitions
ID_COL = "customerID"
TARGET_COL = "Churn"

NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data and performs initial validation.
    
    Args:
        filepath: Path to the CSV dataset.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("The dataset file is empty.")
    
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer churn data:
    - Removes duplicate records
    - Converts TotalCharges to numeric, imputing empty/whitespace strings
    - Formats SeniorCitizen as categorical string ('Yes'/'No') for consistency
    - Cleans whitespace in string columns
    
    Args:
        df: Raw DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicate customer records if any
    initial_count = len(df)
    df = df.drop_duplicates()
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL])
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Fix TotalCharges data type
    if "TotalCharges" in df.columns:
        # Convert spaces or invalid strings to NaN
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
        # When tenure is 0, customers have not been billed yet, so TotalCharges = 0
        df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
        
        # Fill any remaining NaNs with the median
        if df["TotalCharges"].isnull().any():
            median_val = df["TotalCharges"].median()
            df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
            
    # Format SeniorCitizen to Yes/No string for categorical treatment
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No", "1": "Yes", "0": "No"}).fillna("No")
        
    return df


def get_feature_preprocessor() -> ColumnTransformer:
    """
    Constructs a robust Scikit-learn ColumnTransformer preprocessor.
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder
    
    Returns:
        ColumnTransformer: Configured transformer
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop=None, handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ],
        remainder="drop"
    )
    
    return preprocessor


def extract_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """
    Extracts the output feature names from a fitted ColumnTransformer.
    
    Args:
        fitted_preprocessor: Fitted ColumnTransformer instance
        
    Returns:
        list[str]: Transformed feature column names
    """
    feature_names = []
    
    # Numerical names
    feature_names.extend(NUMERICAL_COLS)
    
    # Categorical one-hot names
    cat_transformer = fitted_preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    cat_feature_names = encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names.extend(cat_feature_names)
    
    return feature_names


def prepare_training_data(df: pd.DataFrame):
    """
    Separates features and target, encodes target (Yes=1, No=0).
    
    Args:
        df: Cleaned DataFrame containing features and target
        
    Returns:
        tuple: (X, y)
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in dataset.")
        
    y = df[TARGET_COL].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).astype(int)
    X = df[ALL_FEATURE_COLS].copy()
    
    return X, y
