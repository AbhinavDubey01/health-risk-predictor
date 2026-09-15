"""
Central configuration file for the AI Health Risk Predictor project.

This file defines file paths, feature lists, and preprocessing rules used
consistently across train_models.py, utils/preprocessing.py, utils/prediction.py,
and app.py. Keeping these in one place prevents inconsistent preprocessing
between training and prediction, which would otherwise cause silent bugs.
"""

import os

# ---------------------------------------------------------------------------
# BASE PATHS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

DIABETES_CSV_PATH = os.path.join(DATA_DIR, "diabetes.csv")
HEART_CSV_PATH = os.path.join(DATA_DIR, "heart.csv")

DIABETES_MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
DIABETES_METADATA_PATH = os.path.join(MODELS_DIR, "diabetes_metadata.json")

HEART_MODEL_PATH = os.path.join(MODELS_DIR, "heart_model.pkl")
HEART_METADATA_PATH = os.path.join(MODELS_DIR, "heart_metadata.json")

# ---------------------------------------------------------------------------
# RANDOM SEED
# ---------------------------------------------------------------------------
# Using one fixed seed everywhere ensures reproducible train/test splits and
# model training results — important so results in the report/viva match
# what actually gets produced when re-running the code.

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# DIABETES DATASET CONFIGURATION
# ---------------------------------------------------------------------------

DIABETES_TARGET_COLUMN = "Outcome"

# Column order matters: this must match the order features are collected
# in the Flask form and passed to the model at prediction time.
DIABETES_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# These columns use 0 as a placeholder for "missing measurement" rather than
# a genuine zero value. Confirmed during EDA (health_risk_analysis.ipynb).
DIABETES_ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Reasonable real-world input ranges, used for form validation in the Flask
# app. These are NOT hard model constraints — just sanity bounds to catch
# obviously invalid user input (e.g. negative age, BMI of 500).
DIABETES_INPUT_RANGES = {
    "Pregnancies": (0, 17),
    "Glucose": (0, 199),
    "BloodPressure": (0, 122),
    "SkinThickness": (0, 99),
    "Insulin": (0, 846),
    "BMI": (0, 67),
    "DiabetesPedigreeFunction": (0.078, 2.42),
    "Age": (21, 81),
}
# ---------------------------------------------------------------------------
# HEART DISEASE DATASET CONFIGURATION
# ---------------------------------------------------------------------------

HEART_TARGET_COLUMN = "target"

HEART_FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Categorical columns — used later to decide which features need special
# handling in preprocessing (though for tree-based and linear models here,
# these are already integer-encoded so no further encoding is required).
HEART_CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

HEART_INPUT_RANGES = {
    "age": (29, 77),
    "sex": (0, 1),
    "cp": (0, 3),
    "trestbps": (94, 200),
    "chol": (126, 564),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (71, 202),
    "exang": (0, 1),
    "oldpeak": (0.0, 6.2),
    "slope": (0, 2),
    "ca": (0, 4),
    "thal": (0, 3),
}

# ---------------------------------------------------------------------------
# RISK LEVEL THRESHOLDS
# ---------------------------------------------------------------------------
# Used to convert a model's predicted probability into a human-readable
# risk category, shown on the Prediction Result page.

RISK_THRESHOLDS = {
    "low_max": 0.33,       # probability < 0.33  -> Low risk
    "moderate_max": 0.66,  # 0.33 <= probability < 0.66 -> Moderate risk
                            # probability >= 0.66 -> High risk
}