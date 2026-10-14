"""
Preprocessing utilities for the AI Health Risk Predictor project.

These functions encapsulate the exact cleaning logic developed and verified
in notebooks/health_risk_analysis.ipynb. They are imported by both
train_models.py (training time) and utils/prediction.py (prediction time)
to guarantee identical preprocessing in both places.
"""

import pandas as pd
import numpy as np

import config


def clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Diabetes dataset.

    Steps (matching the analysis in health_risk_analysis.ipynb):
    1. Replace biologically impossible zero values in specific columns
       (Glucose, BloodPressure, SkinThickness, Insulin, BMI) with NaN,
       since zero in these columns represents a missing measurement,
       not a valid reading.
    2. Impute those missing values using each column's median.
    3. Remove exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw diabetes dataframe, as loaded from data/diabetes.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, ready for train/test splitting.
    """
    cleaned_df = df.copy()

    # Step 1: convert placeholder zeros into real missing values
    zero_cols = config.DIABETES_ZERO_AS_MISSING_COLUMNS
    cleaned_df[zero_cols] = cleaned_df[zero_cols].replace(0, np.nan)

    # Step 2: impute with median (robust to outliers, unlike mean)
    for col in zero_cols:
        median_value = cleaned_df[col].median()
        cleaned_df[col] = cleaned_df[col].fillna(median_value)

    # Step 3: remove exact duplicate rows
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    return cleaned_df


def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Heart Disease dataset.

    Steps (matching the analysis in health_risk_analysis.ipynb):
    1. Remove exact duplicate rows. This dataset's Kaggle upload contains
       a large number of duplicated records (723 out of 1025 in our EDA),
       which would cause data leakage between train and test splits if
       left in place.
    2. No zero-as-missing handling is needed here, since 0 is a genuinely
       valid value for this dataset's categorical columns (e.g. sex, fbs).

    Parameters
    ----------
    df : pd.DataFrame
        Raw heart disease dataframe, as loaded from data/heart.csv.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, ready for train/test splitting.
    """
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        # In this Kaggle file the label is reversed: 1 = no disease, 0 = disease.
    # Flip it so 1 always means disease present.
    cleaned_df[config.HEART_TARGET_COLUMN] = 1 - cleaned_df[config.HEART_TARGET_COLUMN]
    return cleaned_df


def get_diabetes_features_and_target(df: pd.DataFrame):
    """
    Split a cleaned diabetes dataframe into features (X) and target (y),
    using the exact column order defined in config.py.

    This column order must match the order used everywhere else in the
    project (training, saved metadata, and the Flask prediction form) —
    scikit-learn models expect features in a fixed, consistent order.
    """
    X = df[config.DIABETES_FEATURE_COLUMNS].copy()
    y = df[config.DIABETES_TARGET_COLUMN].copy()
    return X, y


def get_heart_features_and_target(df: pd.DataFrame):
    """
    Split a cleaned heart disease dataframe into features (X) and target (y),
    using the exact column order defined in config.py.
    """
    X = df[config.HEART_FEATURE_COLUMNS].copy()
    y = df[config.HEART_TARGET_COLUMN].copy()
    return X, y


def validate_input_ranges(input_dict: dict, ranges: dict) -> list:
    """
    Check a dictionary of user-submitted values against expected value
    ranges. Used by the Flask app to catch invalid form input (e.g. a
    negative age, or a BMI far outside plausible bounds) before passing
    data to a trained model.

    Parameters
    ----------
    input_dict : dict
        Feature name -> submitted value (already converted to float/int).
    ranges : dict
        Feature name -> (min_value, max_value), from config.py.

    Returns
    -------
    list
        A list of human-readable error strings. Empty list means all
        values passed validation.
    """
    errors = []

    for feature_name, value in input_dict.items():
        if feature_name not in ranges:
            continue

        min_value, max_value = ranges[feature_name]

        if value < min_value or value > max_value:
            errors.append(
                f"{feature_name} must be between {min_value} and {max_value} "
                f"(you entered {value})."
            )

    return errors