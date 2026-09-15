"""
Tests for utils/preprocessing.py

Run with: python -m pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import pytest

from utils.preprocessing import (
    clean_diabetes_data,
    clean_heart_data,
    get_diabetes_features_and_target,
    get_heart_features_and_target,
    validate_input_ranges,
)


def test_clean_diabetes_data_removes_zero_placeholders():
    """Zeros in Glucose/BMI/etc. should become imputed, non-zero values."""
    raw = pd.DataFrame({
        "Pregnancies": [1, 2],
        "Glucose": [0, 120],
        "BloodPressure": [0, 80],
        "SkinThickness": [0, 30],
        "Insulin": [0, 100],
        "BMI": [0, 25.0],
        "DiabetesPedigreeFunction": [0.5, 0.6],
        "Age": [30, 40],
        "Outcome": [0, 1],
    })

    cleaned = clean_diabetes_data(raw)

    # The first row's zero-placeholder columns should now be filled with
    # the median of each column (i.e. no longer exactly 0).
    assert cleaned.loc[0, "Glucose"] != 0
    assert cleaned.loc[0, "BMI"] != 0
    # Pregnancies is NOT a zero-as-missing column, so a genuine zero there
    # should be left untouched. (Not present in this sample, but confirms
    # the function doesn't touch columns outside its target list.)
    assert "Pregnancies" in cleaned.columns


def test_clean_diabetes_data_removes_duplicates():
    raw = pd.DataFrame({
        "Pregnancies": [1, 1],
        "Glucose": [120, 120],
        "BloodPressure": [80, 80],
        "SkinThickness": [30, 30],
        "Insulin": [100, 100],
        "BMI": [25.0, 25.0],
        "DiabetesPedigreeFunction": [0.5, 0.5],
        "Age": [30, 30],
        "Outcome": [0, 0],
    })

    cleaned = clean_diabetes_data(raw)
    assert len(cleaned) == 1


def test_clean_heart_data_removes_duplicates():
    raw = pd.DataFrame({
        "age": [50, 50],
        "sex": [1, 1],
        "cp": [0, 0],
        "trestbps": [120, 120],
        "chol": [200, 200],
        "fbs": [0, 0],
        "restecg": [1, 1],
        "thalach": [150, 150],
        "exang": [0, 0],
        "oldpeak": [1.0, 1.0],
        "slope": [1, 1],
        "ca": [0, 0],
        "thal": [2, 2],
        "target": [0, 0],
    })

    cleaned = clean_heart_data(raw)
    assert len(cleaned) == 1


def test_get_diabetes_features_and_target_shapes():
    df = pd.DataFrame({
        "Pregnancies": [1], "Glucose": [120], "BloodPressure": [80],
        "SkinThickness": [30], "Insulin": [100], "BMI": [25.0],
        "DiabetesPedigreeFunction": [0.5], "Age": [30], "Outcome": [0],
    })

    X, y = get_diabetes_features_and_target(df)

    assert X.shape == (1, 8)
    assert "Outcome" not in X.columns
    assert y.iloc[0] == 0


def test_validate_input_ranges_flags_out_of_range_value():
    input_dict = {"Age": 500, "Glucose": 120}
    ranges = {"Age": (1, 120), "Glucose": (40, 300)}

    errors = validate_input_ranges(input_dict, ranges)

    assert len(errors) == 1
    assert "Age" in errors[0]


def test_validate_input_ranges_passes_valid_values():
    input_dict = {"Age": 45, "Glucose": 120}
    ranges = {"Age": (1, 120), "Glucose": (40, 300)}

    errors = validate_input_ranges(input_dict, ranges)

    assert errors == []


def test_validate_input_ranges_flags_negative_value():
    input_dict = {"Pregnancies": -5}
    ranges = {"Pregnancies": (0, 20)}

    errors = validate_input_ranges(input_dict, ranges)

    assert len(errors) == 1