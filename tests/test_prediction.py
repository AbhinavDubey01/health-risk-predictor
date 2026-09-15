"""
Tests for utils/prediction.py

Requires trained models to already exist in models/ (run train_models.py
first if these fail with a "model file not found" error).

Run with: python -m pytest tests/test_prediction.py -v
"""

import pytest

from utils.prediction import (
    predict_diabetes_risk,
    predict_heart_disease_risk,
    probability_to_risk_level,
)


SAMPLE_DIABETES_INPUT = {
    "Pregnancies": 2, "Glucose": 150, "BloodPressure": 80,
    "SkinThickness": 30, "Insulin": 100, "BMI": 32.5,
    "DiabetesPedigreeFunction": 0.5, "Age": 45,
}

SAMPLE_HEART_INPUT = {
    "age": 55, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
    "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1,
    "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2,
}


def test_probability_to_risk_level_boundaries():
    assert probability_to_risk_level(0.10) == "Low"
    assert probability_to_risk_level(0.50) == "Moderate"
    assert probability_to_risk_level(0.90) == "High"


def test_predict_diabetes_risk_returns_expected_keys():
    result = predict_diabetes_risk(SAMPLE_DIABETES_INPUT)

    expected_keys = {
        "risk_label", "probability", "risk_level",
        "top_factors", "model_name", "disclaimer",
    }
    assert expected_keys.issubset(result.keys())


def test_predict_diabetes_risk_probability_in_valid_range():
    result = predict_diabetes_risk(SAMPLE_DIABETES_INPUT)
    assert 0 <= result["probability"] <= 100


def test_predict_diabetes_risk_has_disclaimer_text():
    result = predict_diabetes_risk(SAMPLE_DIABETES_INPUT)
    assert "not a medical diagnosis" in result["disclaimer"].lower()


def test_predict_heart_disease_risk_returns_expected_keys():
    result = predict_heart_disease_risk(SAMPLE_HEART_INPUT)

    expected_keys = {
        "risk_label", "probability", "risk_level",
        "top_factors", "model_name", "disclaimer",
    }
    assert expected_keys.issubset(result.keys())


def test_predict_heart_disease_risk_probability_in_valid_range():
    result = predict_heart_disease_risk(SAMPLE_HEART_INPUT)
    assert 0 <= result["probability"] <= 100