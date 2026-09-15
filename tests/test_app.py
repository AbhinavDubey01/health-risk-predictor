"""
Tests for app.py Flask routes.

Uses Flask's built-in test client, so no running server is required.

Run with: python -m pytest tests/test_app.py -v
"""

import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Health Risk Predictor" in response.data


def test_diabetes_assessment_page_loads(client):
    response = client.get("/assessment/diabetes")
    assert response.status_code == 200
    assert b"Diabetes Risk Assessment" in response.data


def test_heart_assessment_page_loads(client):
    response = client.get("/assessment/heart")
    assert response.status_code == 200
    assert b"Heart Disease Risk Assessment" in response.data


def test_unknown_disease_assessment_returns_404(client):
    response = client.get("/assessment/unknown_disease")
    assert response.status_code == 404


def test_dashboard_page_loads(client):
    response = client.get("/dashboard")
    assert response.status_code == 200


def test_about_page_loads(client):
    response = client.get("/about")
    assert response.status_code == 200


def test_predict_diabetes_with_valid_data_shows_result(client):
    response = client.post("/predict/diabetes", data={
        "Pregnancies": "2", "Glucose": "150", "BloodPressure": "80",
        "SkinThickness": "30", "Insulin": "100", "BMI": "32.5",
        "DiabetesPedigreeFunction": "0.5", "Age": "45",
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b"Risk Assessment Result" in response.data


def test_predict_diabetes_with_missing_field_redirects_with_error(client):
    response = client.post("/predict/diabetes", data={
        "Pregnancies": "2", "Glucose": "150", "BloodPressure": "80",
        # SkinThickness intentionally missing
        "Insulin": "100", "BMI": "32.5",
        "DiabetesPedigreeFunction": "0.5", "Age": "45",
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b"is required" in response.data


def test_predict_diabetes_with_invalid_number_redirects_with_error(client):
    response = client.post("/predict/diabetes", data={
        "Pregnancies": "abc", "Glucose": "150", "BloodPressure": "80",
        "SkinThickness": "30", "Insulin": "100", "BMI": "32.5",
        "DiabetesPedigreeFunction": "0.5", "Age": "45",
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b"must be a valid number" in response.data


def test_predict_diabetes_with_out_of_range_value_redirects_with_error(client):
    response = client.post("/predict/diabetes", data={
        "Pregnancies": "2", "Glucose": "9999", "BloodPressure": "80",
        "SkinThickness": "30", "Insulin": "100", "BMI": "32.5",
        "DiabetesPedigreeFunction": "0.5", "Age": "45",
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b"must be between" in response.data