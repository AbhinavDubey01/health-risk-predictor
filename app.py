"""
app.py

Main Flask application for the AI Health Risk Predictor.

Routes:
    /                       Home page
    /assessment/<disease>   Risk assessment form (diabetes or heart)
    /predict/<disease>      Handles form submission, runs prediction
    /dashboard              Data science dashboard (model comparison, EDA stats)
    /about                  About the project

This is an educational risk-assessment tool, NOT a medical diagnostic
system. Every prediction result includes an explicit disclaimer.
"""

import json

from flask import Flask, render_template, request, flash, redirect, url_for

import config
from utils.preprocessing import validate_input_ranges
from utils.prediction import predict_diabetes_risk, predict_heart_disease_risk

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-if-deploying-publicly"


def load_metadata_safe(metadata_path: str):
    """
    Load a metadata JSON file, returning None instead of raising if the
    file is missing. Used by routes that need to display gracefully even
    before train_models.py has been run.
    """
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/assessment/<disease>")
def assessment(disease):
    if disease not in ("diabetes", "heart"):
        return "Unknown disease type.", 404

    if disease == "diabetes":
        input_ranges = config.DIABETES_INPUT_RANGES
        feature_columns = config.DIABETES_FEATURE_COLUMNS
    else:
        input_ranges = config.HEART_INPUT_RANGES
        feature_columns = config.HEART_FEATURE_COLUMNS

    return render_template(
        "assessment.html",
        disease=disease,
        input_ranges=input_ranges,
        feature_columns=feature_columns,
    )


@app.route("/predict/<disease>", methods=["POST"])
def predict(disease):
    if disease not in ("diabetes", "heart"):
        return "Unknown disease type.", 404

    if disease == "diabetes":
        feature_columns = config.DIABETES_FEATURE_COLUMNS
        input_ranges = config.DIABETES_INPUT_RANGES
    else:
        feature_columns = config.HEART_FEATURE_COLUMNS
        input_ranges = config.HEART_INPUT_RANGES

    # --- Step 1: parse and validate submitted form data ---
    input_dict = {}
    parse_errors = []

    for col in feature_columns:
        raw_value = request.form.get(col, "").strip()

        if raw_value == "":
            parse_errors.append(f"{col} is required.")
            continue

        try:
            # Use float for everything at parse time; ranges/model handle
            # both int-like and decimal fields correctly either way.
            input_dict[col] = float(raw_value)
        except ValueError:
            parse_errors.append(f"{col} must be a valid number (got '{raw_value}').")

    if parse_errors:
        for error in parse_errors:
            flash(error, "error")
        return redirect(url_for("assessment", disease=disease))

    # --- Step 2: range validation ---
    range_errors = validate_input_ranges(input_dict, input_ranges)
    if range_errors:
        for error in range_errors:
            flash(error, "error")
        return redirect(url_for("assessment", disease=disease))

    # --- Step 3: run prediction, with friendly error handling ---
    try:
        if disease == "diabetes":
            result = predict_diabetes_risk(input_dict)
        else:
            result = predict_heart_disease_risk(input_dict)
    except FileNotFoundError as e:
        flash(str(e), "error")
        return redirect(url_for("assessment", disease=disease))
    except Exception as e:
        flash(f"Something went wrong while generating the prediction: {e}", "error")
        return redirect(url_for("assessment", disease=disease))

    return render_template(
        "result.html",
        disease=disease,
        result=result,
        input_dict=input_dict,
    )


@app.route("/dashboard")
def dashboard():
    diabetes_metadata = load_metadata_safe(config.DIABETES_METADATA_PATH)
    heart_metadata = load_metadata_safe(config.HEART_METADATA_PATH)

    return render_template(
        "dashboard.html",
        diabetes_metadata=diabetes_metadata,
        heart_metadata=heart_metadata,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)