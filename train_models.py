"""
train_models.py

Trains and compares multiple classification models for two diseases:
Diabetes and Heart Disease. For each disease, this script:
    1. Loads and cleans the raw dataset
    2. Splits into train/test sets (before any scaling, to avoid leakage)
    3. Builds a Pipeline (StandardScaler + model) for each candidate algorithm
    4. Evaluates every model on the held-out test set
    5. Automatically selects the best-performing model
    6. Saves the winning pipeline (.pkl) and its metadata (.json)

Run this script directly:
    python train_models.py
"""

import json
import warnings

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import config
from utils.preprocessing import (
    clean_diabetes_data,
    clean_heart_data,
    get_diabetes_features_and_target,
    get_heart_features_and_target,
)

warnings.filterwarnings("ignore")
def get_candidate_models():
    """
    Returns a dictionary of candidate classification models to train and
    compare. Random Forest is included as our expected primary model, but
    it is evaluated on equal footing with the others — the best model is
    selected based on test-set metrics, not assumed in advance.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=config.RANDOM_SEED, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=config.RANDOM_SEED, max_depth=6
        ),
        "Random Forest": RandomForestClassifier(
            random_state=config.RANDOM_SEED, n_estimators=200, max_depth=8
        ),
        "Support Vector Machine": SVC(
            random_state=config.RANDOM_SEED, probability=True, kernel="rbf"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=config.RANDOM_SEED, n_estimators=150
        ),
    }
    return models
def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test set and return a dictionary of
    standard classification metrics plus the confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def select_best_model(results: dict) -> str:
    """
    Select the best model based on ROC-AUC as the primary criterion, with
    F1 score as a tiebreaker. ROC-AUC is used as the primary metric because
    it evaluates how well a model ranks positive vs negative cases across
    all classification thresholds, making it more robust to class imbalance
    than plain accuracy (relevant here, since the Diabetes dataset has a
    ~65/35 class split, not 50/50).
    """
    best_name = None
    best_score = (-1, -1)  # (roc_auc, f1_score)

    for model_name, result in results.items():
        score = (result["metrics"]["roc_auc"], result["metrics"]["f1_score"])
        if score > best_score:
            best_score = score
            best_name = model_name

    return best_name


def train_and_evaluate_disease(
    disease_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_columns: list,
):
    """
    Full training + evaluation workflow for one disease.

    Steps:
        1. Train/test split (stratified, so both sets keep the same class
           balance as the full dataset — important given class imbalance).
        2. Train each candidate model inside a Pipeline that bundles
           StandardScaler + the classifier, so the exact same scaling is
           always applied during prediction as during training.
        3. Evaluate every model on the untouched test set.
        4. Select the best model by ROC-AUC (tiebreak: F1).

    Returns
    -------
    dict with keys: "best_model_name", "best_pipeline", "all_results"
    """
    print(f"\n{'=' * 60}")
    print(f"Training models for: {disease_name}")
    print(f"{'=' * 60}")

    # Stratified split: preserves class ratio in both train and test sets.
    # Splitting BEFORE any scaling avoids data leakage — the scaler is
    # fitted only on training data inside each Pipeline below.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    models = get_candidate_models()
    results = {}

    for model_name, model in models.items():
        # Pipeline bundles scaling + model together. When we save this
        # pipeline, we save the FITTED scaler along with the model, so
        # prediction-time code never has to re-implement scaling logic
        # separately -- eliminating a common source of train/predict
        # preprocessing mismatch.
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[model_name] = {
            "pipeline": pipeline,
            "metrics": metrics,
        }

        print(
            f"{model_name:25s} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    best_model_name = select_best_model(results)
    best_pipeline = results[best_model_name]["pipeline"]

    print(f"\nBest model for {disease_name}: {best_model_name}")
    print(f"ROC-AUC: {results[best_model_name]['metrics']['roc_auc']}")

    # Feature importance / coefficients, where the model type supports it.
    # This directly feeds the "why this prediction" explanation in the
    # Flask app and the dashboard's feature-importance chart.
    feature_importance = extract_feature_importance(
        best_pipeline.named_steps["model"], feature_columns
    )

    return {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "all_results": results,
        "feature_importance": feature_importance,
        "test_size": X_test.shape[0],
        "train_size": X_train.shape[0],
    }


def extract_feature_importance(model, feature_columns: list) -> dict:
    """
    Extract a feature-importance-like score from a trained model, in a
    way that works across different model types:
        - Tree-based models (Random Forest, Decision Tree, Gradient
          Boosting) expose .feature_importances_ directly.
        - Logistic Regression exposes .coef_ ; we use the absolute value
          so magnitude (not direction) reflects relative importance.
        - SVM with an RBF kernel does not expose a simple linear
          importance measure, so an empty dict is returned in that case
          (the app will fall back to a generic message for SVM results).

    This is explicitly labeled "feature importance" wherever displayed --
    it reflects what the model weighted heavily when making predictions,
    not a medically validated causal relationship.
    """
    importance_dict = {}

    if hasattr(model, "feature_importances_"):
        for col, score in zip(feature_columns, model.feature_importances_):
            importance_dict[col] = round(float(score), 4)

    elif hasattr(model, "coef_"):
        coefficients = np.abs(model.coef_[0])
        for col, score in zip(feature_columns, coefficients):
            importance_dict[col] = round(float(score), 4)

    if importance_dict:
        # Sort descending so the most influential feature appears first
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

    return importance_dict


def save_model_and_metadata(
    disease_name: str,
    training_result: dict,
    feature_columns: list,
    model_path: str,
    metadata_path: str,
):
    """
    Save the winning pipeline to disk with Joblib, and save a metadata
    JSON file alongside it containing the feature order, model name, and
    evaluation metrics for every model that was compared. app.py reads
    this metadata to validate incoming form data and to power the
    dashboard's model comparison charts.
    """
    joblib.dump(training_result["best_pipeline"], model_path)
    print(f"Saved model to: {model_path}")

    # Build a JSON-safe summary of every model's metrics for the dashboard
    comparison_summary = {}
    for model_name, result in training_result["all_results"].items():
        comparison_summary[model_name] = result["metrics"]

    metadata = {
        "disease_name": disease_name,
        "best_model_name": training_result["best_model_name"],
        "feature_columns": feature_columns,
        "train_size": training_result["train_size"],
        "test_size": training_result["test_size"],
        "feature_importance": training_result["feature_importance"],
        "model_comparison": comparison_summary,
        "best_model_metrics": training_result["all_results"][
            training_result["best_model_name"]
        ]["metrics"],
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")


def main():
    # ----------------------- DIABETES -----------------------
    diabetes_raw = pd.read_csv(config.DIABETES_CSV_PATH)
    diabetes_clean = clean_diabetes_data(diabetes_raw)
    X_diabetes, y_diabetes = get_diabetes_features_and_target(diabetes_clean)

    diabetes_result = train_and_evaluate_disease(
        "Diabetes", X_diabetes, y_diabetes, config.DIABETES_FEATURE_COLUMNS
    )

    save_model_and_metadata(
        "Diabetes",
        diabetes_result,
        config.DIABETES_FEATURE_COLUMNS,
        config.DIABETES_MODEL_PATH,
        config.DIABETES_METADATA_PATH,
    )

    # ----------------------- HEART DISEASE -----------------------
    heart_raw = pd.read_csv(config.HEART_CSV_PATH)
    heart_clean = clean_heart_data(heart_raw)
    X_heart, y_heart = get_heart_features_and_target(heart_clean)

    heart_result = train_and_evaluate_disease(
        "Heart Disease", X_heart, y_heart, config.HEART_FEATURE_COLUMNS
    )

    save_model_and_metadata(
        "Heart Disease",
        heart_result,
        config.HEART_FEATURE_COLUMNS,
        config.HEART_MODEL_PATH,
        config.HEART_METADATA_PATH,
    )

    print(f"\n{'=' * 60}")
    print("Training complete for both diseases.")
    print(f"{'=' * 60}")


def load_dataset_safe(csv_path: str, dataset_label: str) -> pd.DataFrame:
    """
    Load a CSV dataset with a clear, actionable error message if the file
    is missing, instead of letting a raw pandas/OS-level traceback surface.
    """
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"\n{'!' * 60}")
        print(f"ERROR: {dataset_label} dataset not found.")
        print(f"Expected file at: {csv_path}")
        print("Please download the dataset and place it at this exact path")
        print("before running train_models.py. See README.md for dataset links.")
        print(f"{'!' * 60}\n")
        raise SystemExit(1)


def load_dataset_safe(csv_path: str, dataset_label: str) -> pd.DataFrame:
    """
    Load a CSV dataset with a clear, actionable error message if the file
    is missing, instead of letting a raw pandas/OS-level traceback surface.
    """
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"\n{'!' * 60}")
        print(f"ERROR: {dataset_label} dataset not found.")
        print(f"Expected file at: {csv_path}")
        print("Please download the dataset and place it at this exact path")
        print("before running train_models.py. See README.md for dataset links.")
        print(f"{'!' * 60}\n")
        raise SystemExit(1)


def main():
    # ----------------------- DIABETES -----------------------
    diabetes_raw = load_dataset_safe(config.DIABETES_CSV_PATH, "Diabetes")
    diabetes_clean = clean_diabetes_data(diabetes_raw)
    X_diabetes, y_diabetes = get_diabetes_features_and_target(diabetes_clean)

    diabetes_result = train_and_evaluate_disease(
        "Diabetes", X_diabetes, y_diabetes, config.DIABETES_FEATURE_COLUMNS
    )

    save_model_and_metadata(
        "Diabetes",
        diabetes_result,
        config.DIABETES_FEATURE_COLUMNS,
        config.DIABETES_MODEL_PATH,
        config.DIABETES_METADATA_PATH,
    )

    # ----------------------- HEART DISEASE -----------------------
    heart_raw = load_dataset_safe(config.HEART_CSV_PATH, "Heart Disease")
    heart_clean = clean_heart_data(heart_raw)
    X_heart, y_heart = get_heart_features_and_target(heart_clean)

    heart_result = train_and_evaluate_disease(
        "Heart Disease", X_heart, y_heart, config.HEART_FEATURE_COLUMNS
    )

    save_model_and_metadata(
        "Heart Disease",
        heart_result,
        config.HEART_FEATURE_COLUMNS,
        config.HEART_MODEL_PATH,
        config.HEART_METADATA_PATH,
    )

    print(f"\n{'=' * 60}")
    print("Training complete for both diseases.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()