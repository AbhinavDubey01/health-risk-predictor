# AI Health Risk Predictor

**Machine Learning Based Health Risk Assessment System**

An educational web application that estimates statistical risk of Diabetes
and Heart Disease using machine learning models trained on public clinical
datasets.

> **This is not a medical diagnostic tool.** All results are statistical
> estimates for educational purposes only. Always consult a qualified
> healthcare professional for medical advice.

---

## Overview

This project demonstrates a complete, end-to-end data science and machine
learning workflow — from raw data cleaning through model training,
evaluation, and deployment as a working Flask web application. It compares
five classification algorithms per disease and automatically selects the
best-performing model based on evaluation metrics, rather than assuming a
single algorithm in advance.

## Features

- Risk assessment for **Diabetes** and **Heart Disease**
- Compares 5 ML algorithms per disease: Logistic Regression, Decision Tree,
  Random Forest, Support Vector Machine, Gradient Boosting
- Automatic best-model selection based on ROC-AUC (F1 score as tiebreaker)
- Prediction results include: risk level (Low/Moderate/High), probability
  score, top contributing factors, and general health suggestions
- Data Science dashboard: dataset statistics, model comparison table,
  feature importance charts, confusion matrix
- Full Jupyter Notebook covering data cleaning, EDA, and visualization
- Client-side and server-side input validation
- Automated test suite (23 tests) covering preprocessing, prediction, and
  Flask routes
- No paid APIs, no user accounts, no external data transmission — runs
  entirely locally

## Technologies

| Layer | Technology |
|---|---|
| Backend | Python 3.13, Flask |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5 |
| Model Persistence | Joblib |
| Testing | Pytest |

## System Architecture

```
Dataset (CSV) → Cleaning → Feature/Target Split → Train/Test Split
    → StandardScaler + Model (Pipeline) → Train 5 candidate models
    → Evaluate on test set → Select best by ROC-AUC → Save (.pkl + .json)
    → Flask loads saved pipeline → User submits form → Validate input
    → Predict → Display risk level, probability, contributing factors
```

## Machine Learning Workflow

1. **Data Collection** — public Kaggle datasets (see below)
2. **Data Cleaning** — handling missing values, placeholder zeros,
   duplicate removal
3. **Exploratory Data Analysis** — distributions, correlation analysis,
   outlier inspection
4. **Feature Engineering** — consistent feature ordering via `config.py`
5. **Train/Test Split** — stratified 80/20 split, performed *before* any
   scaling to prevent data leakage
6. **Model Training** — 5 algorithms trained inside Scikit-learn `Pipeline`
   objects (scaler + model bundled together)
7. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion
   Matrix for every model
8. **Model Selection** — best model chosen automatically by ROC-AUC
9. **Feature Importance** — extracted from the winning model, clearly
   labeled as "model feature importance," not medical causation
10. **Deployment** — saved pipeline loaded by Flask for live prediction

## Dataset Information

| Disease | Dataset | Source | Records Used | Features |
|---|---|---|---|---|
| Diabetes | Pima Indians Diabetes Database | Kaggle | 768 | 8 |
| Heart Disease | Heart Disease Dataset (UCI Cleveland) | Kaggle | 302 (deduplicated from 1025) | 13 |

**Diabetes dataset note:** Zero values in `Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin`, and `BMI` were identified as placeholder
missing values (not genuine zero readings) and imputed using the median.

**Heart Disease dataset note:** The raw file contains 723 exact duplicate
rows (likely from upsampling during dataset compilation). These were
removed before training to prevent data leakage between train and test
splits.

## Installation (Windows + VS Code)

### Prerequisites
- Python 3.10–3.12 installed, added to PATH
- VS Code with the Python and Jupyter extensions

### 1. Clone or download this project
```
cd C:\Users\<YourUsername>\Documents
```
(Place the project outside any OneDrive-synced folder to avoid file-lock
issues during environment setup.)

### 2. Open in VS Code
```
code .
```

### 3. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\Activate.ps1
```
If PowerShell blocks activation:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Install dependencies
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Download the datasets

- **Diabetes:** [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/carolkeeling/pima-indians-diabetes-database)
  Download and place as `data/diabetes.csv`
- **Heart Disease:** [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
  Download and place as `data/heart.csv`

Verify placement:
```
python -c "import pandas as pd; d = pd.read_csv('data/diabetes.csv'); h = pd.read_csv('data/heart.csv'); print('Diabetes:', d.shape); print('Heart:', h.shape)"
```
Expected: `Diabetes: (768, 9)` and `Heart: (1025, 14)`.

### 6. Train the models
```
python train_models.py
```
This trains and compares all 5 algorithms for both diseases and saves the
best model + metadata into `models/`.

### 7. Run the Flask app
```
python app.py
```

### 8. Open the application
Visit `http://127.0.0.1:5000` in your browser.

### 9. Run tests
```
python -m pytest tests\ -v
```

## Usage

1. From the Home page, choose **Diabetes Check** or **Heart Check**
2. Fill in the requested health parameters
3. Submit to view your risk level, probability score, and contributing
   factors
4. Visit the **Dashboard** to see model comparison charts, dataset
   statistics, and confusion matrices
5. Visit **About** for project details, technologies, and limitations

## Model Evaluation Summary

| Disease | Best Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Diabetes | Gradient Boosting | 74.7% | 65.3% | 59.3% | 62.1% | 82.7% |
| Heart Disease | Logistic Regression | 80.3% | 80.0% | 84.9% | 82.4% | 87.1% |

Random Forest was not the top performer for either disease — both were
selected automatically based on ROC-AUC, not assumed in advance.

## Project Structure

```
health-risk-predictor/
│
├── app.py                     # Flask application and routes
├── train_models.py            # Model training and evaluation script
├── config.py                  # Central configuration (paths, features, thresholds)
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── diabetes.csv
│   └── heart.csv
│
├── models/
│   ├── diabetes_model.pkl
│   ├── diabetes_metadata.json
│   ├── heart_model.pkl
│   └── heart_metadata.json
│
├── notebooks/
│   └── health_risk_analysis.ipynb
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── assessment.html
│   ├── result.html
│   ├── dashboard.html
│   └── about.html
│
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/
│
├── utils/
│   ├── preprocessing.py
│   └── prediction.py
│
└── tests/
    ├── test_preprocessing.py
    ├── test_prediction.py
    └── test_app.py
```

## Project Limitations

- Statistical risk estimate only — not a medical diagnosis
- Trained on relatively small public datasets, limiting generalizability
- The Diabetes dataset was originally collected from an adult female
  population; predictions for male users apply the model outside its
  original training population
- The Heart Disease assessment requires specialist clinical test values
  (ECG stress test results, fluoroscopy) not typically available to a
  layperson without recent medical testing
- Feature importance reflects overall model behavior, not a per-user
  causal explanation (no SHAP or similar instance-level explainability
  was used, to keep dependencies minimal)
- Diabetes model recall (~59%) means a meaningful proportion of true
  positive cases are missed — a limitation of the small, imbalanced
  source dataset

## Future Enhancements

- Add a third disease (Hypertension), following the existing modular
  pattern in `config.py` / `utils/`
- Instance-level explainability (e.g., SHAP values) for per-user
  explanations
- Larger, more diverse training datasets
- Deployment to a cloud platform with proper authentication (if scope
  ever extended beyond an educational project)

## Privacy

This application does not use user accounts, does not persist submitted
health data, and does not transmit any information to external APIs or
services. All processing happens locally within the Flask session.

## Medical Disclaimer

This system is an educational project developed for academic purposes.
It is **not** intended for medical diagnosis, treatment decisions, or as
a substitute for professional medical advice. Always consult a qualified
healthcare provider regarding any health concerns.

## Author

Abhinav Dubey
BSc IT (Hons), Amity Institute of Information Technology, Amity University, Noida
