"""
train_model.py
---------------
Trains several classification models on the crop recommendation dataset,
compares them on held-out test data, selects the best performer, and
saves the winning model (plus its label encoder and feature list) to
models/crop_recommendation_model.joblib.

Run directly:
    python -m src.train_model
or:
    python src/train_model.py
"""

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Allow running as a script (python src/train_model.py) as well as a module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import (
    clean_dataset,
    get_feature_target_split,
    load_dataset,
    train_test_split_data,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "crop_recommendation_model.joblib"
METRICS_PATH = MODEL_DIR / "model_comparison_metrics.joblib"

RANDOM_STATE = 42


def build_candidate_models() -> dict:
    """Return a dict of {name: unfitted estimator} to compare."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF Kernel)": SVC(probability=True, random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    except ImportError:
        print("xgboost not installed - skipping XGBoost (optional dependency).")

    return models


def train_and_compare(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Fit every candidate model, evaluate on the test set, and return a
    results DataFrame sorted by F1-score (descending). Also returns the
    dict of fitted models so the best one can be re-used without refitting.
    """
    models = build_candidate_models()
    rows = []
    fitted = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = model.predict(X_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "F1-Score": f1_score(y_test, preds, average="weighted", zero_division=0),
                "Train Time (s)": round(elapsed, 3),
            }
        )
        fitted[name] = model
        print(f"Trained {name:<22} | Accuracy={rows[-1]['Accuracy']:.4f} | F1={rows[-1]['F1-Score']:.4f}")

    results_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return results_df, fitted


def main():
    print("Loading and cleaning dataset...")
    raw_df = load_dataset()
    df = clean_dataset(raw_df)

    X, y = get_feature_target_split(df)

    # Encode the crop name labels to integers for models/metrics that need it,
    # while keeping the encoder so the app can map back to crop names.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y_encoded)

    print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
    print("\nTraining and comparing models...\n")

    results_df, fitted_models = train_and_compare(X_train, X_test, y_train, y_test)

    print("\n=== Model Comparison (sorted by F1-Score) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    # Deploy the model as fitted on the TRAIN split only (not refit on the
    # test set). This keeps the held-out test set honest, so the dashboard's
    # classification report / confusion matrix reflect true generalization
    # performance rather than data the model has already memorized.

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": label_encoder,
        "feature_columns": list(X.columns),
    }
    joblib.dump(artifact, MODEL_PATH)
    joblib.dump(results_df, METRICS_PATH)

    print(f"\nSaved model artifact to: {MODEL_PATH}")
    print(f"Saved comparison metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
