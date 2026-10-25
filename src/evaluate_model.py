"""
evaluate_model.py
------------------
Loads the saved model artifact and produces a detailed evaluation report:
classification report, confusion matrix, and (for tree-based models)
feature importances. Used both as a standalone script and imported by
the Streamlit "Model Performance Dashboard" tab.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import (
    clean_dataset,
    get_feature_target_split,
    load_dataset,
    train_test_split_data,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crop_recommendation_model.joblib"
METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "model_comparison_metrics.joblib"


def load_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.train_model` first."
        )
    return joblib.load(path)


def load_comparison_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return joblib.load(path)


def get_test_split_for_eval(artifact: dict):
    """
    Rebuild the same stratified split used at training time so evaluation
    numbers reported here are consistent with train_model.py's console output.
    Note: the deployed model was refit on all data after this split was
    used for model selection, so these numbers describe generalization
    performance, not the exact weights currently deployed.
    """
    raw_df = load_dataset()
    df = clean_dataset(raw_df, verbose=False)
    X, y = get_feature_target_split(df)

    label_encoder = artifact["label_encoder"]
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y_encoded)
    return X_test, y_test


def evaluate(artifact: dict = None):
    """Return (classification_report_str, confusion_matrix_df, class_names)."""
    if artifact is None:
        artifact = load_artifact()

    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    class_names = list(label_encoder.classes_)

    X_test, y_test = get_test_split_for_eval(artifact)
    preds = model.predict(X_test)

    report = classification_report(
        y_test, preds, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return report, cm_df, class_names


def get_feature_importance(artifact: dict = None) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances if the underlying model
    supports it (tree-based models). Returns an empty DataFrame otherwise
    (e.g. for SVM/KNN/Logistic Regression without coefficients handled).
    """
    if artifact is None:
        artifact = load_artifact()

    model = artifact["model"]
    features = artifact["feature_columns"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Average absolute coefficient magnitude across classes as a proxy
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])

    df = pd.DataFrame({"Feature": features, "Importance": importances})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    artifact = load_artifact()
    print(f"Loaded model: {artifact['model_name']}\n")

    report, cm_df, class_names = evaluate(artifact)
    print("=== Classification Report ===")
    print(report)

    print("\n=== Feature Importance ===")
    print(get_feature_importance(artifact).to_string(index=False))

    comparison = load_comparison_metrics()
    if not comparison.empty:
        print("\n=== Model Comparison (from training run) ===")
        print(comparison.to_string(index=False))
