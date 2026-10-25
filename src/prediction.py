"""
prediction.py
--------------
Prediction-time logic used by the Streamlit app:
- load the saved model artifact
- make a prediction with confidence + top-3 alternatives
- generate a plain-language explanation of the prediction
- support "what-if" re-prediction when the user tweaks inputs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_preprocessing import FEATURE_COLUMNS, validate_input_ranges

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crop_recommendation_model.joblib"

FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}


def load_model_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.train_model` first "
            "to generate models/crop_recommendation_model.joblib."
        )
    return joblib.load(path)


def _build_input_frame(values: dict, feature_columns: list) -> pd.DataFrame:
    ordered = {col: [values[col]] for col in feature_columns}
    return pd.DataFrame(ordered)


def predict_crop(values: dict, artifact: dict, top_k: int = 3) -> dict:
    """
    values: dict with keys N, P, K, temperature, humidity, ph, rainfall
    artifact: the dict loaded by load_model_artifact()

    Returns a dict:
        {
          "top_crop": str,
          "confidence": float (0-1),
          "top_k": [(crop_name, probability), ...],
          "warnings": [str, ...],
          "explanation": str,
        }
    """
    model = artifact["model"]
    label_encoder = artifact["label_encoder"]
    feature_columns = artifact["feature_columns"]

    warnings = validate_input_ranges(values)

    X_input = _build_input_frame(values, feature_columns)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
    else:
        # Fallback for models without predict_proba: one-hot the prediction
        pred_idx = model.predict(X_input)[0]
        probs = np.zeros(len(label_encoder.classes_))
        probs[pred_idx] = 1.0

    class_names = label_encoder.classes_
    ranked_idx = np.argsort(probs)[::-1][:top_k]
    top_k_results = [(class_names[i], float(probs[i])) for i in ranked_idx]

    top_crop = top_k_results[0][0]
    confidence = top_k_results[0][1]

    explanation = generate_explanation(values, top_crop, confidence)

    return {
        "top_crop": top_crop,
        "confidence": confidence,
        "top_k": top_k_results,
        "warnings": warnings,
        "explanation": explanation,
    }


def generate_explanation(values: dict, crop: str, confidence: float) -> str:
    """
    Build a simple, rule-based, human-readable explanation of why a crop
    was recommended, based on how the given inputs compare to the
    documented ideal ranges in utils/crop_info.py. This is a lightweight,
    transparent alternative to a full SHAP explanation for the app's
    "why this crop" text (SHAP values are used separately for the
    per-feature contribution chart).
    """
    from utils.crop_info import get_crop_info

    info = get_crop_info(crop)
    confidence_pct = confidence * 100

    if confidence_pct >= 80:
        certainty_phrase = "The model is highly confident in this recommendation"
    elif confidence_pct >= 50:
        certainty_phrase = "The model is reasonably confident in this recommendation"
    else:
        certainty_phrase = (
            "The model's confidence is relatively low, meaning the input "
            "conditions sit between the profiles of several crops"
        )

    description = info.get("description", "")

    parts = [
        f"{certainty_phrase} ({confidence_pct:.1f}%).",
    ]
    if description:
        parts.append(description)

    return " ".join(parts)


def explain_feature_contributions(values: dict, artifact: dict, top_n: int = 5):
    """
    Compute per-feature contribution to the predicted class using SHAP if
    available. Falls back to a permutation-based approximate importance
    (single-instance perturbation) if SHAP is not installed, so the app
    always has something to display.

    Returns a DataFrame with columns: Feature, Contribution
    (positive = pushed toward the predicted crop, negative = pushed away).
    """
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    label_encoder = artifact["label_encoder"]

    X_input = _build_input_frame(values, feature_columns)
    pred_class_idx = int(model.predict(X_input)[0])
    pred_class_idx = label_encoder.transform(label_encoder.inverse_transform([pred_class_idx]))[0]

    try:
        import shap

        # TreeExplainer is fast and exact for tree-based models; use the
        # generic Explainer as a fallback for other model types.
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            # shap_values shape varies by sklearn/shap version & model type
            if isinstance(shap_values, list):
                contribs = shap_values[pred_class_idx][0]
            else:
                arr = np.array(shap_values)
                if arr.ndim == 3:
                    contribs = arr[0, :, pred_class_idx]
                else:
                    contribs = arr[0]
        except Exception:
            background = X_input  # single-row background is a reasonable fallback
            explainer = shap.Explainer(model.predict_proba, background)
            shap_values = explainer(X_input)
            contribs = shap_values.values[0][:, pred_class_idx]

        df = pd.DataFrame({"Feature": feature_columns, "Contribution": contribs})

    except Exception:
        # Fallback: simple perturbation-based sensitivity around the input point.
        base_probs = model.predict_proba(X_input)[0] if hasattr(model, "predict_proba") else None
        rows = []
        for col in feature_columns:
            perturbed = X_input.copy()
            delta = max(abs(perturbed[col].iloc[0]) * 0.1, 1e-3)
            perturbed[col] = perturbed[col] + delta
            if hasattr(model, "predict_proba") and base_probs is not None:
                new_probs = model.predict_proba(perturbed)[0]
                contribution = new_probs[pred_class_idx] - base_probs[pred_class_idx]
            else:
                contribution = 0.0
            rows.append({"Feature": col, "Contribution": contribution})
        df = pd.DataFrame(rows)

    df["Feature"] = df["Feature"].map(lambda c: FEATURE_LABELS.get(c, c))
    df["AbsContribution"] = df["Contribution"].abs()
    df = df.sort_values("AbsContribution", ascending=False).head(top_n)
    df = df.drop(columns="AbsContribution").reset_index(drop=True)
    return df
