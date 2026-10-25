"""
data_preprocessing.py
----------------------
Loads the raw crop recommendation dataset, validates it, performs basic
cleaning (missing values / duplicates), and splits it into train/test
sets ready for model training.

This module has no side effects on import - call the functions explicitly
from a script or notebook.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Crop_recommendation.csv"

# Reasonable real-world bounds used for input validation elsewhere in the
# project (Streamlit app). Kept here so training/EDA and the app share the
# same source of truth.
FEATURE_RANGES = {
    "N": (0, 140),          # kg/ha
    "P": (5, 145),          # kg/ha
    "K": (5, 205),          # kg/ha
    "temperature": (8, 45), # Celsius
    "humidity": (14, 100),  # %
    "ph": (3.5, 9.5),
    "rainfall": (20, 300),  # mm
}


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Make sure "
            "data/Crop_recommendation.csv exists."
        )
    df = pd.read_csv(path)
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    return df


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Basic cleaning: report + drop missing values and exact duplicate rows.
    Returns a new, cleaned DataFrame (input is not mutated).
    """
    df = df.copy()

    n_missing = df.isnull().sum().sum()
    n_duplicates = df.duplicated().sum()

    if verbose:
        print(f"Missing values found: {n_missing}")
        print(f"Duplicate rows found: {n_duplicates}")

    if n_missing > 0:
        df = df.dropna().reset_index(drop=True)

    if n_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def get_feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into X (features) and y (target label)."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/test split so every crop class is represented in both sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def validate_input_ranges(values: dict) -> list:
    """
    Check a dict of {feature_name: value} against FEATURE_RANGES.
    Returns a list of human-readable warning strings for any value that
    falls outside the realistic range observed in agronomic practice.
    Does NOT raise - callers decide whether warnings block a prediction.
    """
    warnings = []
    for feature, value in values.items():
        if feature not in FEATURE_RANGES:
            continue
        low, high = FEATURE_RANGES[feature]
        if value < low or value > high:
            warnings.append(
                f"{feature} = {value} is outside the typical realistic range "
                f"({low}-{high}). Prediction may be unreliable."
            )
    return warnings


if __name__ == "__main__":
    raw_df = load_dataset()
    clean_df = clean_dataset(raw_df)
    print(f"Raw shape: {raw_df.shape} -> Clean shape: {clean_df.shape}")
    print(clean_df[TARGET_COLUMN].value_counts())
