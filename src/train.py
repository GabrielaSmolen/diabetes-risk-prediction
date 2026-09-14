from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from logger import get_logger
from utils import load_data

logger = get_logger(__name__)

MODEL_PATH = Path("models/diabetes_model.pkl")

ZERO_INVALID_COLS = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]
VALID_ZERO_COLS = ["pregnancies", "diabetes_pedigree", "age"]


def replace_zeros_with_nan(x):
    """Replaces invalid 0 values with np.nan while preserving Pandas DataFrame index."""
    if hasattr(x, "replace"):
        return x.replace(0, np.nan)
    return np.where(x == 0, np.nan, x)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    logger.info("Splitting data into features and target...")

    y = df["outcome"]
    X = df.drop(['outcome'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_and_train_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    logger.info("Building and fitting ML Pipeline...")

    medical_transformer = Pipeline([
        ("zero_to_nan", FunctionTransformer(replace_zeros_with_nan)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    normal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("medical", medical_transformer, ZERO_INVALID_COLS),
            ("normal", normal_transformer, VALID_ZERO_COLS),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42,))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = build_and_train_pipeline(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    logger.info(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    logger.info(f"F1-Score: {f1_score(y_test, predictions) * 100:.2f}%")
    logger.info(f"ROC-AUC:  {roc_auc_score(y_test, probabilities) * 100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model successfully saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
