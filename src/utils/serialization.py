"""
Model serialization and parameter extraction utilities.
Handles conversions between scikit-learn models, NumPy parameter lists (for Flower),
and persisted artifacts.
"""

from pathlib import Path
from typing import Union
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_model_parameters(model: LogisticRegression) -> list[np.ndarray]:
    """
    Extracts weights (coefficients) and bias (intercept) from a trained LogisticRegression model
    as a list of NumPy arrays: [coef_, intercept_].
    """
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("Model is not fitted. Cannot extract parameters.")
    return [model.coef_.copy(), model.intercept_.copy()]


def set_model_parameters(
    model: LogisticRegression,
    parameters: list[np.ndarray],
    n_features: int = None,
    classes: np.ndarray = None,
) -> LogisticRegression:
    """
    Sets weights and bias on a LogisticRegression model from a list of NumPy arrays.
    Initializes metadata attributes if the model has not been fitted yet.
    """
    if len(parameters) != 2:
        raise ValueError(f"Expected 2 parameter arrays [coef, intercept], got {len(parameters)}")

    coef = np.array(parameters[0], dtype=np.float64)
    intercept = np.array(parameters[1], dtype=np.float64)

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if intercept.ndim == 0:
        intercept = np.array([intercept], dtype=np.float64)

    model.coef_ = coef
    model.intercept_ = intercept

    if classes is None:
        classes = np.array([0, 1])
    model.classes_ = classes

    if n_features is None:
        n_features = coef.shape[1]
    model.n_features_in_ = n_features

    return model


def save_model(model, path: Union[str, Path]) -> None:
    """
    Persists a model to disk using joblib.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Union[str, Path]):
    """
    Loads a model from disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)
