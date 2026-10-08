"""
Logistic Regression model definition and evaluation for centralized and federated learning.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import SEED, MAX_ITER
from src.evaluation.metrics import compute_classification_metrics
from src.utils.serialization import get_model_parameters, set_model_parameters


def create_logistic_model(
    max_iter: int = MAX_ITER,
    warm_start: bool = False,
    random_state: int = SEED,
    C: float = 1.0,
    solver: str = "lbfgs",
) -> LogisticRegression:
    """
    Factory to instantiate Logistic Regression models with consistent configuration.
    """
    return LogisticRegression(
        max_iter=max_iter,
        warm_start=warm_start,
        random_state=random_state,
        C=C,
        solver=solver,
    )


def train_centralized_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = MAX_ITER,
    random_state: int = SEED,
) -> LogisticRegression:
    """
    Trains a centralized Logistic Regression baseline on pooled training data.
    """
    model = create_logistic_model(max_iter=max_iter, warm_start=False, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_logistic_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluates a logistic regression model, returning metrics dict, predictions, and probabilities.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = compute_classification_metrics(y, pred, proba)
    return metrics, pred, proba
