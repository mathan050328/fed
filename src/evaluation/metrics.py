"""
Evaluation metrics calculation module.
Computes accuracy, precision, recall, F1, ROC-AUC, log-loss, and confusion matrix.
"""

from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict[str, Any]:
    """
    Computes standard binary classification evaluation metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            metrics["roc_auc"] = round(auc, 4)
        except Exception:
            metrics["roc_auc"] = 0.5

        try:
            loss = float(log_loss(y_true, y_prob, labels=[0, 1]))
            metrics["log_loss"] = round(loss, 4)
        except Exception:
            metrics["log_loss"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    metrics["confusion_matrix"] = cm
    return metrics
