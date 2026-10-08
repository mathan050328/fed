"""
Multi-Disease Evaluation, Comparison, and Feature Explanation Module.
Generates baseline comparisons, convergence charts, confusion matrices,
and feature explanation metrics for Heart Disease, Diabetes, and CKD.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    DISEASES,
    DISEASE_DISPLAY_NAMES,
    MAX_ITER,
    SEED,
    get_disease_paths,
    CENTRALIZED_MODEL_PATH,
    CENTRALIZED_METRICS_PATH,
    FIGURE_ACCURACY_PATH,
    FIGURE_LOSS_PATH,
    FIGURE_COMPARISON_PATH,
    FIGURE_CONFUSION_PATH,
    ROUND_METRICS_PATH,
    ensure_directories,
)
from src.models.logistic_model import train_centralized_model, evaluate_logistic_model
from src.utils.serialization import save_model, load_model


def run_centralized_training(disease: str = "heart") -> dict:
    """Trains centralized baseline model for a given disease."""
    ensure_directories()
    paths = get_disease_paths(disease)

    if not paths["X_train"].exists() or not paths["y_train"].exists():
        raise FileNotFoundError(f"Processed training data missing for {disease}.")

    X_train = pd.read_csv(paths["X_train"]).values.astype(float)
    y_train = pd.read_csv(paths["y_train"]).values.ravel().astype(int)
    X_test = pd.read_csv(paths["X_test"]).values.astype(float)
    y_test = pd.read_csv(paths["y_test"]).values.ravel().astype(int)

    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease.upper())
    print("\n" + "=" * 65)
    print(f"TRAINING CENTRALIZED BASELINE: {display_name.upper()}")
    print(f"Training samples: {len(y_train)} | Test samples: {len(y_test)}")
    print("=" * 65)

    model = train_centralized_model(X_train, y_train, max_iter=MAX_ITER, random_state=SEED)
    metrics, _, _ = evaluate_logistic_model(model, X_test, y_test)

    save_model(model, paths["centralized_model"])
    if disease == "heart":
        save_model(model, CENTRALIZED_MODEL_PATH)

    summary = {
        "disease": disease,
        "model_type": f"Centralized LogisticRegression ({display_name})",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "max_iter": MAX_ITER,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "log_loss": metrics.get("log_loss"),
        "confusion_matrix": metrics["confusion_matrix"],
    }

    with open(paths["centralized_metrics"], "w") as f:
        json.dump(summary, f, indent=2)
    if disease == "heart":
        with open(CENTRALIZED_METRICS_PATH, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"[{disease.upper()}] Centralized Baseline Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    return summary


def train_all_centralized():
    for d in DISEASES:
        run_centralized_training(d)


def plot_disease_figures(disease: str = "heart"):
    """Generates charts for round accuracy, loss, metric comparison, and confusion matrix."""
    paths = get_disease_paths(disease)

    if not paths["round_metrics"].exists() or not paths["centralized_metrics"].exists() or not paths["federated_metrics"].exists():
        return

    with open(paths["centralized_metrics"]) as f:
        cent = json.load(f)
    with open(paths["federated_metrics"]) as f:
        fed = json.load(f)
    df_rounds = pd.read_csv(paths["round_metrics"])

    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease.upper())

    # 1. Accuracy by round
    plt.figure(figsize=(7, 4.8))
    plt.plot(df_rounds["round"], df_rounds["accuracy"], marker="o", lw=2.5, color="#0e7490", label="Federated Global")
    plt.axhline(y=cent["accuracy"], color="#dc2626", ls="--", label=f"Centralized ({cent['accuracy']:.4f})")
    plt.title(f"{display_name} - Global Accuracy by Round", fontsize=12, pad=10)
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.xticks(df_rounds["round"])
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["fig_accuracy"], dpi=180)
    if disease == "heart":
        plt.savefig(FIGURE_ACCURACY_PATH, dpi=180)
    plt.close()

    # 2. Loss by round
    plt.figure(figsize=(7, 4.8))
    plt.plot(df_rounds["round"], df_rounds["loss"], marker="s", lw=2.5, color="#ea580c", label="Federated Loss")
    if cent.get("log_loss"):
        plt.axhline(y=cent["log_loss"], color="#16a34a", ls="--", label=f"Centralized Loss ({cent['log_loss']:.4f})")
    plt.title(f"{display_name} - Global Loss by Round", fontsize=12, pad=10)
    plt.xlabel("Communication Round")
    plt.ylabel("Log Loss")
    plt.xticks(df_rounds["round"])
    plt.grid(True, ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["fig_loss"], dpi=180)
    if disease == "heart":
        plt.savefig(FIGURE_LOSS_PATH, dpi=180)
    plt.close()

    # 3. Metric comparison
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    c_vals = [cent.get(k, 0) for k in keys]
    f_vals = [fed.get(k, 0) for k in keys]

    x = np.arange(len(keys))
    w = 0.35
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(x - w / 2, c_vals, w, label="Centralized ML (Pooled)", color="#1e3a8a")
    plt.bar(x + w / 2, f_vals, w, label="Federated ML (FedAvg)", color="#0d9488")
    plt.title(f"{display_name} - Centralized vs Federated", fontsize=12, pad=10)
    plt.xticks(x, names)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", ls=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(paths["fig_comparison"], dpi=180)
    if disease == "heart":
        plt.savefig(FIGURE_COMPARISON_PATH, dpi=180)
    plt.close()

    # 4. Confusion matrix
    cm_c = np.array(cent.get("confusion_matrix", [[0, 0], [0, 0]]))
    cm_f = np.array(fed.get("confusion_matrix", [[0, 0], [0, 0]]))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, cm, title, cmap in zip(axes, [cm_c, cm_f], ["Centralized", "Federated (FedAvg)"], ["Blues", "Greens"]):
        ax.imshow(cm, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"])
        ax.set_yticklabels(["Negative", "Positive"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontweight="bold")
    plt.tight_layout()
    plt.savefig(paths["fig_confusion"], dpi=180)
    if disease == "heart":
        plt.savefig(FIGURE_CONFUSION_PATH, dpi=180)
    plt.close()

    print(f"[{disease.upper()}] Generated all 4 evaluation charts.")


def get_feature_explanations(disease: str = "heart", top_n: int = 5) -> list[dict]:
    """Calculates feature importance / odds ratios for model explanation."""
    paths = get_disease_paths(disease)
    if not paths["federated_model"].exists() or not paths["X_test"].exists():
        return []

    model = load_model(paths["federated_model"])
    feature_names = pd.read_csv(paths["X_test"]).columns.tolist()
    coefs = model.coef_.ravel()

    # Calculate absolute importance and sign
    importance = []
    for name, w in zip(feature_names, coefs):
        importance.append({
            "feature": name,
            "weight": round(float(w), 4),
            "abs_weight": abs(float(w)),
            "impact": "Increases Risk" if w > 0 else "Decreases Risk",
        })

    importance.sort(key=lambda x: x["abs_weight"], reverse=True)
    return importance[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["centralized", "federated", "compare"], default="compare")
    parser.add_argument("--disease", choices=DISEASES + ["all"], default="all")
    args = parser.parse_args()

    targets = DISEASES if args.disease == "all" else [args.disease]

    if args.mode in ["centralized", "compare"]:
        for d in targets:
            run_centralized_training(d)

    if args.mode in ["federated", "compare"]:
        for d in targets:
            plot_disease_figures(d)


if __name__ == "__main__":
    main()
