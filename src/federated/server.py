"""
Multi-Disease Federated Learning Server orchestration module using Flower.
Coordinates training across Hospital A, Hospital B, and Hospital C using FedAvg.
Manages the Model Registry for Heart Disease, Diabetes, and CKD models.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.config import (
    DISEASES,
    DISEASE_DISPLAY_NAMES,
    NUM_CLIENTS,
    NUM_ROUNDS,
    LOCAL_EPOCHS,
    SEED,
    get_disease_paths,
    FEDERATED_MODEL_PATH,
    FEDERATED_METRICS_PATH,
    ROUND_METRICS_PATH,
    ensure_directories,
)
from src.federated.client import HospitalClient
from src.federated.strategy import MetricsTrackingFedAvg
from src.models.logistic_model import create_logistic_model, evaluate_logistic_model
from src.utils.serialization import set_model_parameters, save_model


def get_client_fn(disease: str):
    """Generates a client factory for a specific disease and hospital."""
    def client_fn(arg):
        if hasattr(arg, "node_config") and "partition-id" in arg.node_config:
            raw_id = arg.node_config["partition-id"]
        elif isinstance(arg, str):
            raw_id = arg
        else:
            raw_id = getattr(arg, "cid", 0)

        try:
            client_id = int(raw_id) + 1
        except Exception:
            client_id = 1

        if client_id > NUM_CLIENTS:
            client_id = ((client_id - 1) % NUM_CLIENTS) + 1

        client = HospitalClient(client_id=client_id, disease=disease, local_epochs=LOCAL_EPOCHS)
        return client.to_client()

    return client_fn


def run_federated_server(
    disease: str = "heart",
    num_rounds: int = NUM_ROUNDS,
    num_clients: int = NUM_CLIENTS,
) -> dict:
    """Executes Flower federated learning training run for a given disease."""
    ensure_directories()
    paths = get_disease_paths(disease)

    if not paths["X_test"].exists() or not paths["y_test"].exists():
        raise FileNotFoundError(f"Processed test data missing for {disease}. Run preprocess.py first.")

    X_test_df = pd.read_csv(paths["X_test"])
    y_test_df = pd.read_csv(paths["y_test"])
    X_test = X_test_df.values.astype(float)
    y_test = y_test_df.values.ravel().astype(int)
    num_features = X_test.shape[1]

    display_name = DISEASE_DISPLAY_NAMES.get(disease, disease.upper())
    print("\n" + "=" * 65)
    print(f"FEDERATED SERVER: TRAINING {display_name.upper()} GLOBAL MODEL")
    print(f"Hospitals:               Hospital A, Hospital B, Hospital C (3 clients)")
    print(f"Communication Rounds:    {num_rounds}")
    print(f"Local Epochs / Round:    {LOCAL_EPOCHS}")
    print(f"Held-out Test Cohort:    {len(y_test)} patients")
    print(f"Feature Dimensions:      {num_features}")
    print("=" * 65)

    global_eval_model = create_logistic_model(random_state=SEED)
    init_coef = np.zeros((1, num_features), dtype=np.float64)
    init_intercept = np.zeros(1, dtype=np.float64)
    initial_parameters = ndarrays_to_parameters([init_coef, init_intercept])

    final_state = {"parameters": [init_coef, init_intercept], "last_metrics": {}}

    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: dict):
        set_model_parameters(global_eval_model, parameters, n_features=num_features)
        metrics, _, _ = evaluate_logistic_model(global_eval_model, X_test, y_test)
        loss = metrics.get("log_loss") or float(1.0 - metrics["accuracy"])

        final_state["parameters"] = [p.copy() for p in parameters]
        final_state["last_metrics"] = metrics

        print(
            f"  [Round {server_round}] Loss={loss:.4f} | Acc={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | ROC-AUC={metrics['roc_auc']:.4f}"
        )

        return float(loss), {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        }

    strategy = MetricsTrackingFedAvg(
        evaluate_fn=evaluate_fn,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )

    client_fn = get_client_fn(disease)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    final_params = final_state["parameters"]
    final_model = create_logistic_model(random_state=SEED)
    set_model_parameters(final_model, final_params, n_features=num_features)

    test_metrics, _, _ = evaluate_logistic_model(final_model, X_test, y_test)

    # Save to Global Model Registry
    save_model(final_model, paths["federated_model"])
    if disease == "heart":
        save_model(final_model, FEDERATED_MODEL_PATH)

    federated_summary = {
        "disease": disease,
        "model_type": f"Global Federated {display_name} Model (FedAvg)",
        "hospitals": ["Hospital A", "Hospital B", "Hospital C"],
        "num_clients": num_clients,
        "rounds": num_rounds,
        "local_epochs": LOCAL_EPOCHS,
        "test_samples": len(y_test),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "log_loss": test_metrics.get("log_loss"),
        "confusion_matrix": test_metrics["confusion_matrix"],
    }

    with open(paths["federated_metrics"], "w") as f:
        json.dump(federated_summary, f, indent=2)
    if disease == "heart":
        with open(FEDERATED_METRICS_PATH, "w") as f:
            json.dump(federated_summary, f, indent=2)

    # Save round history
    if strategy.round_history:
        df_rounds = pd.DataFrame(strategy.round_history)
        df_rounds.to_csv(paths["round_metrics"], index=False)
        if disease == "heart":
            df_rounds.to_csv(ROUND_METRICS_PATH, index=False)

    print(f"\n{display_name.upper()} GLOBAL MODEL SAVED TO REGISTRY: {paths['federated_model'].name}")
    print(f"Final Accuracy: {test_metrics['accuracy']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}")
    return federated_summary


def train_all_diseases():
    """Trains global federated models for Heart Disease, Diabetes, and CKD."""
    results = {}
    for d in DISEASES:
        results[d] = run_federated_server(disease=d)
    print("\n" + "=" * 65)
    print("ALL 3 GLOBAL FEDERATED MODELS SUCCESSFULLY REGISTERED")
    print("=" * 65)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in DISEASES:
        run_federated_server(disease=sys.argv[1])
    else:
        train_all_diseases()
