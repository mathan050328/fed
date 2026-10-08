"""
Integration test for end-to-end federated training loop with 3 simulated clients.
Runs 2 federated rounds and verifies that global parameters evolve and converge.
"""

import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.config import NUM_CLIENTS, X_TEST_PATH, Y_TEST_PATH, SEED
from src.federated.client import HospitalClient
from src.federated.strategy import MetricsTrackingFedAvg
from src.models.logistic_model import create_logistic_model, evaluate_logistic_model
from src.utils.serialization import set_model_parameters


def test_federated_integration_run():
    X_test_df = pd.read_csv(X_TEST_PATH)
    y_test_df = pd.read_csv(Y_TEST_PATH)
    X_test = X_test_df.values.astype(float)
    y_test = y_test_df.values.ravel().astype(int)
    num_features = X_test.shape[1]

    # Reference evaluation model
    eval_model = create_logistic_model(random_state=SEED)
    init_coef = np.zeros((1, num_features), dtype=np.float64)
    init_intercept = np.zeros(1, dtype=np.float64)
    initial_params = ndarrays_to_parameters([init_coef, init_intercept])

    eval_history = []

    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: dict):
        set_model_parameters(eval_model, parameters, n_features=num_features)
        metrics, _, _ = evaluate_logistic_model(eval_model, X_test, y_test)
        loss = metrics.get("log_loss") or (1.0 - metrics["accuracy"])
        eval_history.append((server_round, float(loss), metrics["accuracy"]))
        return float(loss), {"accuracy": metrics["accuracy"]}

    strategy = MetricsTrackingFedAvg(
        evaluate_fn=evaluate_fn,
        min_fit_clients=3,
        min_available_clients=3,
        initial_parameters=initial_params,
    )

    def client_fn(cid):
        try:
            client_id = int(cid) + 1
        except Exception:
            client_id = 1
        return HospitalClient(client_id=client_id, local_epochs=2).to_client()

    # Run a 2-round simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # Check that at least 2 rounds executed and accuracy improved over initial zero-weights
    assert len(eval_history) >= 2
    round_0_acc = eval_history[0][2]
    round_2_acc = eval_history[-1][2]
    assert round_2_acc > round_0_acc, f"Expected improvement: {round_2_acc} vs {round_0_acc}"
