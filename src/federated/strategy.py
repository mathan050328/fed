"""
Custom Federated Averaging (FedAvg) Strategy and weighted aggregation functions.
Tracks round-by-round evaluation metrics on held-out test data and logs them to round_metrics.csv.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.config import ROUND_METRICS_PATH


def aggregate_fedavg(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Standard sample-count-weighted Federated Averaging (FedAvg) algorithm.
    Weighted average of parameters:
    W = sum(n_k * W_k) / sum(n_k)
    """
    if not results:
        raise ValueError("Cannot aggregate empty list of client results.")

    total_samples = sum(num_samples for _, num_samples in results)
    if total_samples <= 0:
        raise ValueError(f"Total samples must be positive, got {total_samples}")

    # Initialize weighted accumulation with zeros matching first client's parameter shapes
    first_params, _ = results[0]
    weighted_weights: List[np.ndarray] = [
        np.zeros_like(layer, dtype=np.float64) for layer in first_params
    ]

    # Compute weighted sum
    for params, num_samples in results:
        weight_factor = float(num_samples) / float(total_samples)
        for i, layer in enumerate(params):
            weighted_weights[i] += layer.astype(np.float64) * weight_factor

    return weighted_weights


class MetricsTrackingFedAvg(FedAvg):
    """
    Flower FedAvg strategy extended to log global test metrics round-by-round to CSV.
    Stores the final aggregated parameters for model saving.
    """

    def __init__(
        self,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        min_fit_clients: int = 3,
        min_available_clients: int = 3,
        initial_parameters: Optional[Parameters] = None,
        **kwargs,
    ):
        super().__init__(
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            **kwargs,
        )
        self.round_history: List[Dict[str, Any]] = []
        self.latest_parameters: Optional[NDArrays] = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters else None
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates client parameters using sample-count weighted FedAvg.
        """
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            self.latest_parameters = parameters_to_ndarrays(aggregated_parameters)

        return aggregated_parameters, metrics

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluates the aggregated parameters using the centralized evaluate_fn,
        recording round metrics to round_metrics.csv.
        """
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is not None and self.evaluate_fn is not None:
            loss, metrics = eval_res
            entry = {
                "round": server_round,
                "loss": round(float(loss), 4),
                "accuracy": round(float(metrics.get("accuracy", 0.0)), 4),
                "precision": round(float(metrics.get("precision", 0.0)), 4),
                "recall": round(float(metrics.get("recall", 0.0)), 4),
                "f1": round(float(metrics.get("f1", 0.0)), 4),
                "roc_auc": round(float(metrics.get("roc_auc", 0.0)), 4),
            }
            self.round_history.append(entry)
            self.save_round_history()

        return eval_res

    def save_round_history(self) -> None:
        """Persists round history to CSV."""
        if self.round_history:
            df = pd.DataFrame(self.round_history)
            ROUND_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(ROUND_METRICS_PATH, index=False)
