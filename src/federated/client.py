"""
Flower NumPyClient implementation for simulated hospital clients.
Supports multi-disease local training (Heart Disease, Diabetes, CKD)
strictly within isolated hospital databases without raw record transmission.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import flwr as fl

from src.config import (
    DATA_CLIENTS_DIR,
    HOSPITAL_IDS,
    LOCAL_EPOCHS,
    SEED,
    TARGET_COLUMN,
)
from src.models.logistic_model import create_logistic_model, evaluate_logistic_model
from src.utils.serialization import get_model_parameters, set_model_parameters


class HospitalClient(fl.client.NumPyClient):
    """
    Simulated Hospital Federated Client.
    Maintains local training data for Heart Disease, Diabetes, and CKD in isolation.
    """

    def __init__(
        self,
        client_id: Union[int, str] = 1,
        disease: str = "heart",
        local_epochs: int = LOCAL_EPOCHS,
    ):
        # Resolve hospital identifier
        if isinstance(client_id, str) and client_id in HOSPITAL_IDS:
            self.hospital_name = client_id
            self.client_id = HOSPITAL_IDS.index(client_id) + 1
        else:
            self.client_id = int(client_id)
            idx = (self.client_id - 1) % len(HOSPITAL_IDS)
            self.hospital_name = HOSPITAL_IDS[idx]

        self.disease = disease.lower()
        self.local_epochs = local_epochs

        # Isolated local database path
        self.client_dir = DATA_CLIENTS_DIR / self.hospital_name

        # Prefer {disease}_train.csv, fallback to train.csv
        data_file = self.client_dir / f"{self.disease}_train.csv"
        if not data_file.exists():
            data_file = self.client_dir / "train.csv"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Hospital {self.hospital_name} data missing for {self.disease} at {data_file}."
            )

        self.data_path = data_file
        df = pd.read_csv(self.data_path)
        self.y = df[TARGET_COLUMN].values.astype(int)
        self.X = df.drop(columns=[TARGET_COLUMN]).values.astype(float)
        self.num_samples = len(self.X)
        self.num_features = self.X.shape[1]

        # Local model
        self.model = create_logistic_model(
            max_iter=self.local_epochs,
            warm_start=True,
            random_state=SEED + self.client_id,
        )

        init_coef = np.zeros((1, self.num_features), dtype=np.float64)
        init_intercept = np.zeros(1, dtype=np.float64)
        set_model_parameters(self.model, [init_coef, init_intercept], n_features=self.num_features)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns model weights and bias. Never transmits raw records."""
        return get_model_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Sets received global weights, trains locally on internal database, returns updated weights."""
        set_model_parameters(self.model, parameters, n_features=self.num_features)
        self.model.fit(self.X, self.y)
        metrics, _, _ = evaluate_logistic_model(self.model, self.X, self.y)
        updated_parameters = get_model_parameters(self.model)

        fit_metrics = {
            "hospital": self.hospital_name,
            "disease": self.disease,
            "samples": self.num_samples,
            "local_accuracy": metrics["accuracy"],
            "local_f1": metrics["f1"],
        }
        if metrics.get("log_loss") is not None:
            fit_metrics["local_loss"] = metrics["log_loss"]

        return updated_parameters, self.num_samples, fit_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluates global weights on local hospital partition."""
        set_model_parameters(self.model, parameters, n_features=self.num_features)
        metrics, _, _ = evaluate_logistic_model(self.model, self.X, self.y)
        loss = metrics.get("log_loss") or (1.0 - metrics["accuracy"])
        eval_metrics = {
            "hospital": self.hospital_name,
            "disease": self.disease,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
        }
        return float(loss), self.num_samples, eval_metrics


def create_client(client_id: int, disease: str = "heart") -> fl.client.Client:
    return HospitalClient(client_id=client_id, disease=disease).to_client()
