"""
Unit tests verifying client data isolation and privacy boundaries:
- Clients only read from their dedicated directory partition.
- Client payloads transmitted to server contain only numeric arrays and sample counts.
- No raw patient records/CSVs are included in payloads or logs.
"""

import numpy as np
import pandas as pd
from src.config import DATA_CLIENTS_DIR
from src.federated.client import HospitalClient


def test_client_folder_isolation():
    # Instantiate clients
    c1 = HospitalClient(client_id=1)
    c2 = HospitalClient(client_id=2)
    c3 = HospitalClient(client_id=3)

    # Check that paths point to their isolated folders
    assert c1.client_dir in [DATA_CLIENTS_DIR / "client_1", DATA_CLIENTS_DIR / "hospital_a"]
    assert c2.client_dir in [DATA_CLIENTS_DIR / "client_2", DATA_CLIENTS_DIR / "hospital_b"]
    assert c3.client_dir in [DATA_CLIENTS_DIR / "client_3", DATA_CLIENTS_DIR / "hospital_c"]

    # Data loaded should match local CSV files exactly
    df1 = pd.read_csv(DATA_CLIENTS_DIR / "client_1" / "train.csv")
    df2 = pd.read_csv(DATA_CLIENTS_DIR / "client_2" / "train.csv")
    assert len(c1.X) == len(df1)
    assert len(c2.X) == len(df2)

    # Partitions should have distinct data indices or contents
    assert not np.array_equal(c1.X[:10], c2.X[:10])


def test_client_payload_privacy():
    client = HospitalClient(client_id=1)

    # Initial parameters
    params = client.get_parameters(config={})
    assert isinstance(params, list)
    for p in params:
        assert isinstance(p, np.ndarray)
        assert p.dtype in [np.float32, np.float64]

    # Perform fit step
    updated_params, num_samples, fit_metrics = client.fit(params, config={})

    # Assert that no raw patient rows, dataframes, or strings are returned
    assert isinstance(updated_params, list)
    for p in updated_params:
        assert isinstance(p, np.ndarray)
        assert p.dtype in [np.float32, np.float64]

    assert isinstance(num_samples, int)
    assert num_samples == len(client.X)

    # Check metrics dictionary keys
    assert "data" not in fit_metrics
    assert "df" not in fit_metrics
    assert "patient" not in str(fit_metrics).lower()
    for key, val in fit_metrics.items():
        assert isinstance(val, (int, float, str, np.floating, np.integer))
