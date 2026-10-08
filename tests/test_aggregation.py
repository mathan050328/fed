"""
Unit tests for Federated Averaging (FedAvg) sample-count-weighted aggregation.
"""

import numpy as np
import pytest
from src.federated.strategy import aggregate_fedavg


def test_fedavg_synthetic_weighted_average():
    # Client 1: 10 samples, weights = [2.0, 4.0], bias = [1.0]
    c1_params = [np.array([2.0, 4.0]), np.array([1.0])]
    c1_samples = 10

    # Client 2: 30 samples, weights = [6.0, 8.0], bias = [5.0]
    c2_params = [np.array([6.0, 8.0]), np.array([5.0])]
    c2_samples = 30

    # Total samples = 40
    # Expected layer 0: (10*2 + 30*6)/40 = (20 + 180)/40 = 200/40 = 5.0
    #                   (10*4 + 30*8)/40 = (40 + 240)/40 = 280/40 = 7.0
    # Expected layer 1: (10*1 + 30*5)/40 = (10 + 150)/40 = 160/40 = 4.0
    results = [
        (c1_params, c1_samples),
        (c2_params, c2_samples),
    ]

    aggregated = aggregate_fedavg(results)

    assert len(aggregated) == 2
    np.testing.assert_array_almost_equal(aggregated[0], np.array([5.0, 7.0]))
    np.testing.assert_array_almost_equal(aggregated[1], np.array([4.0]))


def test_fedavg_single_client():
    params = [np.array([[1.0, 2.0, 3.0]]), np.array([0.5])]
    results = [(params, 50)]
    aggregated = aggregate_fedavg(results)
    np.testing.assert_array_almost_equal(aggregated[0], params[0])
    np.testing.assert_array_almost_equal(aggregated[1], params[1])


def test_fedavg_empty_input():
    with pytest.raises(ValueError, match="empty list"):
        aggregate_fedavg([])
