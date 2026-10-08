"""
Unit tests for model parameter extraction, setting, and persistence round-trips.
"""

import tempfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.serialization import (
    get_model_parameters,
    set_model_parameters,
    save_model,
    load_model,
)
from src.models.logistic_model import create_logistic_model


def test_parameter_extraction_and_restoration():
    # Create and train a small model
    X = np.random.randn(40, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model1 = create_logistic_model(max_iter=100, random_state=42)
    model1.fit(X, y)

    params = get_model_parameters(model1)
    assert len(params) == 2  # [coef, intercept]
    assert params[0].shape == (1, 5)
    assert params[1].shape == (1,)

    # Create a fresh empty model and set parameters
    model2 = create_logistic_model(random_state=42)
    set_model_parameters(model2, params, n_features=5)

    # Predictions and probabilities must match exactly
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    proba1 = model1.predict_proba(X)
    proba2 = model2.predict_proba(X)

    np.testing.assert_array_equal(pred1, pred2)
    np.testing.assert_array_almost_equal(proba1, proba2, decimal=6)


def test_model_persistence():
    X = np.random.randn(20, 4)
    y = np.array([0, 1] * 10)
    model = create_logistic_model(max_iter=50)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.joblib"
        save_model(model, model_path)
        assert model_path.exists()

        loaded_model = load_model(model_path)
        preds_orig = model.predict(X)
        preds_loaded = loaded_model.predict(X)
        np.testing.assert_array_equal(preds_orig, preds_loaded)
