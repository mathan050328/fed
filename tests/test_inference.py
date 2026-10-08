"""
Integration test for Streamlit inference pipeline with saved model artifacts.
"""

import numpy as np
import pandas as pd
import joblib
from src.config import FEDERATED_MODEL_PATH, PREPROCESSOR_PATH


def test_inference_pipeline():
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor not found at {PREPROCESSOR_PATH}"
    assert FEDERATED_MODEL_PATH.exists(), f"Federated model not found at {FEDERATED_MODEL_PATH}"

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(FEDERATED_MODEL_PATH)

    # Test patient data payload matching Streamlit inputs
    sample_patient = {
        "age": 62,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    df = pd.DataFrame([sample_patient])
    X_transformed = preprocessor.transform(df)

    assert X_transformed.shape[0] == 1
    assert not np.isnan(X_transformed).any()

    probabilities = model.predict_proba(X_transformed)
    prediction = model.predict(X_transformed)

    assert probabilities.shape == (1, 2)
    assert 0.0 <= probabilities[0, 1] <= 1.0
    assert prediction[0] in [0, 1]
    assert np.isclose(probabilities[0, 0] + probabilities[0, 1], 1.0)
