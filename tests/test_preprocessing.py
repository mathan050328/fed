"""
Unit tests for data preprocessing pipeline, missing-value handling, and feature consistency.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.config import (
    RAW_DATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    FEATURE_NAMES,
)
from src.data.preprocess import (
    build_preprocessor_pipeline,
    validate_schema,
    run_preprocessing,
    transform_single_input,
)


def test_schema_validation():
    # Valid dataframe
    data = {col: [1] for col in FEATURE_NAMES}
    data[TARGET_COLUMN] = [1]
    df_valid = pd.DataFrame(data)
    validate_schema(df_valid)  # should not raise

    # Missing target
    df_missing_target = df_valid.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="missing target column"):
        validate_schema(df_missing_target)

    # Missing feature
    df_missing_feat = df_valid.drop(columns=["chol"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df_missing_feat)


def test_preprocessing_imputation_and_consistency():
    num_cols = NUMERICAL_FEATURES
    cat_cols = CATEGORICAL_FEATURES

    # Create dummy data with missing values (NaNs)
    df_train = pd.DataFrame({
        "age": [50.0, np.nan, 60.0],
        "sex": [1, 0, np.nan],
        "cp": [0, 1, 2],
        "trestbps": [120.0, 140.0, np.nan],
        "chol": [np.nan, 220.0, 260.0],
        "fbs": [0, 1, 0],
        "restecg": [0, 1, 0],
        "thalach": [150.0, np.nan, 170.0],
        "exang": [0, 1, 0],
        "oldpeak": [1.0, 2.0, np.nan],
        "slope": [0, 1, 2],
        "ca": [0, 1, 0],
        "thal": [2, 3, 2],
    })

    pipe = build_preprocessor_pipeline(num_cols, cat_cols)
    transformed = pipe.fit_transform(df_train)

    # Verify no NaNs remain after imputation
    assert not np.isnan(transformed).any(), "Transformed output contains NaNs"

    # Verify deterministic output on identical input
    transformed_repeat = pipe.transform(df_train)
    np.testing.assert_array_almost_equal(transformed, transformed_repeat)


def test_single_input_transformation():
    sample_patient = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }
    result = transform_single_input(sample_patient)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1
    assert result.shape[1] > 13  # One-hot expansion
    assert not np.isnan(result).any()
