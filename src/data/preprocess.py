"""
Multi-Disease Preprocessing Pipeline.
Processes Heart Disease, Diabetes, and Chronic Kidney Disease (CKD) datasets.
Fits transformers strictly on train subsets and persists fitted preprocessor artifacts.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import (
    DISEASES,
    DISEASE_CONFIG,
    TARGET_COLUMN,
    SEED,
    TEST_SIZE,
    get_disease_paths,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    PREPROCESSOR_PATH,
    FEATURE_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ensure_directories,
)


def validate_schema(df: pd.DataFrame, disease: str = "heart") -> None:
    """Validates presence of required features and target."""
    req_cols = DISEASE_CONFIG[disease]["features"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset for {disease} missing columns: {missing}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset for {disease} missing target column: '{TARGET_COLUMN}'")


def build_preprocessor_pipeline(numerical_cols, categorical_cols):
    """Constructs imputer + scaler + one-hot encoder pipeline."""
    transformers = []
    if numerical_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical_cols))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names_out(preprocessor: ColumnTransformer, numerical_cols, categorical_cols):
    """Extracts transformed column names."""
    names = []
    if numerical_cols:
        names.extend(numerical_cols)
    if categorical_cols and "cat" in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        names.extend(cat_names)
    return names


def preprocess_single_disease(disease: str) -> dict:
    """Preprocesses a single disease dataset and writes artifacts."""
    ensure_directories()
    paths = get_disease_paths(disease)
    cfg = DISEASE_CONFIG[disease]
    raw_path = paths["raw"]

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data missing for {disease} at {raw_path}")

    df = pd.read_csv(raw_path)
    validate_schema(df, disease=disease)

    num_cols = cfg["numerical"]
    cat_cols = cfg["categorical"]
    all_features = num_cols + cat_cols

    X = df[all_features].copy()
    y = df[TARGET_COLUMN].copy().astype(int)

    # Stratified train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Fit pipeline on train only
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    preprocessor.fit(X_train_raw)

    # Transform
    X_train_trans = preprocessor.transform(X_train_raw)
    X_test_trans = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names_out(preprocessor, num_cols, cat_cols)

    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test_raw.index)

    train_df = X_train_df.copy()
    train_df[TARGET_COLUMN] = y_train.values

    test_df = X_test_df.copy()
    test_df[TARGET_COLUMN] = y_test.values

    # Save disease-specific paths
    train_df.to_csv(paths["train"], index=False)
    test_df.to_csv(paths["test"], index=False)
    X_train_df.to_csv(paths["X_train"], index=False)
    X_test_df.to_csv(paths["X_test"], index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(paths["y_train"], index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(paths["y_test"], index=False)
    joblib.dump(preprocessor, paths["preprocessor"])

    # If heart disease, also save to root backward-compatible paths
    if disease == "heart":
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        X_train_df.to_csv(X_TRAIN_PATH, index=False)
        X_test_df.to_csv(X_TEST_PATH, index=False)
        y_train.to_frame(TARGET_COLUMN).to_csv(Y_TRAIN_PATH, index=False)
        y_test.to_frame(TARGET_COLUMN).to_csv(Y_TEST_PATH, index=False)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print(f"[{disease.upper()}] Preprocessed: Train={train_df.shape} | Test={test_df.shape} -> {paths['preprocessor'].name}")
    return {
        "disease": disease,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "features": feature_names,
    }


def preprocess_all_diseases():
    """Runs preprocessing across all 3 diseases."""
    summaries = {}
    for d in DISEASES:
        summaries[d] = preprocess_single_disease(d)
    return summaries


def transform_single_input(input_dict: dict, disease: str = "heart", preprocessor=None) -> np.ndarray:
    """Transforms a single clinical input row."""
    if preprocessor is None:
        paths = get_disease_paths(disease)
        preprocessor = joblib.load(paths["preprocessor"])

    input_df = pd.DataFrame([input_dict])
    return preprocessor.transform(input_df)


# Legacy alias
def run_preprocessing():
    return preprocess_single_disease("heart")


if __name__ == "__main__":
    preprocess_all_diseases()
