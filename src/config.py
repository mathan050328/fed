"""
Central configuration module for Multi-Disease Federated Learning System.
Supports Heart Disease, Diabetes, and Chronic Kidney Disease (CKD) across
Hospital A, Hospital B, and Hospital C.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_CLIENTS_DIR = DATA_DIR / "clients"

# Artifacts Paths
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Supported Diseases
DISEASES = ["heart", "diabetes", "ckd"]
DISEASE_DISPLAY_NAMES = {
    "heart": "Heart Disease",
    "diabetes": "Diabetes",
    "ckd": "Chronic Kidney Disease (CKD)",
}

# Hospitals
HOSPITAL_IDS = ["hospital_a", "hospital_b", "hospital_c"]
HOSPITAL_NAMES = {
    "hospital_a": "Hospital A",
    "hospital_b": "Hospital B",
    "hospital_c": "Hospital C",
}

# Reproducibility & Hyperparameters
SEED = 42
NUM_CLIENTS = 3
NUM_ROUNDS = 5
LOCAL_EPOCHS = 5
LEARNING_RATE = 0.05
MAX_ITER = 2000
TEST_SIZE = 0.20

TARGET_COLUMN = "target"

# ==========================================
# DISEASE SCHEMAS & FEATURES
# ==========================================

# 1. Heart Disease Schema
HEART_FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
HEART_NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
HEART_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# 2. Diabetes Schema (Pima Indians / Clinical Diabetes standard)
DIABETES_FEATURE_NAMES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age"
]
DIABETES_NUMERICAL = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age"
]
DIABETES_CATEGORICAL = []

# 3. Chronic Kidney Disease (CKD) Schema (UCI benchmark)
CKD_FEATURE_NAMES = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc",
    "htn", "dm", "cad", "appet", "pe", "ane"
]
CKD_NUMERICAL = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
]
CKD_CATEGORICAL = ["htn", "dm", "cad", "appet", "pe", "ane"]

# Aggregated configurations by disease
DISEASE_CONFIG = {
    "heart": {
        "features": HEART_FEATURE_NAMES,
        "numerical": HEART_NUMERICAL,
        "categorical": HEART_CATEGORICAL,
        "raw_file": DATA_RAW_DIR / "heart.csv",
    },
    "diabetes": {
        "features": DIABETES_FEATURE_NAMES,
        "numerical": DIABETES_NUMERICAL,
        "categorical": DIABETES_CATEGORICAL,
        "raw_file": DATA_RAW_DIR / "diabetes.csv",
    },
    "ckd": {
        "features": CKD_FEATURE_NAMES,
        "numerical": CKD_NUMERICAL,
        "categorical": CKD_CATEGORICAL,
        "raw_file": DATA_RAW_DIR / "ckd.csv",
    },
}

# Legacy Heart paths for backward compatibility
RAW_DATA_PATH = DATA_RAW_DIR / "heart.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"
X_TRAIN_PATH = DATA_PROCESSED_DIR / "X_train.csv"
X_TEST_PATH = DATA_PROCESSED_DIR / "X_test.csv"
Y_TRAIN_PATH = DATA_PROCESSED_DIR / "y_train.csv"
Y_TEST_PATH = DATA_PROCESSED_DIR / "y_test.csv"
CENTRALIZED_MODEL_PATH = MODELS_DIR / "centralized_model.joblib"
FEDERATED_MODEL_PATH = MODELS_DIR / "federated_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CENTRALIZED_METRICS_PATH = METRICS_DIR / "centralized_metrics.json"
FEDERATED_METRICS_PATH = METRICS_DIR / "federated_metrics.json"
ROUND_METRICS_PATH = METRICS_DIR / "round_metrics.csv"
FEATURE_NAMES = HEART_FEATURE_NAMES
NUMERICAL_FEATURES = HEART_NUMERICAL
CATEGORICAL_FEATURES = HEART_CATEGORICAL

FIGURE_ACCURACY_PATH = FIGURES_DIR / "accuracy_by_round.png"
FIGURE_LOSS_PATH = FIGURES_DIR / "loss_by_round.png"
FIGURE_COMPARISON_PATH = FIGURES_DIR / "metric_comparison.png"
FIGURE_CONFUSION_PATH = FIGURES_DIR / "confusion_matrix.png"


def get_disease_paths(disease: str):
    """Returns standardized file paths for a given disease."""
    return {
        "raw": DATA_RAW_DIR / f"{disease}.csv",
        "train": DATA_PROCESSED_DIR / f"{disease}_train.csv",
        "test": DATA_PROCESSED_DIR / f"{disease}_test.csv",
        "X_train": DATA_PROCESSED_DIR / f"{disease}_X_train.csv",
        "X_test": DATA_PROCESSED_DIR / f"{disease}_X_test.csv",
        "y_train": DATA_PROCESSED_DIR / f"{disease}_y_train.csv",
        "y_test": DATA_PROCESSED_DIR / f"{disease}_y_test.csv",
        "preprocessor": MODELS_DIR / f"preprocessor_{disease}.joblib",
        "centralized_model": MODELS_DIR / f"centralized_{disease}.joblib",
        "federated_model": MODELS_DIR / f"federated_{disease}.joblib",
        "centralized_metrics": METRICS_DIR / f"centralized_{disease}_metrics.json",
        "federated_metrics": METRICS_DIR / f"federated_{disease}_metrics.json",
        "round_metrics": METRICS_DIR / f"round_metrics_{disease}.csv",
        "fig_accuracy": FIGURES_DIR / f"{disease}_accuracy_by_round.png",
        "fig_loss": FIGURES_DIR / f"{disease}_loss_by_round.png",
        "fig_comparison": FIGURES_DIR / f"{disease}_metric_comparison.png",
        "fig_confusion": FIGURES_DIR / f"{disease}_confusion_matrix.png",
    }


def ensure_directories():
    """Ensure all required directories and hospital partitions exist."""
    dirs = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_CLIENTS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
    ]
    for h in HOSPITAL_IDS:
        dirs.append(DATA_CLIENTS_DIR / h)
    # Also support client_1, client_2, client_3 aliases
    for i in range(1, NUM_CLIENTS + 1):
        dirs.append(DATA_CLIENTS_DIR / f"client_{i}")

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Multi-disease directories verified.")
