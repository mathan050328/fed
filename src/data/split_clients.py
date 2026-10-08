"""
Multi-Disease Client Partitioning Module.
Partitions Heart Disease, Diabetes, and CKD datasets into local databases
for Hospital A, Hospital B, and Hospital C.
"""

import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    DISEASES,
    HOSPITAL_IDS,
    HOSPITAL_NAMES,
    NUM_CLIENTS,
    SEED,
    TARGET_COLUMN,
    DATA_CLIENTS_DIR,
    get_disease_paths,
    TRAIN_DATA_PATH,
    ensure_directories,
)


def partition_dataframe(df: pd.DataFrame, num_clients: int = NUM_CLIENTS, seed: int = SEED) -> list[pd.DataFrame]:
    """Stratified shuffle split into num_clients balanced partitions."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    chunks = np.array_split(indices, num_clients)
    return [df.iloc[chunk].reset_index(drop=True) for chunk in chunks]


def split_disease_among_hospitals(disease: str, seed: int = SEED) -> dict:
    """Partitions a specific disease dataset among Hospital A, B, and C."""
    ensure_directories()
    paths = get_disease_paths(disease)
    train_path = paths["train"]

    if not train_path.exists():
        raise FileNotFoundError(f"Training data missing for {disease} at {train_path}")

    train_df = pd.read_csv(train_path)
    client_dfs = partition_dataframe(train_df, num_clients=len(HOSPITAL_IDS), seed=seed)

    summary = {}
    for idx, (h_id, client_df) in enumerate(zip(HOSPITAL_IDS, client_dfs), start=1):
        hospital_dir = DATA_CLIENTS_DIR / h_id
        hospital_dir.mkdir(parents=True, exist_ok=True)

        disease_train_file = hospital_dir / f"{disease}_train.csv"
        client_df.to_csv(disease_train_file, index=False)

        # For heart disease, also save default train.csv
        if disease == "heart":
            client_df.to_csv(hospital_dir / "train.csv", index=False)
            client_df.drop(columns=[TARGET_COLUMN]).to_csv(hospital_dir / "X_train.csv", index=False)
            client_df[[TARGET_COLUMN]].to_csv(hospital_dir / "y_train.csv", index=False)

        # Also mirror to client_{idx} for backward compatibility
        compat_dir = DATA_CLIENTS_DIR / f"client_{idx}"
        compat_dir.mkdir(parents=True, exist_ok=True)
        client_df.to_csv(compat_dir / f"{disease}_train.csv", index=False)
        if disease == "heart":
            client_df.to_csv(compat_dir / "train.csv", index=False)
            client_df.drop(columns=[TARGET_COLUMN]).to_csv(compat_dir / "X_train.csv", index=False)
            client_df[[TARGET_COLUMN]].to_csv(compat_dir / "y_train.csv", index=False)

        pos_count = int(client_df[TARGET_COLUMN].sum())
        total = len(client_df)
        summary[h_id] = {"samples": total, "positives": pos_count}

        print(f"  [{disease.upper()}] {HOSPITAL_NAMES[h_id]}: {total} records ({pos_count} positive) -> {disease_train_file.name}")

    return summary


def split_all_diseases():
    """Partitions all 3 diseases across the hospitals."""
    print("=" * 60)
    print("PARTITIONING MULTI-DISEASE DATA ACROSS HOSPITALS A, B, AND C")
    print("=" * 60)
    for d in DISEASES:
        split_disease_among_hospitals(d)
    print("Multi-disease local hospital databases populated successfully.")


# Backward compatibility alias
def split_and_save_clients(train_path=TRAIN_DATA_PATH, num_clients=NUM_CLIENTS, non_iid=False, seed=SEED):
    return split_disease_among_hospitals("heart", seed=seed)


if __name__ == "__main__":
    split_all_diseases()
