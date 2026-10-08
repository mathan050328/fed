# Federated Learning-Based Disease Prediction System

A privacy-preserving academic prototype for collaborative heart disease risk prediction across simulated healthcare institutions using **Federated Learning (Flower / FedAvg)** and **Scikit-Learn**.

---

## 1. Project Goal & Problem Statement
Healthcare institutions maintain sensitive patient records protected by strict privacy regulations (e.g., HIPAA, GDPR). Traditional machine learning requires aggregating raw clinical data into a central data warehouse, which introduces severe governance, data leakage, and security risks.

This prototype implements **Federated Learning (FedAvg)**:
- Raw patient records remain strictly local to three simulated hospital clients (`data/clients/client_1`, `client_2`, `client_3`).
- Each hospital client trains a local Logistic Regression model on its own records.
- Only numerical parameter updates (weights $\mathbf{w}$ and bias $b$) and sample counts ($n_k$) are shared with the central server.
- The central server aggregates updates using sample-count-weighted Federated Averaging (FedAvg).
- A centralized baseline trained on pooled data is provided for controlled empirical comparison.

---

## 2. Experimental Results (Actual Executed Run)

The controlled experiment evaluated both models on an identical held-out test set ($N=61$, 20% split) with consistent feature transformations:

| Evaluation Metric / Attribute | Centralized ML (Pooled Data) | Federated ML (FedAvg) |
| :--- | :--- | :--- |
| **Raw training records sent to server** | **Yes (All 242 pooled)** | **No (0 records transmitted)** |
| **Training location** | Central server | Each hospital client |
| **Aggregation method** | None (direct fit) | Sample-weighted FedAvg |
| **Accuracy** | **86.89%** (`0.8689`) | **85.25%** (`0.8525`) |
| **Precision** | **85.71%** (`0.8571`) | **85.29%** (`0.8529`) |
| **Recall / Sensitivity** | **90.91%** (`0.9091`) | **87.88%** (`0.8788`) |
| **F1-Score** | **88.24%** (`0.8824`) | **86.57%** (`0.8657`) |
| **ROC-AUC** | **91.02%** (`0.9102`) | **89.50%** (`0.8950`) |

> **Key Finding:** Federated learning achieves nearly identical diagnostic discrimination (ROC-AUC 89.5% vs 91.0%, accuracy 85.3% vs 86.9%) without exposing or centralizing a single patient record.

---

## 3. Repository Structure

```
federated-disease-prediction/
├── data/
│   ├── raw/                  # Downloaded raw Heart Disease dataset
│   ├── processed/            # Clean train/test splits (leakage-free)
│   └── clients/              # Isolated local partitions for simulated hospitals
│       ├── client_1/         # Hospital 1 local partition
│       ├── client_2/         # Hospital 2 local partition
│       └── client_3/         # Hospital 3 local partition
├── src/
│   ├── config.py             # Global constants, seeds, paths, hyperparams
│   ├── data/
│   │   ├── download.py       # Public dataset retrieval with fallback
│   │   ├── preprocess.py     # Leakage-free preprocessing pipeline
│   │   └── split_clients.py  # Data partitioning among hospital clients
│   ├── models/
│   │   └── logistic_model.py # Logistic Regression definition & evaluation
│   ├── federated/
│   │   ├── client.py         # Flower NumPyClient for simulated hospitals
│   │   ├── server.py         # Flower server orchestrator (FedAvg)
│   │   └── strategy.py       # Custom FedAvg tracking round metrics
│   ├── evaluation/
│   │   ├── metrics.py        # Accuracy, Precision, Recall, F1, ROC-AUC
│   │   └── compare.py        # Centralized vs Federated comparison & plotting
│   └── utils/
│       └── serialization.py  # Parameter array extraction & persistence
├── app/
│   └── streamlit_app.py      # Interactive multi-tab web dashboard
├── tests/                    # Comprehensive unit and integration test suite
├── artifacts/
│   ├── models/               # Saved models and fitted preprocessor
│   ├── metrics/              # Centralized, federated, and round JSON/CSV
│   └── figures/              # Generated publication-quality PNG charts
├── requirements.txt          # Frozen dependencies
├── pytest.ini               # Pytest configuration
├── DATASET.md                # Dataset provenance, schema, and preprocessing
└── README.md                 # Project documentation
```

---

## 4. Setup and Installation

### 1. Create and Activate Virtual Environment
```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 5. Step-by-Step Execution Guide

To reproduce all experiments, models, metrics, and figures from scratch, execute the following commands in sequence:

### Step 1: Prepare and Partition Dataset
```bash
python -m src.data.download
python -m src.data.preprocess
python -m src.data.split_clients
```

### Step 2: Train Centralized Baseline Model
```bash
python -m src.evaluation.compare --mode centralized
```
*Outputs: `artifacts/models/centralized_model.joblib` and `artifacts/metrics/centralized_metrics.json`.*

### Step 3: Run Federated Training
```bash
python -m src.federated.server
```
*Outputs: `artifacts/models/federated_model.joblib`, `artifacts/metrics/federated_metrics.json`, and `artifacts/metrics/round_metrics.csv`.*

### Step 4: Evaluate and Generate Figures
```bash
python -m src.evaluation.compare --mode federated
```
*Outputs:*
- `artifacts/figures/accuracy_by_round.png`
- `artifacts/figures/loss_by_round.png`
- `artifacts/figures/metric_comparison.png`
- `artifacts/figures/confusion_matrix.png`

### Step 5: Launch Streamlit Web Dashboard
```bash
streamlit run app/streamlit_app.py
```
Access the interactive dashboard in your web browser at `http://localhost:8501`.

---

## 6. Running Tests

Run the full automated test suite verifying data isolation, preprocessing, model serialization, FedAvg aggregation, and inference:
```bash
pytest tests/ -v
```

All 12 unit and integration tests pass successfully:
- `test_schema_validation`, `test_preprocessing_imputation_and_consistency`, `test_single_input_transformation`
- `test_parameter_extraction_and_restoration`, `test_model_persistence`
- `test_client_folder_isolation`, `test_client_payload_privacy`
- `test_fedavg_synthetic_weighted_average`, `test_fedavg_single_client`, `test_fedavg_empty_input`
- `test_inference_pipeline`
- `test_federated_integration_run`

---

## 7. Privacy and Security Analysis

- **Local Storage:** Client processes only read from their assigned directory partition (`data/clients/client_{id}`).
- **Zero Raw Data Transfer:** Clients send only parameter arrays (`coef_` and `intercept_`) and sample counts. No raw CSV rows or patient identifiers are transmitted.
- **Privacy Boundary Limitation:** While FL eliminates raw data pooling, model updates can theoretically leak subtle statistical signals under white-box gradient inversion or membership inference attacks. Future extensions should incorporate **Differential Privacy (DP-FedAvg)** and **Secure Aggregation (SecAgg)**.

---

## 8. Academic Disclaimer
This system is an **academic engineering prototype** built for educational and research demonstration purposes. It is **not** a certified medical diagnostic device and must not be used for clinical decision-making or real-world patient diagnosis.
