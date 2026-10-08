# Final-Year Project Presentation Guide: Federated Learning-Based Disease Prediction System

A presentation guide accompanying the generated PowerPoint deck:  
📁 **File:** [`Federated_Disease_Prediction_Presentation.pptx`](file:///c:/Users/asus/OneDrive/Desktop/fed/Federated_Disease_Prediction_Presentation.pptx)

---

## Slide-by-Slide Outline & Speaking Script

### Slide 1: Title Slide (Dark Hero Background)
- **Title:** Federated Learning-Based Disease Prediction System
- **Subtitle:** Privacy-Preserving Collaborative Machine Learning for Cardiovascular Risk Assessment
- **Tech Stack:** Python 3.11 | Scikit-Learn | Flower (flwr) | Streamlit | FedAvg
- **Speaker Notes:**
  > *"Good morning respected evaluators and committee members. Today I am presenting our capstone project: a Federated Learning-Based Disease Prediction System. We investigate how healthcare institutions can collaboratively build high-accuracy clinical prediction models while maintaining strict data privacy and local record custody."*

---

### Slide 2: Problem Statement: The Healthcare Data Paradox
- **Left Panel (Centralized ML Dilemma):** Data silos, HIPAA/GDPR legal prohibitions on sharing raw medical records, central data warehouses representing vulnerable single points of failure.
- **Right Panel (Federated Solution):** Keeping raw patient data strictly inside hospital client boundaries; training decentralized local models; exchanging only mathematical parameter updates.
- **Speaker Notes:**
  > *"Healthcare machine learning faces a fundamental paradox: models need diverse multi-center data to achieve high clinical generalization, but regulations like HIPAA and GDPR strictly prohibit centralizing raw patient files. Federated learning resolves this paradox by bringing the computation to the hospital data, rather than moving patient data to a central server."*

---

### Slide 3: System Architecture & Federated Workflow
- **Four Core Stages:**
  1. *Data Isolation:* 3 simulated hospital partitions (`client_1`, `client_2`, `client_3`).
  2. *Local Training:* Hospital nodes train local Logistic Regression models for 5 local epochs.
  3. *Parameter Uplink:* Transmission of parameter arrays $(\mathbf{w}, b)$ and patient count $n_k$. No raw CSV rows or identifiers are sent.
  4. *Server Aggregation:* Central Flower server executes sample-count-weighted Federated Averaging (FedAvg).
- **Speaker Notes:**
  > *"Our architecture features complete separation of concerns. Notice that the server coordinates aggregation and global evaluation without having direct read access to client filesystem partitions. The server only receives abstract weight arrays."*

---

### Slide 4: Clinical Dataset & Preprocessing Pipeline
- **Dataset:** UCI Cleveland Heart Disease Cohort (303 records, 13 clinical biomarkers).
- **Partitioning:** 80% Training (242 samples) partitioned across 3 hospital clients; 20% Held-out Test (61 samples) reserved strictly for server evaluation.
- **Leakage-Free Preprocessing:**
  - `ColumnTransformer` fitted *strictly* on training records.
  - Continuous variables: Median imputation + `StandardScaler` ($Z$-score).
  - Categorical variables: Mode imputation + `OneHotEncoder(handle_unknown='ignore')` (30 features total).
  - Persisted as `preprocessor.joblib` and reused identically in Streamlit.
- **Speaker Notes:**
  > *"To ensure rigorous science, we implemented a leakage-free preprocessing pipeline. The preprocessor is fit solely on the 242 training records and saved as an artifact. The 61 held-out test records are never seen during preprocessor fitting or client optimization."*

---

### Slide 5: Mathematical Foundations: Federated Averaging (FedAvg)
- **Core Formulation:**
  $$W_{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_{t+1}^k$$
- **Rationale for Sample Weighting:** Ensures that client influence is directly proportional to its patient volume $n_k$, preventing statistical distortion from small clinics.
- **Local Optimization:** Warm-started optimization from the incoming global parameter vector $W_t$.
- **Speaker Notes:**
  > *"In FedAvg, rather than computing an unweighted arithmetic mean, each client's weight contribution is weighted by its sample count $n_k$ over total samples $N$. This matches the global loss minimization objective of pooled training."*

---

### Slide 6: Controlled Experimental Protocol
- **Controlled Comparison Table:**
  - *Identical Feature Space:* 30 one-hot transformed clinical features.
  - *Identical Test Benchmark:* 61 held-out patient samples.
  - *Identical Model Architecture:* Logistic Regression.
  - *Independent Variable:* Centralized data pooling vs. Federated parameter exchange.
- **Speaker Notes:**
  > *"To validate our system, we designed a controlled experiment. Both centralized and federated models share the exact same model architecture and test set. The only difference is the data custody model."*

---

### Slide 7: Experimental Results & Comparative Analysis
- **Key Findings Table (Actual System Run):**
  - **Accuracy:** Centralized `86.89%` vs Federated `85.25%` (Difference: $-1.64\%$)
  - **Precision:** Centralized `85.71%` vs Federated `85.29%` (Difference: $-0.42\%$)
  - **Recall:** Centralized `90.91%` vs Federated `87.88%` (Difference: $-3.03\%$)
  - **F1-Score:** Centralized `88.24%` vs Federated `86.57%` (Difference: $-1.67\%$)
  - **ROC-AUC:** Centralized `91.02%` vs Federated `89.50%` (Difference: $-1.52\%$)
- **Visual Chart:** Embedded `metric_comparison.png`.
- **Speaker Notes:**
  > *"These are actual empirical results from our runs. Centralized pooling achieved 86.89% accuracy and 91.02% ROC-AUC. Federated learning achieved 85.25% accuracy and 89.50% ROC-AUC. In exchange for a negligible ~1.5% performance delta, the system achieves 100% data decentralization with zero raw records shared."*

---

### Slide 8: Federated Training Convergence Dynamics
- **Visual Charts:** Embedded `accuracy_by_round.png` and `loss_by_round.png`.
- **Key Takeaways:**
  - Fast convergence: From baseline 54.1% (Round 0) to 85.25% by Round 1.
  - Global loss decreases monotonically from 0.6931 to 0.4072.
  - Asymptotic parameter stability across simulated hospital nodes.
- **Speaker Notes:**
  > *"Looking at the communication dynamics, the global model converges rapidly. By Round 1, the model reaches 85.25% accuracy, and the loss stabilizes smoothly across rounds, showing robust convergence without divergence."*

---

### Slide 9: Diagnostic Classification Analysis
- **Visual Chart:** Embedded `confusion_matrix.png`.
- **Clinical Breakdown:**
  - *High Sensitivity (87.88%):* Correctly detects 29 out of 33 diseased cases. Essential in cardiovascular screening to minimize lethal false negatives.
  - *High Precision (85.29%):* Minimizes unnecessary invasive angiographies.
  - *Total Errors:* Only 4 false negatives and 5 false positives across 61 unseen patients.
- **Speaker Notes:**
  > *"In cardiology, sensitivity is the most critical diagnostic metric. Our federated model achieves 87.88% recall, identifying 29 out of 33 cardiac patients. The diagnostic profile closely mirrors the centralized model."*

---

### Slide 10: Interactive Streamlit Web Dashboard
- **Features Overview:**
  - 🏠 *Home:* Architectural workflow and privacy principles.
  - 🩺 *Risk Prediction:* Live inference using saved preprocessor and federated global model.
  - 🔄 *Federated Analytics:* Communication round metrics and convergence plots.
  - 📊 *Model Comparison:* Side-by-side metric tables and confusion heatmaps.
  - 🔒 *Privacy & Security:* Technical breakdown of isolation guarantees.
  - ℹ️ *About & Limitations:* Academic prototype disclaimer and dataset provenance.
- **Speaker Notes:**
  > *"We packaged the entire solution into a production-ready Streamlit dashboard. Clinicians can input 13 physiological parameters and obtain real-time risk scores with appropriate medical disclaimers."*

---

### Slide 11: Privacy Boundaries, Threat Models & Roadmap
- **What is Guaranteed:**
  - Zero raw CSVs or records sent over network RPC.
  - Client payloads contain only `float64` weight arrays and sample count integers.
  - Verified by unit test `test_client_payload_privacy`.
- **What is NOT Guaranteed by FL Alone:**
  - Model weights can leak subtle gradients under adversarial white-box attacks.
- **Future Enhancements:**
  - Differential Privacy (DP-SGD) with $(\epsilon, \delta)$ privacy budget.
  - Secure Aggregation (SecAgg) via cryptographic multi-party computation.
- **Speaker Notes:**
  > *"A key contribution of our thesis is acknowledging that FL alone does not mean absolute privacy. In threat models with malicious servers, weight updates could be subject to gradient reconstruction attacks. Therefore, production deployments should incorporate Differential Privacy and Secure Aggregation."*

---

### Slide 12: Conclusion & Q&A
- **Summary:** Feasibility of collaborative medical AI proved, minimal utility trade-off (~1.5%), fully reproducible, 12 passing tests.
- **Disclaimer:** Final-year academic prototype; not certified for clinical patient management.
- **Speaker Notes:**
  > *"To conclude, our prototype proves that privacy-preserving collaborative healthcare AI is technically practical and effective. Thank you for your time, and we welcome your questions."*

---

## 🎯 Anticipated Examiner Questions & Model Answers

**Q1: Why choose Logistic Regression over a Deep Neural Network?**
> *Answer:* Logistic Regression provides convex optimization guarantees, clear parameter interpretability, and robust performance on small-to-medium clinical cohorts ($N=303$) without overfitting. The same FedAvg architecture extends naturally to multi-layer perceptrons or PyTorch neural networks.

**Q2: What happens if client data is Non-IID (different patient distributions)?**
> *Answer:* Non-IID data can cause client updates to drift apart, slowing down convergence. We implemented both IID and Non-IID partitioning in `src/data/split_clients.py`. Algorithms like FedProx (adding a proximal term $\frac{\mu}{2} \|w - w_t\|^2$) or Scaffold can be integrated to mitigate non-IID client drift.

**Q3: How does your system ensure no test data leakage occurred?**
> *Answer:* Preprocessing pipelines (`StandardScaler`, imputers, `OneHotEncoder`) were fit *only* on the 80% training partition (`X_train`). The 20% test partition was transformed using the fitted artifact and kept strictly on the server for post-aggregation evaluation.
