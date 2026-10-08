# Dataset Documentation: Heart Disease Risk Classification

## 1. Overview & Provenance
- **Dataset Name:** UCI Heart Disease Dataset (Cleveland Database)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) / Public de-identified clinical repository.
- **Original Curators:**
  - Robert Detrano, M.D., Ph.D. (Cleveland Clinic Foundation)
  - Jeróme Pfisterer, M.D. (University Hospital, Basel, Switzerland)
  - Andras Janosi, M.D. (Hungarian Institute of Cardiology, Budapest)
  - William Steinbrunn, M.D. (University Hospital, Zurich, Switzerland)
- **License / Usage Conditions:** Open access public domain for educational and academic research. All patient records are fully de-identified and anonymized.

## 2. Target Variable
- **Target Column:** `target` (originally `num` in Cleveland data)
- **Classification Task:** Binary Risk Classification
  - `0`: Absence of heart disease (< 50% diameter narrowing in major vessel)
  - `1`: Presence of heart disease (> 50% diameter narrowing in major vessel)

## 3. Feature Descriptions and Schema

| Feature Name | Type | Unit / Allowed Range | Description | Clinical Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Numeric | 29 – 77 years | Age of the patient | Key demographic cardiovascular risk factor |
| `sex` | Categorical | 0, 1 | Biological sex (`1` = male, `0` = female) | Cardiovascular incidence varies by sex |
| `cp` | Categorical | 0, 1, 2, 3 | Chest Pain Type | 0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic |
| `trestbps` | Numeric | 94 – 200 mm Hg | Resting blood pressure on hospital admission | Systemic hypertension indicator |
| `chol` | Numeric | 126 – 564 mg/dl | Serum cholesterol in mg/dl | Atherosclerosis / lipid profile biomarker |
| `fbs` | Categorical | 0, 1 | Fasting blood sugar > 120 mg/dl | `1` = true, `0` = false (indicator of diabetic metabolic state) |
| `restecg` | Categorical | 0, 1, 2 | Resting electrocardiographic results | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| `thalach` | Numeric | 71 – 202 bpm | Maximum heart rate achieved during exercise | Chronotropic response / exercise capacity |
| `exang` | Categorical | 0, 1 | Exercise induced angina | `1` = yes, `0` = no (ischemic symptom trigger) |
| `oldpeak` | Numeric | 0.0 – 6.2 mm | ST depression induced by exercise relative to rest | Electrocardiographic stress marker for myocardial ischemia |
| `slope` | Categorical | 0, 1, 2 | Slope of the peak exercise ST segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| `ca` | Categorical | 0, 1, 2, 3 | Number of major vessels (0–3) colored by fluoroscopy | Anatomical severity of coronary artery disease |
| `thal` | Categorical | 1, 2, 3 | Thalassemia / perfusion scintigraphy defect | 1: Normal, 2: Fixed defect, 3: Reversible defect |

## 4. Preprocessing & Leakage Prevention Decisions
1. **Fit-Only-On-Train Strategy:**
   - Preprocessing transformers (imputers, scalers, one-hot encoders) are fit strictly on `X_train` (80% split).
   - Test data and inference data are transformed using the fitted preprocessor without updating parameters, preventing data leakage.
2. **Missing Value Handling:**
   - Numerical attributes are imputed using the median of the training distribution.
   - Categorical attributes are imputed using the most frequent category.
3. **Feature Scaling & Encoding:**
   - Continuous numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are standardized using `StandardScaler` ($\mu=0, \sigma=1$).
   - Categorical features are encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
4. **Client Data Partitioning:**
   - The processed training set is partitioned across 3 simulated hospital clients (`data/clients/client_{1,2,3}/`).
   - The test set (`data/processed/test.csv`) is held out exclusively on the central evaluation environment and is never distributed to clients.
