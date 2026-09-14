# Diabetes Risk Prediction Project (End-to-End ML Pipeline & MLOps)

A production-ready Machine Learning system engineered for **Diabetes Risk Prediction** from clinical tabular data. Built with strict MLOps principles and production standards, this project demonstrates data leakage prevention, domain-aware feature processing, clinical decision-threshold optimization, containerization with Docker, and a decoupled REST API + Web UI architecture.

---

## 🌟 Key Engineering & MLOps Highlights

* **Explicit Diabetes Risk Modeling**: Predicts patient diabetes risk based on key physiological metrics (Glucose, Blood Pressure, BMI, Insulin, Age, etc.) using machine learning classification.
* **Zero Data Leakage**: Integrated feature preprocessing directly inside a unified `scikit-learn` `Pipeline` and `ColumnTransformer`. All imputations and scalings are fit strictly on training folds during cross-validation.
* **Domain-Aware Biological Imputation**: Biologically impossible zero values in clinical fields (e.g., Glucose = 0, BMI = 0, Blood Pressure = 0) are treated as missing observations and imputed using median strategies with missingness indicators (`add_indicator=True`).
* **Imbalanced Data & Threshold Tuning**: Addresses class imbalance (~35% positive outcome) using cost-sensitive learning (`class_weight='balanced'`) and automated Precision-Recall curve threshold optimization to maximize clinical sensitivity and $F_1$-score.
* **Full Containerization (Docker & Compose)**: Containerized backend (FastAPI) and frontend (Streamlit) services orchestrated seamlessly via `docker-compose.yml`.
* **Software Engineering Best Practices**: Modular code architecture under `src/`, centralized logging, custom exception handling, and automated unit testing powered by `pytest`.

---

## 🏗 System Architecture

```
                       [ Raw Diabetes Data ]
                                 │
                                 ▼
                   [ Stratified Train/Test Split ]
                                 │
                                 ▼
                     [ Scikit-Learn Pipeline ]
     ├── ColumnTransformer
     │    ├── Zero-to-NaN Conversion & Median Imputation
     │    ├── Missingness Indicators (add_indicator=True)
     │    └── Feature Scaling (StandardScaler)
     └── Classifier (RandomForest / LogisticRegression)
                                 │
                                 ▼
             [ Precision-Recall Decision Optimization ]
                                 │
                                 ▼
             [ Serialized Pipeline Artifact ] (.pkl)
                                 │
      ┌──────────────────────────┴──────────────────────────┐
      ▼                                                     ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│       FastAPI Backend        │          │      Streamlit Web UI        │
│    (REST API Service)        │◄─────────┤   (Interactive Interface)    │
│  [Docker Container - 8000]   │          │  [Docker Container - 8501]   │
└──────────────────────────────┘          └──────────────────────────────┘
```

---

## 📁 Repository Structure

```
DiabetesProject/
├── data/
│   └── diabetes.csv           # Clinical dataset for diabetes risk prediction
├── models/
│   └── diabetes_model.pkl     # Serialized scikit-learn pipeline artifact
├── src/
│   ├── api.py                 # FastAPI backend endpoints
│   ├── app_ui.py              # Streamlit interactive user interface
│   ├── eda.ipynb              # Exploratory Data Analysis notebook
│   ├── logger.py              # Centralized logging configuration
│   ├── train.py               # Model training & threshold tuning script
│   └── utils.py               # Preprocessing and helper functions
├── tests/
│   └── test_api.py            # Automated unit and integration test suite
├── .gitignore                 # Version control exclusion rules
├── conftest.py                # Pytest configuration & fixtures
├── Dockerfile                 # Docker configuration for FastAPI backend
├── Dockerfile.frontend        # Docker configuration for Streamlit frontend
├── docker-compose.yml         # Container orchestration setup
├── requirements.txt           # Python dependency specification
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart & Deployment

### Option A: Running via Docker Compose (Recommended)

The entire application stack (API + Web UI) can be launched in isolated containers with a single command:

```bash
# Build and run backend and frontend services
docker-compose up --build
```

Access the interfaces:
* **Streamlit Web UI**: `http://localhost:8501`
* **FastAPI Swagger Docs**: `http://localhost:8000/docs`

---

### Option B: Local Python Environment

#### 1. Setup Virtual Environment
```bash
git clone https://github.com/GabrielaSmolen/DiabetesProject.git
cd DiabetesProject

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 2. Execute Training Pipeline
Train the model, optimize decision cutoffs, and save the serialized model artifact:
```bash
python src/train.py
```

#### 3. Run Test Suite
Execute unit tests to verify API endpoints and pipeline transformations:
```bash
pytest
```

#### 4. Launch Backend API & Frontend UI
In separate terminal sessions:

* **FastAPI Service**:
  ```bash
  uvicorn src.api:app --reload
  ```

* **Streamlit Dashboard**:
  ```bash
  streamlit run src/app_ui.py
  ```

---

## 📊 Methodology & Clinical Decisions

1. **Handling Biological Artifacts**: In tabular diabetes records, zero values in metrics like `BloodPressure`, `BMI`, or `Glucose` reflect missing measurements rather than real zero values. Replacing them prior to pipeline fitting causes data leakage. The `ColumnTransformer` handles this transformation during pipeline execution.
2. **Decision Cutoff Optimization**: Default $0.5$ decision thresholds fail in high-stakes clinical risk classification. The training pipeline evaluates $F_1$-score trajectories over Precision-Recall curves to select an optimal decision cutoff for diabetes risk identification.

---

## 🛠 Tech Stack

* **Language**: Python 3.10+
* **Machine Learning**: `scikit-learn`, `pandas`, `numpy`
* **Serving & Web**: `FastAPI`, `Streamlit`, `uvicorn`
* **MLOps & DevOps**: `Docker`, `Docker Compose`, `joblib`, `pytest`, `logging`
