# DiabetesProject - End-to-End MLOps Pipeline

An end-to-end Machine Learning microservice architecture for diabetes risk prediction. The project features a FastAPI backend serving a trained Scikit-learn model and an interactive Streamlit frontend, orchestrated using Docker Compose.

---

## System Architecture

The application is built on a dual-service containerized architecture:

```
+--------------------------+       HTTP POST       +--------------------------+
|                          |    (JSON Payload)     |                          |
|    Streamlit Frontend    | --------------------> |      FastAPI Backend     |
|   http://localhost:8501  |                       |   http://localhost:8000  |
|                          | <-------------------- |                          |
+--------------------------+    (Risk & Prob %)    +------------+-------------+
                                                                |
                                                                v
                                                     +--------------------+
                                                     | Scikit-learn Model |
                                                     |(diabetes_model.pkl)|
                                                     +--------------------+
```

* **Frontend Container (`diabetes_frontend`):** Streamlit web interface for inputting patient clinical features and visualizing predictions.
* **Backend API Container (`diabetes_api`):** REST API powered by FastAPI, performing data validation using Pydantic models and executing real-time inference.

---

## Tech Stack

* **Language:** Python 3.10
* **Machine Learning:** Scikit-learn, Pandas, Joblib / Pickle
* **API Framework:** FastAPI, Uvicorn, Pydantic
* **Frontend Framework:** Streamlit
* **Containerization:** Docker, Docker Compose
* **Testing:** Pytest, HTTPX

---

## Project Structure

```text
DiabetesProject/
├── venv/                     # Local virtual environment
├── models/
│   └── diabetes_model.pkl    # Serialized ML model artifact
├── src/
│   ├── __init__.py           # Package initialization
│   ├── api.py                # FastAPI REST endpoints and Pydantic schemas
│   ├── app_ui.py             # Streamlit interactive web interface
│   ├── logger.py             # Logging configuration
│   └── train.py              # Machine Learning pipeline training script
├── tests/
│   └── test_api.py           # Pytest unit and integration tests
├── .gitignore                # Git ignore rules
├── conftest.py               # Pytest configuration and fixtures
├── Dockerfile                # Backend container configuration
├── Dockerfile.frontend       # Frontend container configuration
├── docker-compose.yml        # Orchestration configuration for multi-container stack
├── explore_data.py           # Data exploration and analysis script
├── README.md                 # Project documentation
└── requirements.txt          # Frozen Python dependencies
```

---

## Prerequisites

* **Docker Desktop** (with **WSL 2** backend on Windows).
* Hardware virtualization (**SVM Mode** / **VT-x**) enabled in system BIOS.

---

## Quickstart Guide

### 1. Run via Docker Compose

Build and launch the entire dual-container stack with a single command:

```powershell
docker compose up --build
```

### 2. Access the Applications

* **Streamlit Web UI:** Open http://localhost:8501
* **Interactive API Documentation (Swagger UI):** Open http://localhost:8000/docs
* **Backend Health Check:** Open http://localhost:8000/health

### 3. Stop the Application

To shut down all running services:

```powershell
docker compose down
```

---

## API Documentation

### `POST /predict`

Accepts medical parameters and returns classification prediction and confidence probability.

**Request Payload:**

```json
{
  "pregnancies": 2,
  "glucose": 130,
  "blood_pressure": 70,
  "skin_thickness": 20,
  "insulin": 80,
  "bmi": 28.5,
  "diabetes_pedigree": 0.45,
  "age": 35
}
```

**Response (`200 OK`):**

```json
{
  "prediction": 0,
  "probability": 0.23
}
```

---

## Local Development & Model Training

To run or train the project locally outside of Docker:

1. **Activate local virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Explore dataset:**
   ```powershell
   python explore_data.py
   ```

4. **Re-train the ML Model:**
   ```powershell
   python src/train.py
   ```

5. **Run Pytest suite:**
   ```powershell
   pytest
   ```
