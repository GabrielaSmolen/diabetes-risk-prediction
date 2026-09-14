from contextlib import asynccontextmanager
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/diabetes_model.pkl"
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the ML model pipeline at startup and cleans up on shutdown."""
    global model_pipeline
    logger.info("Loading model pipeline from disk...")
    model_pipeline = joblib.load(MODEL_PATH) 

    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Diabetes Prediction API",
    description="Production-grade API for diabetes risk assessment",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientData(BaseModel):
    """Schema for incoming patient medical metrics."""
    pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    glucose: int = Field(..., ge=0, description="Glucose concentration")
    blood_pressure: int = Field(..., ge=0, description="Blood pressure")
    skin_thickness: int = Field(..., ge=0, description="Skin fold thickeness")
    insulin: int = Field(..., ge=0, description="Insulin level")
    bmi: float = Field(..., ge=0, description="Body Mass Index")
    diabetes_pedigree: float = Field(..., ge=0, description="Diabetes pedigree function (genetic history)")
    age: int = Field(..., ge=0, description="Age")


class PredictionOutput(BaseModel):
    """Schema for prediction response."""
    prediction: int
    probability: float


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify API operational status."""
    return {"status": "healthy", "model_loaded": str(model_pipeline is not None)}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientData) -> PredictionOutput:
    """Predicts diabetes outcome for a given patient."""
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    df = pd.DataFrame([patient.model_dump()])

    pred = model_pipeline.predict(df) 
    prob = model_pipeline.predict_proba(df)

    prediction_val = int(pred[0])
    probability_val = float(prob[0][1])

    return PredictionOutput(prediction=prediction_val, probability=probability_val)


@app.get("/")
def read_root():
    return {"message": "Welcome to Diabetes Prediction API. Navigate to /docs for interactive testing."}
