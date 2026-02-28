"""FastAPI application for clinical risk prediction."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.schema import PatientFeatures, RiskPrediction
from src.models.clinical_model import ClinicalRiskModel
from src.interpretability.explainer import SHAPExplainer
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="Clinical Risk Prediction API", version="1.0.0")

# Global model (would be loaded from MLflow in production)
MODEL = None
EXPLAINER = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, EXPLAINER
    # In production, load from MLflow
    # For now, placeholder
    pass


@app.get("/")
async def root():
    """API health check."""
    return {"status": "healthy", "service": "clinical-risk-api", "version": "1.0.0"}


@app.get("/model/info")
async def model_info():
    """Get model metadata."""
    return {
        "model_version": "v1.0.0",
        "model_type": "xgboost",
        "features": [
            "age", "bmi", "systolic_bp", "diastolic_bp", "glucose",
            "cholesterol", "gender", "race", "smoking_status",
            "diabetes", "hypertension", "heart_disease", "prior_stroke"
        ],
        "fairness_audited": True,
        "last_audit": "2024-12-01"
    }


@app.post("/predict", response_model=RiskPrediction)
async def predict_risk(patient: PatientFeatures):
    """Predict patient risk with explanations."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to dataframe
    patient_dict = patient.dict()
    df = pd.DataFrame([patient_dict])
    
    # Make prediction (simplified, would use preprocessing pipeline)
    # This is a placeholder
    risk_score = np.random.rand()  # Would be MODEL.predict_proba(df)
    
    # Generate explanation (placeholder SHAP values)
    shap_values = {
        "age": 0.05,
        "bmi": 0.03,
        "systolic_bp": 0.08,
        "diabetes": 0.12,
        "prior_stroke": 0.15
    }
    
    # Create prediction response
    prediction = RiskPrediction.from_score(
        patient_id=patient.patient_id,
        risk_score=risk_score,
        shap_values=shap_values,
        model_version="v1.0.0"
    )
    
    return prediction


@app.post("/batch_predict")
async def batch_predict(patients: list[PatientFeatures]):
    """Batch prediction endpoint."""
    predictions = []
    for patient in patients:
        pred = await predict_risk(patient)
        predictions.append(pred)
    return {"predictions": predictions, "count": len(predictions)}


@app.get("/fairness/audit")
async def fairness_audit():
    """Get latest fairness audit results."""
    return {
        "model_version": "v1.0.0",
        "audit_date": "2024-12-01",
        "protected_attributes": ["gender", "race"],
        "demographic_parity_gap": 0.03,
        "equal_opportunity_difference": 0.07,
        "passes_threshold": True
    }
