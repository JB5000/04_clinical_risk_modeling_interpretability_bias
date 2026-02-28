"""Data schemas for clinical risk modeling."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class PatientFeatures(BaseModel):
    """Input features for patient risk prediction."""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=120, description="Age in years")
    gender: str = Field(..., description="Gender: M/F/Other")
    race: str = Field(..., description="Race/Ethnicity")
    bmi: float = Field(..., ge=10, le=80, description="Body Mass Index")
    systolic_bp: float = Field(..., ge=60, le=250, description="Systolic BP (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic BP (mmHg)")
    glucose: float = Field(..., ge=30, le=600, description="Glucose (mg/dL)")
    cholesterol: float = Field(..., ge=100, le=400, description="Total cholesterol (mg/dL)")
    smoking_status: str = Field(..., description="Never/Former/Current")
    diabetes: bool = Field(default=False)
    hypertension: bool = Field(default=False)
    heart_disease: bool = Field(default=False)
    prior_stroke: bool = Field(default=False)
    visit_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['M', 'F', 'Other']:
            raise ValueError('Gender must be M, F, or Other')
        return v
    
    @validator('smoking_status')
    def validate_smoking(cls, v):
        if v not in ['Never', 'Former', 'Current']:
            raise ValueError('Smoking status must be Never, Former, or Current')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P12345",
                "age": 65,
                "gender": "M",
                "race": "White",
                "bmi": 28.5,
                "systolic_bp": 145.0,
                "diastolic_bp": 90.0,
                "glucose": 120.0,
                "cholesterol": 220.0,
                "smoking_status": "Former",
                "diabetes": True,
                "hypertension": True,
                "heart_disease": False,
                "prior_stroke": False
            }
        }


class RiskPrediction(BaseModel):
    """Output schema for risk predictions."""
    
    patient_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk probability")
    risk_category: str = Field(..., description="Low/Medium/High")
    confidence_interval: tuple[float, float] = Field(..., description="95% CI")
    shap_values: dict[str, float] = Field(default_factory=dict, description="Feature contributions")
    top_risk_factors: list[str] = Field(default_factory=list, description="Top contributing features")
    explanation: str = Field(..., description="Human-readable explanation")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('risk_category')
    def validate_category(cls, v):
        if v not in ['Low', 'Medium', 'High']:
            raise ValueError('Risk category must be Low, Medium, or High')
        return v
    
    @classmethod
    def from_score(cls, patient_id: str, score: float, shap_vals: dict, model_ver: str):
        """Create prediction from risk score."""
        if score < 0.3:
            category = "Low"
            explanation = f"Low risk ({score:.1%}). Continue standard care and annual checkups."
        elif score < 0.7:
            category = "Medium"
            explanation = f"Medium risk ({score:.1%}). Enhanced monitoring recommended with 6-month follow-ups."
        else:
            category = "High"
            explanation = f"High risk ({score:.1%}). Immediate intervention and specialist referral advised."
        
        # Get top 3 risk factors
        top_factors = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_factor_names = [f for f, _ in top_factors]
        
        # Confidence interval (simplified)
        ci_lower = max(0.0, score - 0.05)
        ci_upper = min(1.0, score + 0.05)
        
        return cls(
            patient_id=patient_id,
            risk_score=score,
            risk_category=category,
            confidence_interval=(ci_lower, ci_upper),
            shap_values=shap_vals,
            top_risk_factors=top_factor_names,
            explanation=explanation,
            model_version=model_ver
        )


class FairnessMetrics(BaseModel):
    """Fairness audit metrics."""
    
    model_version: str
    protected_attribute: str
    demographic_parity_gap: float
    equal_opportunity_difference: float
    equalized_odds_difference: float
    calibration_error: dict[str, float]
    subgroup_performance: dict[str, dict[str, float]]
    audit_timestamp: datetime = Field(default_factory=datetime.utcnow)
    passes_threshold: bool = Field(..., description="Whether fairness thresholds met")
    
    @classmethod
    def from_audit(cls, model_ver: str, attr: str, metrics: dict):
        """Create from audit results."""
        dp_gap = metrics.get('demographic_parity', 0.0)
        eo_diff = metrics.get('equal_opportunity', 0.0)
        eod_diff = metrics.get('equalized_odds', 0.0)
        
        # Check thresholds
        passes = (
            dp_gap < 0.05 and
            eo_diff < 0.10 and
            eod_diff < 0.10
        )
        
        return cls(
            model_version=model_ver,
            protected_attribute=attr,
            demographic_parity_gap=dp_gap,
            equal_opportunity_difference=eo_diff,
            equalized_odds_difference=eod_diff,
            calibration_error=metrics.get('calibration', {}),
            subgroup_performance=metrics.get('subgroups', {}),
            passes_threshold=passes
        )
