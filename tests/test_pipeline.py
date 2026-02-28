"""Test complete pipeline."""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from src.models.clinical_model import ClinicalRiskModel
from src.data.preprocessing import DataPreprocessor
from src.fairness.auditor import FairnessAuditor


def generate_test_data(n=100):
    """Generate small test dataset."""
    return pd.DataFrame({
        'patient_id': [f'P{i}' for i in range(n)],
        'age': np.random.randint(20, 80, n),
        'gender': np.random.choice(['M', 'F'], n),
        'race': np.random.choice(['White', 'Black'], n),
        'bmi': np.random.normal(25, 5, n),
        'systolic_bp': np.random.normal(120, 15, n),
        'diastolic_bp': np.random.normal(80, 10, n),
        'glucose': np.random.normal(100, 20, n),
        'cholesterol': np.random.normal(200, 30, n),
        'smoking_status': np.random.choice(['Never', 'Current'], n),
        'diabetes': np.random.choice([True, False], n),
        'hypertension': np.random.choice([True, False], n),
        'heart_disease': np.random.choice([True, False], n),
        'prior_stroke': np.random.choice([True, False], n),
    })


def test_preprocessing():
    """Test data preprocessing."""
    df = generate_test_data()
    preprocessor = DataPreprocessor()
    
    df_processed = preprocessor.fit_transform(df)
    
    assert 'patient_id_hash' in df_processed.columns or 'patient_id' not in df.columns
    assert df_processed.shape[0] == df.shape[0]
    assert not df_processed.isnull().any().any()


def test_model_training():
    """Test model training."""
    df = generate_test_data(n=200)
    preprocessor = DataPreprocessor()
    
    df_processed = preprocessor.fit_transform(df)
    y = np.random.randint(0, 2, len(df))
    
    model = ClinicalRiskModel(model_type='xgboost')
    model.fit(df_processed, y)
    
    predictions = model.predict_proba(df_processed)
    assert len(predictions) == len(df)
    assert all(0 <= p <= 1 for p in predictions)


def test_fairness_audit():
    """Test fairness auditing."""
    n = 200
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    y_proba = np.random.rand(n)
    
    sensitive_features = {
        'gender': np.random.choice(['M', 'F'], n),
        'race': np.random.choice(['White', 'Black'], n)
    }
    
    auditor = FairnessAuditor(protected_attributes=['gender', 'race'])
    metrics = auditor.compute_metrics(y_true, y_pred, y_proba, sensitive_features)
    
    assert 'gender' in metrics
    assert 'race' in metrics
    assert 'demographic_parity' in metrics['gender']
    assert 'equal_opportunity' in metrics['gender']
