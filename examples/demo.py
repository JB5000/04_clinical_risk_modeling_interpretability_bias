"""End-to-end demo of clinical risk modeling system."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from src.models.clinical_model import ClinicalRiskModel
from src.data.preprocessing import DataPreprocessor
from src.interpretability.explainer import SHAPExplainer
from src.fairness.auditor import FairnessAuditor
from src.data.schema import PatientFeatures, RiskPrediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def generate_synthetic_data(n_samples=1000):
    """Generate synthetic clinical data."""
    np.random.seed(42)
    
    data = {
        'patient_id': [f'P{i:06d}' for i in range(n_samples)],
        'age': np.random.randint(18, 90, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
        'bmi': np.random.normal(28, 6, n_samples).clip(18, 45),
        'systolic_bp': np.random.normal(130, 20, n_samples).clip(90, 200),
        'diastolic_bp': np.random.normal(85, 12, n_samples).clip(60, 120),
        'glucose': np.random.normal(110, 30, n_samples).clip(70, 250),
        'cholesterol': np.random.normal(200, 40, n_samples).clip(150, 350),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'diabetes': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
        'hypertension': np.random.choice([True, False], n_samples, p=[0.25, 0.75]),
        'heart_disease': np.random.choice([True, False], n_samples, p=[0.10, 0.90]),
        'prior_stroke': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
        'visit_timestamp': [datetime.now() for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    risk_score = (
        0.02 * df['age'] + 0.3 * df['diabetes'].astype(int) +
        0.25 * df['heart_disease'].astype(int) + 
        0.4 * df['prior_stroke'].astype(int) +
        np.random.normal(0, 0.5, n_samples)
    )
    df['high_risk'] = (risk_score > 1.5).astype(int)
    return df


def main():
    """Run complete demo pipeline."""
    print("=" * 60)
    print("Clinical Risk Modeling - Complete Demo")
    print("=" * 60)
    
    # Generate and preprocess data
    df = generate_synthetic_data(n_samples=1000)
    print(f"\n1. Generated {len(df)} records ({df['high_risk'].mean()*100:.1f}% high-risk)")
    
    X = df.drop(columns=['high_risk'])
    y = df['high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = DataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"2. Preprocessed: {X_train_proc.shape[1]} features")
    
    # Train models
    print("\n3. Training models...")
    models = {}
    for mtype in ['logistic', 'random_forest', 'xgboost']:
        model = ClinicalRiskModel(model_type=mtype)
        model.fit(X_train_proc, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_proc))
        models[mtype] = {'model': model, 'auc': auc}
        print(f"   {mtype}: AUC = {auc:.3f}")
    
    best_type = max(models, key=lambda k: models[k]['auc'])
    best_model = models[best_type]['model']
    print(f"   → Best: {best_type}")
    
    # SHAP explanations
    print("\n4. SHAP explanations...")
    explainer = SHAPExplainer(best_model, X_train_proc.sample(100))
    shap_vals = explainer.explain_instance(X_test_proc.iloc[[0]])
    print("   Top features:", list(sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True))[:3])
    
    # Fairness audit
    print("\n5. Fairness audit...")
    auditor = FairnessAuditor(['gender', 'race'])
    metrics = auditor.compute_metrics(
        y_test.values, 
        best_model.predict(X_test_proc),
        best_model.predict_proba(X_test_proc),
        {'gender': X_test['gender'].values, 'race': X_test['race'].values}
    )
    for attr in metrics:
        print(f"   {attr}: DP={metrics[attr]['demographic_parity']:.3f}")
    
    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
