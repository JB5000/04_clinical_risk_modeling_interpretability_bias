#!/usr/bin/env python3
"""Main training script for clinical risk models."""
import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

from src.data.preprocessing import DataPreprocessor
from src.models.clinical_model import ClinicalRiskModel
from src.interpretability.explainer import SHAPExplainer
from src.fairness.auditor import FairnessAuditor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load clinical data from CSV."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns")
    return df


def train_model(config: dict, data_path: str, experiment_name: str = "clinical_risk"):
    """Train clinical risk model with MLflow tracking."""
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load data
    df = load_data(data_path)
    
    # Separate features and target
    if 'target' not in df.columns and 'high_risk' not in df.columns:
        raise ValueError("Data must contain 'target' or 'high_risk' column")
    
    target_col = 'high_risk' if 'high_risk' in df.columns else 'target'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train-test split
    test_size = config['training']['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=config['model']['random_state'],
        stratify=y if config['training']['stratify'] else None
    )
    
    # Preprocessing
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Train models
    model_type = config['model']['default_type']
    logger.info(f"Training {model_type} model...")
    
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_params(config['model'])
        mlflow.log_params({f"{model_type}_{k}": v for k, v in config[model_type].items()})
        
        # Train model
        model = ClinicalRiskModel(model_type=model_type)
        model.fit(X_train_proc, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)
        
        # Metrics
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"Test AUC: {auc:.4f}")
        
        # Log metrics
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", (y_pred == y_test).mean())
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metrics({
            "precision": report['1']['precision'],
            "recall": report['1']['recall'],
            "f1_score": report['1']['f1-score']
        })
        
        # Cross-validation
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(
            model.model, X_train_proc, y_train,
            cv=config['training']['cv_folds'],
            scoring='roc_auc'
        )
        mlflow.log_metric("cv_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_auc_std", cv_scores.std())
        logger.info(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # SHAP explanations
        logger.info("Computing SHAP explanations...")
        explainer = SHAPExplainer(model, X_train_proc.sample(min(100, len(X_train_proc))))
        global_importance = explainer.global_importance(X_test_proc.sample(min(200, len(X_test_proc))))
        
        # Log top features
        top_features = list(global_importance.items())[:10]
        for i, (feat, importance) in enumerate(top_features, 1):
            mlflow.log_metric(f"feature_importance_{i}_{feat}", importance)
        
        # Fairness audit
        logger.info("Conducting fairness audit...")
        protected_attrs = config.get('protected_attributes', ['gender', 'race'])
        auditor = FairnessAuditor(protected_attributes=protected_attrs)
        
        sensitive_features = {}
        for attr in protected_attrs:
            if attr in X_test.columns:
                sensitive_features[attr] = X_test[attr].values
        
        if sensitive_features:
            fairness_metrics = auditor.compute_metrics(
                y_test.values, y_pred, y_proba, sensitive_features
            )
            
            # Log fairness metrics
            for attr, metrics in fairness_metrics.items():
                mlflow.log_metric(f"fairness_{attr}_dp", metrics['demographic_parity'])
                mlflow.log_metric(f"fairness_{attr}_eo", metrics['equal_opportunity'])
            
            passes_fairness = auditor.passes_fairness_threshold(fairness_metrics)
            mlflow.log_metric("passes_fairness_threshold", int(passes_fairness))
            logger.info(f"Fairness check: {'PASS' if passes_fairness else 'FAIL'}")
        
        # Save model
        model_path = f"models/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        Path("models").mkdir(exist_ok=True)
        model.save(model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model.model, "model")
        
        logger.info("Training completed successfully!")
        
        return model, auc, fairness_metrics if sensitive_features else None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train clinical risk model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--experiment', type=str, default='clinical_risk',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train model
    model, auc, fairness = train_model(config, args.data, args.experiment)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"AUC: {auc:.4f}")
    if fairness:
        print(f"Fairness metrics computed for: {list(fairness.keys())}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
