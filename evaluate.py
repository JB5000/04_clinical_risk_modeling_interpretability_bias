#!/usr/bin/env python3
"""Model evaluation script."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.clinical_model import ClinicalRiskModel
from src.data.preprocessing import DataPreprocessor
from src.fairness.auditor import FairnessAuditor


def evaluate_model(model_path: str, data_path: str, output_dir: str = "evaluation_results"):
    """Evaluate trained model on test data."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load model
    print(f"Loading model from {model_path}")
    model = ClinicalRiskModel.load(model_path)
    
    # Load test data
    print(f"Loading test data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    target_col = 'high_risk' if 'high_risk' in df.columns else 'target'
    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_test_proc = preprocessor.fit_transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)
    
    # Compute metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print("\n" + "="*60)
    print("Model Performance Metrics")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{output_dir}/confusion_matrix_{timestamp}.png")
    print(f"\nConfusion matrix saved to {output_dir}/confusion_matrix_{timestamp}.png")
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{output_dir}/roc_curve_{timestamp}.png")
    print(f"ROC curve saved to {output_dir}/roc_curve_{timestamp}.png")
    
    # Classification report
    report = classification_report(y_test, y_pred)
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(report)
    
    # Fairness evaluation
    print("\n" + "="*60)
    print("Fairness Evaluation")
    print("="*60)
    
    protected_attrs = ['gender', 'race']
    auditor = FairnessAuditor(protected_attributes=protected_attrs)
    
    sensitive_features = {}
    for attr in protected_attrs:
        if attr in X_test.columns:
            sensitive_features[attr] = X_test[attr].values
    
    if sensitive_features:
        fairness_metrics = auditor.compute_metrics(
            y_test.values, y_pred, y_proba, sensitive_features
        )
        
        for attr, attr_metrics in fairness_metrics.items():
            print(f"\n{attr.upper()}:")
            print(f"  Demographic Parity: {attr_metrics['demographic_parity']:.4f}")
            print(f"  Equal Opportunity: {attr_metrics['equal_opportunity']:.4f}")
            
        passes = auditor.passes_fairness_threshold(fairness_metrics)
        print(f"\nFairness Check: {'✓ PASS' if passes else '✗ FAIL'}")
        
        # Save fairness metrics
        fairness_output = {
            'metrics': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                           for kk, vv in v.items() if kk != 'subgroups'} 
                       for k, v in fairness_metrics.items()},
            'passes_threshold': passes
        }
        
        with open(f"{output_dir}/fairness_metrics_{timestamp}.json", 'w') as f:
            json.dump(fairness_output, f, indent=2)
    
    # Save all metrics
    metrics_output = {
        'performance': metrics,
        'confusion_matrix': cm.tolist(),
        'timestamp': timestamp
    }
    
    with open(f"{output_dir}/metrics_{timestamp}.json", 'w') as f:
        json.dump(metrics_output, f, indent=2)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to {output_dir}/")
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Evaluate clinical risk model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
