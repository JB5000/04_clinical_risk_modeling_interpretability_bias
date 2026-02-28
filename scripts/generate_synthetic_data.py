#!/usr/bin/env python3
"""Generate synthetic clinical data for testing."""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_data(
    n_samples: int = 10000,
    output_path: str = "data/synthetic_clinical_data.csv",
    random_seed: int = 42
):
    """Generate synthetic clinical dataset.
    
    Args:
        n_samples: Number of patients to generate
        output_path: Path to save CSV file
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    print(f"Generating {n_samples} synthetic patient records...")
    
    # Generate features
    data = {
        'patient_id': [f'P{i:08d}' for i in range(n_samples)],
        'age': np.random.randint(18, 95, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(
            ['White', 'Black', 'Asian', 'Hispanic', 'Other'],
            n_samples,
            p=[0.60, 0.13, 0.06, 0.18, 0.03]
        ),
        'bmi': np.clip(np.random.normal(28, 6.5, n_samples), 15, 50),
        'systolic_bp': np.clip(np.random.normal(130, 18, n_samples), 80, 220),
        'diastolic_bp': np.clip(np.random.normal(85, 11, n_samples), 50, 130),
        'glucose': np.clip(np.random.normal(110, 35, n_samples), 60, 400),
        'cholesterol': np.clip(np.random.normal(205, 42, n_samples), 120, 400),
        'smoking_status': np.random.choice(
            ['Never', 'Former', 'Current'],
            n_samples,
            p=[0.55, 0.25, 0.20]
        ),
        'diabetes': np.random.choice([True, False], n_samples, p=[0.12, 0.88]),
        'hypertension': np.random.choice([True, False], n_samples, p=[0.32, 0.68]),
        'heart_disease': np.random.choice([True, False], n_samples, p=[0.09, 0.91]),
        'prior_stroke': np.random.choice([True, False], n_samples, p=[0.04, 0.96]),
    }
    
    # Generate timestamps
    base_date = datetime(2024, 1, 1)
    data['visit_timestamp'] = [
        base_date + timedelta(days=int(np.random.randint(0, 365)))
        for _ in range(n_samples)
    ]
    
    df = pd.DataFrame(data)
    
    # Generate realistic target variable
    # Risk score based on clinical factors
    risk_score = (
        0.015 * df['age'] +
        0.25 * df['diabetes'].astype(int) +
        0.20 * df['heart_disease'].astype(int) +
        0.35 * df['prior_stroke'].astype(int) +
        0.008 * (df['systolic_bp'] - 120).clip(0, None) +
        0.006 * (df['glucose'] - 100).clip(0, None) +
        0.004 * (df['bmi'] - 25).clip(0, None) +
        0.15 * (df['smoking_status'] == 'Current').astype(int) +
        np.random.normal(0, 0.4, n_samples)
    )
    
    # Convert to binary outcome (threshold at 1.5)
    df['high_risk'] = (risk_score > 1.5).astype(int)
    
    # Add some realistic missingness (5%)
    for col in ['bmi', 'glucose', 'cholesterol']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  High-risk patients: {df['high_risk'].sum()} ({df['high_risk'].mean()*100:.1f}%)")
    print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to {output_path}")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate synthetic clinical data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/synthetic_clinical_data.csv',
                       help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    generate_synthetic_data(args.samples, args.output, args.seed)


if __name__ == "__main__":
    main()
