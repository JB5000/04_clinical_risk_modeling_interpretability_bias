"""Data preprocessing pipeline."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple
import hashlib


class DataPreprocessor:
    """Preprocesses clinical data with HIPAA compliance."""
    
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = None
        self.numerical_features = ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 
                                    'glucose', 'cholesterol']
        self.categorical_features = ['gender', 'race', 'smoking_status']
        self.boolean_features = ['diabetes', 'hypertension', 'heart_disease', 'prior_stroke']
    
    def anonymize_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for HIPAA compliance."""
        return hashlib.sha256(patient_id.encode()).hexdigest()[:16]
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit preprocessing transformers."""
        # Fit numerical imputer and scaler
        if self.numerical_features:
            self.numerical_imputer.fit(df[self.numerical_features])
            df_imputed = pd.DataFrame(
                self.numerical_imputer.transform(df[self.numerical_features]),
                columns=self.numerical_features
            )
            self.scaler.fit(df_imputed)
        
        # Fit categorical imputer
        if self.categorical_features:
            self.categorical_imputer.fit(df[self.categorical_features])
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data for modeling."""
        df_processed = df.copy()
        
        # Anonymize patient IDs if present
        if 'patient_id' in df_processed.columns:
            df_processed['patient_id_hash'] = df_processed['patient_id'].apply(
                self.anonymize_patient_id
            )
            df_processed = df_processed.drop(columns=['patient_id'])
        
        # Handle numerical features
        if self.numerical_features:
            df_num = pd.DataFrame(
                self.numerical_imputer.transform(df_processed[self.numerical_features]),
                columns=self.numerical_features,
                index=df_processed.index
            )
            df_num_scaled = pd.DataFrame(
                self.scaler.transform(df_num),
                columns=self.numerical_features,
                index=df_processed.index
            )
            for col in self.numerical_features:
                df_processed[col] = df_num_scaled[col]
        
        # Handle categorical features
        if self.categorical_features:
            df_cat = pd.DataFrame(
                self.categorical_imputer.transform(df_processed[self.categorical_features]),
                columns=self.categorical_features,
                index=df_processed.index
            )
            # One-hot encode
            df_processed = pd.get_dummies(df_processed, columns=self.categorical_features, drop_first=True)
        
        # Boolean features to int
        for col in self.boolean_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # Drop timestamp if present
        if 'visit_timestamp' in df_processed.columns:
            df_processed = df_processed.drop(columns=['visit_timestamp'])
        
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
