"""
Customer Churn Prediction Model
This module contains the machine learning pipeline for predicting customer churn.
I created this to serve the trained model through FastAPI endpoints.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Dict, Any, List


class ChurnPredictor:
    """
    Customer churn prediction model class.
    I designed this to handle the complete ML pipeline from data preprocessing to prediction.
    """
    
    def __init__(self):
        """Initialize the churn predictor with default model."""
        self.model = None
        self.preprocessor = None
        self.numerical_columns = None
        self.categorical_columns = None
        self.feature_names = None
        
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create the preprocessing pipeline.
        I use this to handle both numerical and categorical data consistently.
        """
        # Identify numerical and categorical columns
        self.numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Numerical preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(transformers=[
            ("num", numerical_transformer, self.numerical_columns),
            ("cat", categorical_transformer, self.categorical_columns)
        ])
        
        return preprocessor
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train the churn prediction model.
        I implemented multiple algorithms to choose the best performing one.
        """
        # Create preprocessor
        self.preprocessor = self.create_preprocessor(X)
        
        # Select model based on type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=42)
        elif model_type == "decision_tree":
            self.model = DecisionTreeClassifier(random_state=42)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "xgboost":
            self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create complete pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("model", self.model)
        ])
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store the trained pipeline
        self.model = pipeline
        
        # Get feature names after preprocessing
        self._extract_feature_names(X)
        
        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
    
    def _extract_feature_names(self, X: pd.DataFrame):
        """Extract feature names after preprocessing for API responses."""
        # Fit preprocessor to get feature names
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names from preprocessor
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(self.numerical_columns)
        
        # Add categorical feature names (after one-hot encoding)
        if hasattr(self.preprocessor.named_transformers_['cat'], 'named_steps_'):
            ohe = self.preprocessor.named_transformers_['cat'].named_steps_['onehot']
            if hasattr(ohe, 'get_feature_names_out'):
                cat_features = ohe.get_feature_names_out(self.categorical_columns)
                feature_names.extend(cat_features)
        
        self.feature_names = feature_names
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make churn prediction for new customer data.
        I designed this to handle single predictions through the API.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = self.model.predict(df)[0]
        prediction_proba = self.model.predict_proba(df)[0]
        
        return {
            "prediction": int(prediction),
            "probability": {
                "no_churn": float(prediction_proba[0]),
                "churn": float(prediction_proba[1])
            },
            "confidence": float(max(prediction_proba))
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load a pre-trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = joblib.load(filepath)
        
        # Extract preprocessor from the loaded pipeline
        if hasattr(self.model, 'named_steps_') and 'preprocessor' in self.model.named_steps_:
            self.preprocessor = self.model.named_steps_['preprocessor']
            
            # Extract column information
            if hasattr(self.preprocessor, 'transformers_'):
                self.numerical_columns = self.preprocessor.transformers_[0][2]
                self.categorical_columns = self.preprocessor.transformers_[1][2]
