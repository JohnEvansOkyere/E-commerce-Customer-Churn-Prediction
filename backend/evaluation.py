"""
Model Evaluation and Performance Monitoring
I created this to monitor model performance in production and detect data drift.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation and monitoring class.
    I designed this to track model performance and detect issues in production.
    """
    
    def __init__(self, model_path: str = "models/trained_model.pkl"):
        """
        Initialize the model evaluator.
        I created this to load the trained model and set up evaluation metrics.
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.metrics_history = []
        self.performance_thresholds = {
            'accuracy': 0.85,  # Minimum acceptable accuracy
            'precision': 0.80,  # Minimum precision
            'recall': 0.75,    # Minimum recall
            'f1_score': 0.77,  # Minimum F1 score
            'auc': 0.80        # Minimum AUC
        }
        
    def load_model(self):
        """Load the trained model and preprocessor."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                if hasattr(self.model, 'named_steps_') and 'preprocessor' in self.model.named_steps_:
                    self.preprocessor = self.model.named_steps_['preprocessor']
                logger.info("✅ Model loaded successfully for evaluation")
                return True
            else:
                logger.error("❌ Model file not found")
                return False
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            return False
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Comprehensive model evaluation.
        I implemented this to calculate all important performance metrics.
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            # Check for performance degradation
            metrics['performance_status'] = self._check_performance(metrics)
            
            # Store metrics history
            self.metrics_history.append(metrics)
            
            logger.info(f"📊 Model evaluation completed - Accuracy: {metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error during evaluation: {str(e)}")
            return {"error": str(e)}
    
    def _check_performance(self, metrics: Dict) -> str:
        """
        Check if model performance meets thresholds.
        I created this to automatically detect performance degradation.
        """
        issues = []
        
        for metric, threshold in self.performance_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                issues.append(f"{metric}: {metrics[metric]:.3f} < {threshold}")
        
        if issues:
            return f"⚠️ Performance issues detected: {', '.join(issues)}"
        else:
            return "✅ Performance within acceptable range"
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Perform cross-validation to assess model stability.
        I implemented this to check model consistency across different data splits.
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            # Extract the actual model from the pipeline
            if hasattr(self.model, 'named_steps_') and 'model' in self.model.named_steps_:
                actual_model = self.model.named_steps_['model']
            else:
                actual_model = self.model
            
            # Perform cross-validation
            cv_scores = cross_val_score(actual_model, X, y, cv=cv, scoring='accuracy')
            
            cv_results = {
                'timestamp': datetime.now().isoformat(),
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_range': [cv_scores.min(), cv_scores.max()],
                'stability': 'stable' if cv_scores.std() < 0.05 else 'unstable'
            }
            
            logger.info(f"🔄 Cross-validation completed - Mean: {cv_results['mean_cv_score']:.3f}, Std: {cv_results['std_cv_score']:.3f}")
            return cv_results
            
        except Exception as e:
            logger.error(f"❌ Error during cross-validation: {str(e)}")
            return {"error": str(e)}
    
    def detect_data_drift(self, new_data: pd.DataFrame, reference_data: pd.DataFrame) -> Dict:
        """
        Detect data drift between reference and new data.
        I created this to identify when input data distribution changes significantly.
        """
        try:
            drift_report = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': False,
                'drift_details': {}
            }
            
            # Compare numerical columns
            numerical_cols = new_data.select_dtypes(include=[np.number]).columns
            
            for col in numerical_cols:
                if col in reference_data.columns:
                    # Statistical tests for drift
                    ref_mean = reference_data[col].mean()
                    new_mean = new_data[col].mean()
                    ref_std = reference_data[col].std()
                    new_std = new_data[col].std()
                    
                    # Calculate drift metrics
                    mean_drift = abs(new_mean - ref_mean) / ref_std if ref_std > 0 else 0
                    std_drift = abs(new_std - ref_std) / ref_std if ref_std > 0 else 0
                    
                    drift_details = {
                        'mean_drift': mean_drift,
                        'std_drift': std_drift,
                        'reference_mean': ref_mean,
                        'new_mean': new_mean,
                        'reference_std': ref_std,
                        'new_std': new_std
                    }
                    
                    # Flag significant drift
                    if mean_drift > 2.0 or std_drift > 1.5:  # Thresholds for significant drift
                        drift_report['drift_detected'] = True
                        drift_report['drift_details'][col] = drift_details
            
            # Compare categorical columns
            categorical_cols = new_data.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col in reference_data.columns:
                    ref_dist = reference_data[col].value_counts(normalize=True)
                    new_dist = new_data[col].value_counts(normalize=True)
                    
                    # Calculate distribution difference
                    all_categories = set(ref_dist.index) | set(new_dist.index)
                    distribution_diff = 0
                    
                    for cat in all_categories:
                        ref_prob = ref_dist.get(cat, 0)
                        new_prob = new_dist.get(cat, 0)
                        distribution_diff += abs(new_prob - ref_prob)
                    
                    drift_details = {
                        'distribution_difference': distribution_diff,
                        'reference_distribution': ref_dist.to_dict(),
                        'new_distribution': new_dist.to_dict()
                    }
                    
                    if distribution_diff > 0.3:  # Threshold for categorical drift
                        drift_report['drift_detected'] = True
                        drift_report['drift_details'][col] = drift_details
            
            if drift_report['drift_detected']:
                logger.warning("⚠️ Data drift detected in production data")
            else:
                logger.info("✅ No significant data drift detected")
            
            return drift_report
            
        except Exception as e:
            logger.error(f"❌ Error detecting data drift: {str(e)}")
            return {"error": str(e)}
    
    def generate_performance_report(self, save_path: str = "reports/performance_report.json") -> Dict:
        """
        Generate comprehensive performance report.
        I created this to provide detailed insights into model performance.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'metrics_history': self.metrics_history[-10:],  # Last 10 evaluations
                'performance_summary': self._summarize_performance(),
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Performance report saved to {save_path}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {str(e)}")
            return {"error": str(e)}
    
    def _summarize_performance(self) -> Dict:
        """Summarize overall model performance."""
        if not self.metrics_history:
            return {"status": "No evaluation data available"}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            'latest_accuracy': latest_metrics.get('accuracy', 0),
            'latest_f1_score': latest_metrics.get('f1_score', 0),
            'performance_trend': self._calculate_trend(),
            'overall_status': latest_metrics.get('performance_status', 'Unknown')
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend over time."""
        if len(self.metrics_history) < 2:
            return "Insufficient data"
        
        recent_accuracy = self.metrics_history[-1].get('accuracy', 0)
        previous_accuracy = self.metrics_history[-2].get('accuracy', 0)
        
        if recent_accuracy > previous_accuracy + 0.01:
            return "Improving"
        elif recent_accuracy < previous_accuracy - 0.01:
            return "Declining"
        else:
            return "Stable"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on performance."""
        recommendations = []
        
        if not self.metrics_history:
            return ["No evaluation data available for recommendations"]
        
        latest_metrics = self.metrics_history[-1]
        
        # Check accuracy
        if latest_metrics.get('accuracy', 0) < 0.85:
            recommendations.append("Consider retraining the model - accuracy below threshold")
        
        # Check F1 score
        if latest_metrics.get('f1_score', 0) < 0.77:
            recommendations.append("Model may need hyperparameter tuning - F1 score low")
        
        # Check trend
        if self._calculate_trend() == "Declining":
            recommendations.append("Performance is declining - investigate data quality and consider model update")
        
        # Check data drift
        if len(self.metrics_history) > 5:
            recent_accuracies = [m.get('accuracy', 0) for m in self.metrics_history[-5:]]
            if max(recent_accuracies) - min(recent_accuracies) > 0.1:
                recommendations.append("High variance in performance - check for data drift")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory - continue monitoring")
        
        return recommendations
    
    def monitor_production_performance(self, new_predictions: List[Dict], actual_outcomes: List[int]) -> Dict:
        """
        Monitor model performance on new production data.
        I implemented this to track real-world model performance.
        """
        try:
            if len(new_predictions) != len(actual_outcomes):
                return {"error": "Mismatch between predictions and actual outcomes"}
            
            # Calculate performance on new data
            y_pred = [p.get('prediction', 0) for p in new_predictions]
            y_true = actual_outcomes
            
            performance = {
                'timestamp': datetime.now().isoformat(),
                'sample_size': len(y_true),
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add to history
            self.metrics_history.append(performance)
            
            logger.info(f"📈 Production performance monitored - Accuracy: {performance['accuracy']:.3f}")
            return performance
            
        except Exception as e:
            logger.error(f"❌ Error monitoring production performance: {str(e)}")
            return {"error": str(e)}


def run_evaluation_pipeline():
    """
    Run complete evaluation pipeline.
    I created this to automate the entire evaluation process.
    """
    evaluator = ModelEvaluator()
    
    # Load test data (you can modify this path)
    try:
        dataset = pd.read_csv("e-ccomerce_data.csv")
        X = dataset.drop(columns=["CustomerID", "Churn"])
        y = dataset["Churn"]
        
        logger.info(f"📊 Dataset loaded: {len(dataset)} samples, {X.shape[1]} features")
        logger.info(f"📊 Churn rate: {y.mean():.3f}")
        logger.info(f"📊 Missing values: {X.isnull().sum().sum()}")
        
        # Split data for evaluation (use same split as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info(f"📊 Train set: {len(X_train)} samples")
        logger.info(f"📊 Test set: {len(X_test)} samples")
        logger.info(f"📊 Test churn rate: {y_test.mean():.3f}")
        
        logger.info("🔄 Starting model evaluation pipeline...")
        
        # 1. Evaluate model performance
        evaluation_results = evaluator.evaluate_model(X_test, y_test)
        print("📊 Evaluation Results:")
        print(f"Accuracy: {evaluation_results.get('accuracy', 0):.3f}")
        print(f"Precision: {evaluation_results.get('precision', 0):.3f}")
        print(f"Recall: {evaluation_results.get('recall', 0):.3f}")
        print(f"F1 Score: {evaluation_results.get('f1_score', 0):.3f}")
        print(f"AUC: {evaluation_results.get('auc', 0):.3f}")
        print(f"Status: {evaluation_results.get('performance_status', 'Unknown')}")
        
        # 2. Cross-validation
        cv_results = evaluator.cross_validate_model(X, y)
        print(f"\n🔄 Cross-Validation Results:")
        print(f"Mean CV Score: {cv_results.get('mean_cv_score', 0):.3f}")
        print(f"CV Std: {cv_results.get('std_cv_score', 0):.3f}")
        print(f"CV Range: {cv_results.get('cv_range', [0, 0])}")
        print(f"CV Stability: {cv_results.get('stability', 'Unknown')}")
        
        # 3. Generate performance report
        report = evaluator.generate_performance_report()
        print(f"\n📋 Performance report generated")
        
        return {
            'evaluation': evaluation_results,
            'cross_validation': cv_results,
            'report': report
        }
        
    except Exception as e:
        logger.error(f"❌ Error in evaluation pipeline: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run evaluation when script is executed directly
    results = run_evaluation_pipeline()
    print("\n🎯 Evaluation pipeline completed!")
