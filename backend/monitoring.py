"""
Production Model Monitoring System
I created this to provide real-time monitoring and alerting for the ML model in production.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
from evaluation import ModelEvaluator
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionMonitor:
    """
    Production monitoring system for ML model.
    I designed this to track model performance, detect issues, and send alerts.
    """
    
    def __init__(self, model_path: str = "models/trained_model.pkl"):
        """
        Initialize the production monitor.
        I created this to set up monitoring infrastructure and alerting.
        """
        self.model_path = model_path
        self.evaluator = ModelEvaluator(model_path)
        self.alert_thresholds = {
            'accuracy_drop': 0.05,  # Alert if accuracy drops by 5%
            'response_time': 2.0,   # Alert if response time > 2 seconds
            'error_rate': 0.1,      # Alert if error rate > 10%
            'prediction_confidence': 0.7  # Alert if confidence < 70%
        }
        self.monitoring_data = []
        self.alerts = []
        
    def log_prediction(self, input_data: Dict, prediction: Dict, response_time: float):
        """
        Log a prediction for monitoring.
        I implemented this to track all predictions and their performance.
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'input_data': input_data,
                'prediction': prediction,
                'response_time': response_time,
                'confidence': prediction.get('confidence', 0),
                'predicted_class': prediction.get('prediction', 0)
            }
            
            self.monitoring_data.append(log_entry)
            
            # Keep only last 1000 entries to manage memory
            if len(self.monitoring_data) > 1000:
                self.monitoring_data = self.monitoring_data[-1000:]
            
            # Check for immediate alerts
            self._check_immediate_alerts(log_entry)
            
            logger.info(f"📝 Prediction logged - Confidence: {log_entry['confidence']:.3f}, Response time: {response_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Error logging prediction: {str(e)}")
    
    def _check_immediate_alerts(self, log_entry: Dict):
        """Check for immediate alert conditions."""
        alerts = []
        
        # Check response time
        if log_entry['response_time'] > self.alert_thresholds['response_time']:
            alerts.append({
                'type': 'high_response_time',
                'message': f"Response time {log_entry['response_time']:.2f}s exceeds threshold {self.alert_thresholds['response_time']}s",
                'timestamp': log_entry['timestamp'],
                'severity': 'warning'
            })
        
        # Check prediction confidence
        if log_entry['confidence'] < self.alert_thresholds['prediction_confidence']:
            alerts.append({
                'type': 'low_confidence',
                'message': f"Prediction confidence {log_entry['confidence']:.3f} below threshold {self.alert_thresholds['prediction_confidence']}",
                'timestamp': log_entry['timestamp'],
                'severity': 'warning'
            })
        
        # Add alerts to the list
        self.alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"🚨 {alert['severity'].upper()}: {alert['message']}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict:
        """
        Get performance summary for the last N hours.
        I created this to provide real-time performance insights.
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter data for the specified time period
            recent_data = [
                entry for entry in self.monitoring_data
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            if not recent_data:
                return {
                    'status': 'no_data',
                    'message': f'No data available for the last {hours} hours'
                }
            
            # Calculate metrics
            total_predictions = len(recent_data)
            avg_response_time = np.mean([entry['response_time'] for entry in recent_data])
            avg_confidence = np.mean([entry['confidence'] for entry in recent_data])
            
            # Count predictions by class
            churn_predictions = sum(1 for entry in recent_data if entry['predicted_class'] == 1)
            no_churn_predictions = total_predictions - churn_predictions
            
            # Calculate error rate (if we had actual outcomes)
            error_rate = 0.0  # This would be calculated with actual outcomes
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'time_period_hours': hours,
                'total_predictions': total_predictions,
                'avg_response_time': round(avg_response_time, 3),
                'avg_confidence': round(avg_confidence, 3),
                'churn_predictions': churn_predictions,
                'no_churn_predictions': no_churn_predictions,
                'churn_rate': round(churn_predictions / total_predictions, 3),
                'error_rate': error_rate,
                'status': 'healthy' if avg_confidence > 0.8 and avg_response_time < 1.0 else 'degraded'
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {str(e)}")
            return {"error": str(e)}
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        return recent_alerts
    
    def detect_performance_degradation(self) -> Dict:
        """
        Detect performance degradation over time.
        I implemented this to identify when model performance is declining.
        """
        try:
            if len(self.monitoring_data) < 10:
                return {"status": "insufficient_data", "message": "Need at least 10 predictions to detect degradation"}
            
            # Get recent and older data for comparison
            recent_data = self.monitoring_data[-10:]  # Last 10 predictions
            older_data = self.monitoring_data[-20:-10] if len(self.monitoring_data) >= 20 else self.monitoring_data[:-10]
            
            # Compare metrics
            recent_confidence = np.mean([entry['confidence'] for entry in recent_data])
            older_confidence = np.mean([entry['confidence'] for entry in older_data])
            
            recent_response_time = np.mean([entry['response_time'] for entry in recent_data])
            older_response_time = np.mean([entry['response_time'] for entry in older_data])
            
            degradation_detected = False
            issues = []
            
            # Check confidence degradation
            if recent_confidence < older_confidence - 0.1:
                degradation_detected = True
                issues.append(f"Confidence dropped from {older_confidence:.3f} to {recent_confidence:.3f}")
            
            # Check response time degradation
            if recent_response_time > older_response_time + 0.5:
                degradation_detected = True
                issues.append(f"Response time increased from {older_response_time:.3f}s to {recent_response_time:.3f}s")
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'degradation_detected': degradation_detected,
                'issues': issues,
                'recent_confidence': recent_confidence,
                'older_confidence': older_confidence,
                'recent_response_time': recent_response_time,
                'older_response_time': older_response_time
            }
            
            if degradation_detected:
                logger.warning(f"⚠️ Performance degradation detected: {', '.join(issues)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error detecting performance degradation: {str(e)}")
            return {"error": str(e)}
    
    def generate_monitoring_report(self) -> Dict:
        """
        Generate comprehensive monitoring report.
        I created this to provide detailed insights into system performance.
        """
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'monitoring_period': f"{len(self.monitoring_data)} predictions",
                'performance_summary': self.get_performance_summary(24),
                'recent_alerts': self.get_recent_alerts(24),
                'degradation_analysis': self.detect_performance_degradation(),
                'recommendations': self._generate_monitoring_recommendations()
            }
            
            # Save report
            os.makedirs("reports", exist_ok=True)
            with open("reports/monitoring_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("📊 Monitoring report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating monitoring report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate recommendations based on monitoring data."""
        recommendations = []
        
        if not self.monitoring_data:
            return ["No monitoring data available for recommendations"]
        
        # Check recent performance
        recent_summary = self.get_performance_summary(1)  # Last hour
        
        if recent_summary.get('status') == 'degraded':
            recommendations.append("Model performance is degraded - consider retraining")
        
        if recent_summary.get('avg_confidence', 0) < 0.8:
            recommendations.append("Average prediction confidence is low - review model quality")
        
        if recent_summary.get('avg_response_time', 0) > 1.0:
            recommendations.append("Response times are high - check system resources")
        
        # Check for alerts
        recent_alerts = self.get_recent_alerts(1)
        if len(recent_alerts) > 5:
            recommendations.append("High number of alerts - investigate system stability")
        
        if not recommendations:
            recommendations.append("System is performing well - continue monitoring")
        
        return recommendations
    
    def simulate_production_data(self, num_samples: int = 100) -> List[Dict]:
        """
        Simulate production data for testing.
        I created this to test the monitoring system with realistic data.
        """
        try:
            # Load the dataset to get realistic data
            dataset = pd.read_csv("e-ccomerce_data.csv")
            sample_data = dataset.sample(n=min(num_samples, len(dataset)), random_state=42)
            
            simulated_predictions = []
            
            for _, row in sample_data.iterrows():
                # Simulate input data
                input_data = {
                    'Tenure': row.get('Tenure', 12.0),
                    'PreferredLoginDevice': row.get('PreferredLoginDevice', 'Mobile Phone'),
                    'CityTier': row.get('CityTier', 2),
                    'WarehouseToHome': row.get('WarehouseToHome', 15.2),
                    'PreferredPaymentMode': row.get('PreferredPaymentMode', 'Debit Card'),
                    'Gender': row.get('Gender', 'Male'),
                    'HourSpendOnApp': row.get('HourSpendOnApp', 3.5),
                    'NumberOfDeviceRegistered': row.get('NumberOfDeviceRegistered', 2),
                    'PreferedOrderCat': row.get('PreferedOrderCat', 'Laptop & Accessory'),
                    'SatisfactionScore': row.get('SatisfactionScore', 4),
                    'MaritalStatus': row.get('MaritalStatus', 'Married'),
                    'NumberOfAddress': row.get('NumberOfAddress', 1),
                    'Complain': row.get('Complain', 0),
                    'OrderAmountHikeFromlastYear': row.get('OrderAmountHikeFromlastYear', 15.5),
                    'CouponUsed': row.get('CouponUsed', 2.5),
                    'OrderCount': row.get('OrderCount', 5.0),
                    'DaySinceLastOrder': row.get('DaySinceLastOrder', 7.5),
                    'CashbackAmount': row.get('CashbackAmount', 100)
                }
                
                # Simulate prediction
                prediction = {
                    'prediction': np.random.choice([0, 1], p=[0.7, 0.3]),  # Simulate 30% churn rate
                    'confidence': np.random.uniform(0.6, 0.95),
                    'probability': {
                        'no_churn': np.random.uniform(0.3, 0.9),
                        'churn': np.random.uniform(0.1, 0.7)
                    }
                }
                
                # Simulate response time
                response_time = np.random.uniform(0.1, 2.0)
                
                # Log the prediction
                self.log_prediction(input_data, prediction, response_time)
                
                simulated_predictions.append({
                    'input': input_data,
                    'prediction': prediction,
                    'response_time': response_time
                })
            
            logger.info(f"🎭 Simulated {len(simulated_predictions)} production predictions")
            return simulated_predictions
            
        except Exception as e:
            logger.error(f"❌ Error simulating production data: {str(e)}")
            return []


# Global monitor instance
production_monitor = ProductionMonitor()


def run_monitoring_demo():
    """
    Run monitoring system demo.
    I created this to demonstrate the monitoring capabilities.
    """
    print("🔍 Starting Production Monitoring Demo...")
    
    # Simulate some production data
    print("📊 Simulating production predictions...")
    simulated_data = production_monitor.simulate_production_data(50)
    
    # Get performance summary
    print("\n📈 Performance Summary (Last 24 hours):")
    summary = production_monitor.get_performance_summary(24)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Check for alerts
    print("\n🚨 Recent Alerts:")
    alerts = production_monitor.get_recent_alerts(1)
    if alerts:
        for alert in alerts:
            print(f"  {alert['severity'].upper()}: {alert['message']}")
    else:
        print("  No recent alerts")
    
    # Check for degradation
    print("\n📉 Performance Degradation Analysis:")
    degradation = production_monitor.detect_performance_degradation()
    for key, value in degradation.items():
        print(f"  {key}: {value}")
    
    # Generate comprehensive report
    print("\n📋 Generating monitoring report...")
    report = production_monitor.generate_monitoring_report()
    print("✅ Monitoring report saved to reports/monitoring_report.json")
    
    return report


if __name__ == "__main__":
    # Run monitoring demo
    report = run_monitoring_demo()
    print("\n🎯 Monitoring demo completed!")
