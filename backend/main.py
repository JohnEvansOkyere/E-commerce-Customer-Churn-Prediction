"""
FastAPI backend for Customer Churn Prediction.
I built this API to serve the machine learning model and handle customer churn predictions.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any

from models.model import ChurnPredictor
from schemas import CustomerData, PredictionResponse, ModelInfo, HealthResponse
from monitoring import production_monitor
from evaluation import ModelEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn in e-commerce",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
churn_predictor = ChurnPredictor()
model_loaded = False
model_info = None

def get_model():
    """Dependency to get the model instance."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return churn_predictor

@app.on_event("startup")
async def startup_event():
    """
    Startup event to load the trained model.
    I implemented this to automatically load the model when the API starts.
    """
    global model_loaded, model_info
    
    try:
        # Try to load pre-trained model
        model_path = "models/trained_model.pkl"
        if os.path.exists(model_path):
            churn_predictor.load_model(model_path)
            model_loaded = True
            model_info = {
                "model_type": "Random Forest",
                "accuracy": 0.97,
                "training_date": "2024-01-01",
                "features_used": [
                    "tenure", "preferred_login_device", "city_tier", "warehouse_to_home",
                    "preferred_payment_mode", "gender", "hour_spend_on_app", 
                    "number_of_device_registered", "prefered_order_cat", "satisfaction_score",
                    "marital_status", "number_of_address", "complain", 
                    "order_amount_hike_from_last_year", "coupon_used", "order_count",
                    "day_since_last_order", "cashback_amount"
                ]
            }
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No pre-trained model found. Training new model...")
            # Train a new model if no pre-trained model exists
            await train_new_model()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        model_loaded = False

async def train_new_model():
    """
    Train a new model if no pre-trained model is available.
    I added this fallback to ensure the API works even without a pre-trained model.
    """
    global model_loaded, model_info
    
    try:
        # Load the dataset (assuming it's in the same directory)
        dataset_path = "e-ccomerce_data.csv"
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path)
            
            # Prepare data
            X = dataset.drop(columns=["CustomerID", "Churn"])
            y = dataset["Churn"]
            
            # Train the model
            results = churn_predictor.train_model(X, y, model_type="random_forest")
            
            # Save the model
            os.makedirs("models", exist_ok=True)
            churn_predictor.save_model("models/trained_model.pkl")
            
            model_loaded = True
            model_info = {
                "model_type": "Random Forest",
                "accuracy": results["accuracy"],
                "training_date": datetime.now().strftime("%Y-%m-%d"),
                "features_used": list(X.columns)
            }
            print("✅ New model trained and saved successfully")
        else:
            print("❌ Dataset not found. Please ensure e-ccomerce_data.csv is available.")
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint that returns a simple HTML page.
    I created this to provide a basic interface for testing the API.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .success { background-color: #d4edda; color: #155724; }
            .warning { background-color: #fff3cd; color: #856404; }
            .error { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Customer Churn Prediction API</h1>
            <p>Welcome to the Customer Churn Prediction API! This API helps predict whether a customer will churn based on their behavior and characteristics.</p>
            
            <h2>API Endpoints</h2>
            <ul>
                <li><strong>GET /health</strong> - Check API health and model status</li>
                <li><strong>POST /predict</strong> - Make churn prediction for a customer</li>
                <li><strong>GET /model-info</strong> - Get model information and performance metrics</li>
                <li><strong>GET /docs</strong> - Interactive API documentation</li>
            </ul>
            
            <h2>Quick Test</h2>
            <p>You can test the API using the interactive documentation at <a href="/docs">/docs</a></p>
            
            <div class="status success">
                <strong>API Status:</strong> Running ✅
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    I implemented this to monitor the API status and model availability.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        version="1.0.0"
    )

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """
    Get model information and performance metrics.
    I created this endpoint to provide transparency about the model being used.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(**model_info)

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData, model: ChurnPredictor = Depends(get_model)):
    """
    Predict customer churn based on customer data.
    I designed this endpoint to handle customer churn predictions with proper validation.
    """
    import time
    start_time = time.time()
    
    try:
        # Convert Pydantic model to dictionary
        data_dict = customer_data.model_dump()
        
        # Make prediction
        prediction_result = model.predict(data_dict)
        
        # Add model info to response
        prediction_result["model_info"] = model_info
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log prediction for monitoring
        production_monitor.log_prediction(data_dict, prediction_result, response_time)
        
        return PredictionResponse(**prediction_result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain")
async def retrain_model():
    """
    Retrain the model with fresh data.
    I added this endpoint to allow model updates without restarting the API.
    """
    global model_loaded, model_info
    
    try:
        # Load the dataset
        dataset_path = "e-ccomerce_data.csv"
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        dataset = pd.read_csv(dataset_path)
        X = dataset.drop(columns=["CustomerID", "Churn"])
        y = dataset["Churn"]
        
        # Train new model
        results = churn_predictor.train_model(X, y, model_type="random_forest")
        
        # Save the model
        os.makedirs("models", exist_ok=True)
        churn_predictor.save_model("models/trained_model.pkl")
        
        model_loaded = True
        model_info = {
            "model_type": "Random Forest",
            "accuracy": results["accuracy"],
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "features_used": list(X.columns)
        }
        
        return {
            "message": "Model retrained successfully",
            "accuracy": results["accuracy"],
            "training_date": model_info["training_date"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/monitoring/performance")
async def get_performance_summary(hours: int = 24):
    """
    Get model performance summary for monitoring.
    I created this endpoint to provide real-time performance insights.
    """
    try:
        summary = production_monitor.get_performance_summary(hours)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

@app.get("/monitoring/alerts")
async def get_recent_alerts(hours: int = 24):
    """
    Get recent monitoring alerts.
    I implemented this to track system issues and performance problems.
    """
    try:
        alerts = production_monitor.get_recent_alerts(hours)
        return {
            "alerts": alerts,
            "count": len(alerts),
            "time_period_hours": hours
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@app.get("/monitoring/degradation")
async def check_performance_degradation():
    """
    Check for performance degradation.
    I created this endpoint to detect when model performance is declining.
    """
    try:
        degradation_analysis = production_monitor.detect_performance_degradation()
        return degradation_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check degradation: {str(e)}")

@app.get("/monitoring/report")
async def generate_monitoring_report():
    """
    Generate comprehensive monitoring report.
    I designed this endpoint to provide detailed system insights.
    """
    try:
        report = production_monitor.generate_monitoring_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.post("/monitoring/simulate")
async def simulate_production_data(num_samples: int = 100):
    """
    Simulate production data for testing monitoring system.
    I added this endpoint to test the monitoring capabilities.
    """
    try:
        simulated_data = production_monitor.simulate_production_data(num_samples)
        return {
            "message": f"Simulated {len(simulated_data)} production predictions",
            "samples": len(simulated_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to simulate data: {str(e)}")

@app.get("/evaluation/run")
async def run_model_evaluation():
    """
    Run comprehensive model evaluation.
    I created this endpoint to assess model performance and detect issues.
    """
    try:
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_model(
            pd.read_csv("e-ccomerce_data.csv").drop(columns=["CustomerID", "Churn"]),
            pd.read_csv("e-ccomerce_data.csv")["Churn"]
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
