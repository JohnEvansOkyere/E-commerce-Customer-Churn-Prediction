"""
Pydantic schemas for API request/response validation.
I created these to ensure type safety and proper data validation in the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class CustomerData(BaseModel):
    """
    Schema for customer data input.
    I defined this to validate the input data structure for churn prediction.
    """
    Tenure: Optional[float] = Field(None, description="Customer tenure in months")
    PreferredLoginDevice: str = Field(..., description="Preferred login device")
    CityTier: int = Field(..., description="City tier (1, 2, or 3)")
    WarehouseToHome: Optional[float] = Field(None, description="Distance from warehouse to home")
    PreferredPaymentMode: str = Field(..., description="Preferred payment mode")
    Gender: str = Field(..., description="Customer gender")
    HourSpendOnApp: Optional[float] = Field(None, description="Hours spent on app")
    NumberOfDeviceRegistered: int = Field(..., description="Number of devices registered")
    PreferedOrderCat: str = Field(..., description="Preferred order category")
    SatisfactionScore: int = Field(..., ge=1, le=5, description="Satisfaction score (1-5)")
    MaritalStatus: str = Field(..., description="Marital status")
    NumberOfAddress: int = Field(..., description="Number of addresses")
    Complain: int = Field(..., ge=0, le=1, description="Complaint status (0 or 1)")
    OrderAmountHikeFromlastYear: Optional[float] = Field(None, description="Order amount hike from last year")
    CouponUsed: Optional[float] = Field(None, description="Number of coupons used")
    OrderCount: Optional[float] = Field(None, description="Number of orders")
    DaySinceLastOrder: Optional[float] = Field(None, description="Days since last order")
    CashbackAmount: int = Field(..., description="Cashback amount")


class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    I designed this to provide clear and structured prediction results.
    """
    prediction: int = Field(..., description="Prediction: 0 for no churn, 1 for churn")
    probability: Dict[str, float] = Field(..., description="Prediction probabilities")
    confidence: float = Field(..., description="Confidence score")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Model information")


class ModelInfo(BaseModel):
    """
    Schema for model information.
    I created this to provide model metadata and performance metrics.
    """
    model_type: str = Field(..., description="Type of model used")
    accuracy: float = Field(..., description="Model accuracy")
    training_date: str = Field(..., description="Date when model was trained")
    features_used: list = Field(..., description="List of features used in the model")


class HealthResponse(BaseModel):
    """
    Schema for health check response.
    I added this to monitor the API status and model availability.
    """
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
