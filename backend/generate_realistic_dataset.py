"""
Generate a realistic e-commerce dataset for proper model evaluation.
I created this to generate a larger, more realistic dataset that will give proper evaluation results.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_realistic_dataset(num_samples=1000):
    """
    Generate a realistic e-commerce customer dataset.
    I created this to produce a dataset that will give realistic evaluation results.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic distributions and relationships
    devices = ['Mobile Phone', 'Computer', 'Phone']
    payment_modes = ['Debit Card', 'Credit Card', 'E wallet', 'UPI', 'COD', 'CC', 'Cash on Delivery']
    genders = ['Male', 'Female']
    order_categories = ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Mobile', 'Grocery', 'Others']
    marital_statuses = ['Married', 'Single', 'Divorced']
    
    data = []
    
    for i in range(1, num_samples + 1):
        # Generate customer with realistic relationships
        gender = np.random.choice(genders)
        age = np.random.normal(35, 10)  # Average age around 35
        age = max(18, min(80, age))  # Clamp to realistic range
        
        # Tenure affects churn probability
        tenure = np.random.exponential(12)  # Average 12 months
        tenure = min(60, tenure)  # Max 5 years
        
        # City tier affects behavior
        city_tier = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        
        # Satisfaction score affects churn
        satisfaction_score = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.2, 0.4, 0.2])
        
        # Complain status
        complain = np.random.choice([0, 1], p=[0.8, 0.2])
        
        # Hours spent on app (correlates with engagement)
        hour_spend_on_app = np.random.gamma(2, 1.5)  # Skewed towards lower values
        hour_spend_on_app = min(10, hour_spend_on_app)
        
        # Number of devices (more devices = more engaged)
        number_of_devices = np.random.poisson(2) + 1
        number_of_devices = min(5, number_of_devices)
        
        # Warehouse to home distance
        warehouse_to_home = np.random.exponential(15)
        warehouse_to_home = min(50, warehouse_to_home)
        
        # Order count (active customers order more)
        order_count = np.random.poisson(6)
        order_count = max(0, order_count)
        
        # Days since last order (affects churn)
        day_since_last_order = np.random.exponential(7)
        day_since_last_order = min(30, day_since_last_order)
        
        # Coupon usage
        coupon_used = np.random.poisson(3)
        coupon_used = max(0, coupon_used)
        
        # Order amount hike
        order_amount_hike = np.random.normal(10, 15)
        order_amount_hike = max(-20, order_amount_hike)  # Can be negative
        
        # Cashback amount
        cashback_amount = np.random.poisson(100)
        cashback_amount = max(0, cashback_amount)
        
        # Number of addresses
        number_of_address = np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
        
        # Generate churn based on realistic factors
        churn_probability = 0.0
        
        # Base churn rate
        churn_probability += 0.1
        
        # Low satisfaction increases churn
        if satisfaction_score <= 2:
            churn_probability += 0.3
        elif satisfaction_score == 3:
            churn_probability += 0.1
        
        # Complaints increase churn
        if complain == 1:
            churn_probability += 0.2
        
        # Long time since last order increases churn
        if day_since_last_order > 14:
            churn_probability += 0.2
        elif day_since_last_order > 7:
            churn_probability += 0.1
        
        # Low engagement increases churn
        if hour_spend_on_app < 2:
            churn_probability += 0.15
        
        # Few orders increases churn
        if order_count < 3:
            churn_probability += 0.1
        
        # High order amount hike might indicate price sensitivity
        if order_amount_hike > 20:
            churn_probability += 0.1
        
        # Add some randomness
        churn_probability += np.random.normal(0, 0.1)
        churn_probability = max(0, min(1, churn_probability))
        
        # Determine churn
        churn = 1 if np.random.random() < churn_probability else 0
        
        # Create the record
        record = {
            'CustomerID': i,
            'Churn': churn,
            'Tenure': round(tenure, 1),
            'PreferredLoginDevice': np.random.choice(devices),
            'CityTier': city_tier,
            'WarehouseToHome': round(warehouse_to_home, 1),
            'PreferredPaymentMode': np.random.choice(payment_modes),
            'Gender': gender,
            'HourSpendOnApp': round(hour_spend_on_app, 1),
            'NumberOfDeviceRegistered': number_of_devices,
            'PreferedOrderCat': np.random.choice(order_categories),
            'SatisfactionScore': satisfaction_score,
            'MaritalStatus': np.random.choice(marital_statuses),
            'NumberOfAddress': number_of_address,
            'Complain': complain,
            'OrderAmountHikeFromlastYear': round(order_amount_hike, 1),
            'CouponUsed': round(coupon_used, 1),
            'OrderCount': round(order_count, 1),
            'DaySinceLastOrder': round(day_since_last_order, 1),
            'CashbackAmount': cashback_amount
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to make it more realistic
    missing_columns = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 
                       'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
    
    for col in missing_columns:
        # Randomly set 5-10% of values to NaN
        missing_rate = np.random.uniform(0.05, 0.1)
        missing_indices = np.random.choice(df.index, size=int(len(df) * missing_rate), replace=False)
        df.loc[missing_indices, col] = np.nan
    
    return df

if __name__ == "__main__":
    print("🔄 Generating realistic e-commerce dataset...")
    
    # Generate dataset
    dataset = generate_realistic_dataset(1000)
    
    # Save to CSV
    dataset.to_csv("e-ccomerce_data.csv", index=False)
    
    print(f"✅ Generated dataset with {len(dataset)} samples")
    print(f"📊 Churn rate: {dataset['Churn'].mean():.3f}")
    print(f"📊 Missing values per column:")
    print(dataset.isnull().sum())
    
    # Show sample
    print("\n📋 Sample data:")
    print(dataset.head())
