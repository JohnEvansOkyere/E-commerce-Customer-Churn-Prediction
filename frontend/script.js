/**
 * Customer Churn Prediction Frontend JavaScript
 * I created this to handle form submission, API communication, and result display.
 */

// API Configuration
const API_BASE_URL = "http://localhost:8000";  
// or
const API_BASE_URL = "https://e-commerce-customer-churn-prediction.onrender.com";

// DOM Elements
const form = document.getElementById('churnForm');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
const loading = document.getElementById('loading');

/**
 * Show loading spinner
 * I implemented this to provide user feedback during API calls.
 */
function showLoading() {
    loading.style.display = 'flex';
    resultSection.style.display = 'none';
}

/**
 * Hide loading spinner
 */
function hideLoading() {
    loading.style.display = 'none';
}

/**
 * Show error message
 * I added this to handle and display API errors gracefully.
 */
function showError(message) {
    hideLoading();
    resultCard.innerHTML = `
        <div class="error-message">
            <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
            <p>${message}</p>
        </div>
    `;
    resultSection.style.display = 'block';
}

/**
 * Show success message with prediction results
 * I designed this to display prediction results in a user-friendly format.
 */
function showResult(prediction) {
    hideLoading();
    
    const isChurn = prediction.prediction === 1;
    const churnProbability = prediction.probability.churn;
    const noChurnProbability = prediction.probability.no_churn;
    const confidence = prediction.confidence;
    
    resultCard.innerHTML = `
        <div class="prediction-result">
            <h3>Prediction Result</h3>
            <div class="prediction-value ${isChurn ? 'churn' : 'no-churn'}">
                ${isChurn ? 'CHURN' : 'NO CHURN'}
            </div>
            <div class="confidence-score">
                Confidence: ${(confidence * 100).toFixed(1)}%
            </div>
        </div>
        
        <div class="probability-details">
            <div class="probability-item">
                <h4>No Churn Probability</h4>
                <div class="probability-value">${(noChurnProbability * 100).toFixed(1)}%</div>
            </div>
            <div class="probability-item">
                <h4>Churn Probability</h4>
                <div class="probability-value">${(churnProbability * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        ${prediction.model_info ? `
        <div class="model-info">
            <h4><i class="fas fa-info-circle"></i> Model Information</h4>
            <p><strong>Model Type:</strong> ${prediction.model_info.model_type}</p>
            <p><strong>Accuracy:</strong> ${(prediction.model_info.accuracy * 100).toFixed(1)}%</p>
            <p><strong>Training Date:</strong> ${prediction.model_info.training_date}</p>
        </div>
        ` : ''}
    `;
    
    resultSection.style.display = 'block';
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Validate form data
 * I implemented this to ensure all required fields are filled.
 */
function validateForm(formData) {
    const requiredFields = [
        'preferred_login_device', 'city_tier', 'preferred_payment_mode', 
        'gender', 'number_of_device_registered', 'prefered_order_cat', 
        'satisfaction_score', 'marital_status', 'number_of_address', 
        'complain', 'cashback_amount'
    ];
    
    for (const field of requiredFields) {
        if (!formData[field] || formData[field] === '') {
            throw new Error(`Please fill in the ${field.replace(/_/g, ' ')} field`);
        }
    }
    
    return true;
}

/**
 * Convert form data to API format
 * I created this to transform form data into the format expected by the API.
 */
function formatDataForAPI(formData) {
    const apiData = {};
    
    // Map form fields to API fields
    const fieldMapping = {
        'tenure': 'Tenure',
        'preferred_login_device': 'PreferredLoginDevice',
        'city_tier': 'CityTier',
        'warehouse_to_home': 'WarehouseToHome',
        'preferred_payment_mode': 'PreferredPaymentMode',
        'gender': 'Gender',
        'hour_spend_on_app': 'HourSpendOnApp',
        'number_of_device_registered': 'NumberOfDeviceRegistered',
        'prefered_order_cat': 'PreferedOrderCat',
        'satisfaction_score': 'SatisfactionScore',
        'marital_status': 'MaritalStatus',
        'number_of_address': 'NumberOfAddress',
        'complain': 'Complain',
        'order_amount_hike_from_last_year': 'OrderAmountHikeFromlastYear',
        'coupon_used': 'CouponUsed',
        'order_count': 'OrderCount',
        'day_since_last_order': 'DaySinceLastOrder',
        'cashback_amount': 'CashbackAmount'
    };
    
    for (const [formField, apiField] of Object.entries(fieldMapping)) {
        const value = formData[formField];
        if (value !== '' && value !== null && value !== undefined) {
            // Convert numeric fields
            if (['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
                 'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 
                 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
                 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'].includes(apiField)) {
                apiData[apiField] = parseFloat(value) || parseInt(value);
            } else {
                apiData[apiField] = value;
            }
        }
    }
    
    return apiData;
}

/**
 * Make API request to predict churn
 * I implemented this to communicate with the FastAPI backend.
 */
async function predictChurn(customerData) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(customerData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw new Error(`Failed to get prediction: ${error.message}`);
    }
}

/**
 * Handle form submission
 * I created this to process the form submission and handle the prediction flow.
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    try {
        showLoading();
        
        // Get form data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        // Validate form
        validateForm(data);
        
        // Format data for API
        const apiData = formatDataForAPI(data);
        
        // Make prediction
        const prediction = await predictChurn(apiData);
        
        // Show results
        showResult(prediction);
        
    } catch (error) {
        console.error('Form submission error:', error);
        showError(error.message);
    }
}

/**
 * Clear form
 * I added this to allow users to reset the form easily.
 */
function clearForm() {
    form.reset();
    resultSection.style.display = 'none';
}

/**
 * Check API health
 * I implemented this to verify the API is available before making predictions.
 */
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const health = await response.json();
            console.log('API Health:', health);
            return health.model_loaded;
        }
        return false;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        return false;
    }
}

/**
 * Initialize the application
 * I created this to set up event listeners and check API availability.
 */
async function initializeApp() {
    // Add form submit event listener
    form.addEventListener('submit', handleFormSubmit);
    
    // Check API health on load
    const isAPIHealthy = await checkAPIHealth();
    if (!isAPIHealthy) {
        console.warn('API is not available or model is not loaded');
        // You could show a warning message to the user here
    }
    
    // Add some helpful tooltips or validation messages
    addFormValidation();
}

/**
 * Add form validation
 * I implemented this to provide real-time validation feedback.
 */
function addFormValidation() {
    const requiredFields = form.querySelectorAll('[required]');
    
    requiredFields.forEach(field => {
        field.addEventListener('blur', function() {
            if (this.value === '') {
                this.style.borderColor = '#e74c3c';
            } else {
                this.style.borderColor = '#27ae60';
            }
        });
        
        field.addEventListener('input', function() {
            if (this.value !== '') {
                this.style.borderColor = '#3498db';
            }
        });
    });
}

/**
 * Add sample data button functionality
 * I added this to help users test the form with sample data.
 */
function addSampleData() {
    const sampleData = {
        tenure: 12.5,
        preferred_login_device: 'Mobile Phone',
        city_tier: 2,
        warehouse_to_home: 15.2,
        preferred_payment_mode: 'Debit Card',
        gender: 'Male',
        hour_spend_on_app: 3.5,
        number_of_device_registered: 2,
        prefered_order_cat: 'Laptop & Accessory',
        satisfaction_score: 4,
        marital_status: 'Married',
        number_of_address: 1,
        complain: 0,
        order_amount_hike_from_last_year: 15.5,
        coupon_used: 2.5,
        order_count: 5.0,
        day_since_last_order: 7.5,
        cashback_amount: 100
    };
    
    // Fill form with sample data
    Object.entries(sampleData).forEach(([key, value]) => {
        const field = form.querySelector(`[name="${key}"]`);
        if (field) {
            field.value = value;
        }
    });
}

// Add sample data button to form actions
document.addEventListener('DOMContentLoaded', function() {
    const formActions = document.querySelector('.form-actions');
    const sampleButton = document.createElement('button');
    sampleButton.type = 'button';
    sampleButton.className = 'btn btn-secondary';
    sampleButton.innerHTML = '<i class="fas fa-flask"></i> Load Sample Data';
    sampleButton.onclick = addSampleData;
    formActions.appendChild(sampleButton);
});

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);
