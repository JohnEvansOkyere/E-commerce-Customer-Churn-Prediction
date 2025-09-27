# Customer Churn Prediction System

A production-ready machine learning application for predicting customer churn in e-commerce businesses. This project transforms a Jupyter notebook into a full-stack web application with FastAPI backend and modern frontend.

## 🚀 Features

- **Machine Learning Pipeline**: Random Forest, Decision Tree, Logistic Regression, and XGBoost models
- **FastAPI Backend**: RESTful API with automatic documentation
- **Modern Frontend**: Responsive HTML/CSS/JavaScript interface
- **Production Ready**: Dockerized and deployable to cloud platforms
- **Real-time Predictions**: Instant churn predictions with confidence scores
- **Model Management**: Retrain models without downtime

## 📁 Project Structure

```
E-commerce-Customer-Churn-Prediction/
├── backend/                 # FastAPI backend
│   ├── models/             # ML model classes
│   ├── main.py            # FastAPI application
│   ├── schemas.py          # Pydantic schemas
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Container configuration
│   └── render.yaml         # Render deployment config
├── frontend/               # Frontend application
│   ├── index.html         # Main HTML file
│   ├── styles.css         # CSS styles
│   ├── script.js          # JavaScript logic
│   ├── vercel.json        # Vercel deployment config
│   └── package.json       # Frontend dependencies
├── venv/                   # Python virtual environment
└── README.md              # This file
```

## 🛠️ Local Development

### Prerequisites

- Python 3.11+
- pip
- Git

### Backend Setup

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Run the backend:**
   ```bash
   python main.py
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Start the development server:**
   ```bash
   python3 -m http.server 3000
   ```

   The frontend will be available at `http://localhost:3000`

### Testing the Application

1. **Test API endpoints:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Train model
   curl -X POST http://localhost:8000/retrain
   
   # Make prediction
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "Tenure": 12.0,
       "PreferredLoginDevice": "Mobile Phone",
       "CityTier": 2,
       "WarehouseToHome": 15.2,
       "PreferredPaymentMode": "Debit Card",
       "Gender": "Male",
       "HourSpendOnApp": 3.5,
       "NumberOfDeviceRegistered": 2,
       "PreferedOrderCat": "Laptop & Accessory",
       "SatisfactionScore": 4,
       "MaritalStatus": "Married",
       "NumberOfAddress": 1,
       "Complain": 0,
       "OrderAmountHikeFromlastYear": 15.5,
       "CouponUsed": 2.5,
       "OrderCount": 5.0,
       "DaySinceLastOrder": 7.5,
       "CashbackAmount": 100
     }'
   ```

2. **Access the web interface:**
   - Open `http://localhost:3000` in your browser
   - Fill out the customer information form
   - Click "Predict Churn" to get predictions

## 🚀 Deployment

### Backend Deployment (Render)

1. **Push your code to GitHub**
2. **Connect to Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the `backend` folder as the root directory
   - Use the following settings:
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Python Version:** 3.11

3. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment to complete
   - Note the deployed URL (e.g., `https://your-app.onrender.com`)

### Frontend Deployment (Vercel)

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy from frontend directory:**
   ```bash
   cd frontend
   vercel --prod
   ```

3. **Update API URL:**
   - Edit `frontend/script.js`
   - Update `API_BASE_URL` to your deployed backend URL
   - Redeploy to Vercel

### Alternative: Docker Deployment

**Backend:**
```bash
cd backend
docker build -t churn-backend .
docker run -p 8000:8000 churn-backend
```

**Frontend:**
```bash
cd frontend
docker run -p 3000:3000 -v $(pwd):/usr/share/nginx/html nginx
```

## 📊 API Documentation

Once the backend is running, visit:
- **Interactive API Docs:** `http://localhost:8000/docs`
- **ReDoc Documentation:** `http://localhost:8000/redoc`

### Key Endpoints

- `GET /health` - API health check
- `POST /predict` - Make churn prediction
- `GET /model-info` - Get model information
- `POST /retrain` - Retrain the model

## 🧠 Machine Learning Models

The system supports multiple ML algorithms:

- **Random Forest Classifier** (Default)
- **Decision Tree Classifier**
- **Logistic Regression**
- **XGBoost Classifier**

### Model Performance

Based on the original notebook analysis:
- **XGBoost:** 98% accuracy
- **Random Forest:** 97% accuracy
- **Decision Tree:** 97% accuracy
- **Logistic Regression:** 92% accuracy

## 🔧 Configuration

### Environment Variables

- `PYTHON_VERSION` - Python version for deployment
- `API_BASE_URL` - Backend API URL (frontend)

### Model Configuration

The model can be configured in `backend/models/model.py`:
- Change model type in `train_model()` method
- Adjust hyperparameters
- Modify preprocessing pipeline

## 📈 Usage Examples

### Python API Client

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "Tenure": 12.0,
        "PreferredLoginDevice": "Mobile Phone",
        "CityTier": 2,
        # ... other fields
    }
)

prediction = response.json()
print(f"Churn Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### JavaScript Frontend

```javascript
// The frontend automatically handles form submission
// and displays results with confidence scores
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not loading:**
   - Check if dataset exists in backend directory
   - Run `/retrain` endpoint to train new model

2. **CORS errors:**
   - Ensure backend CORS settings allow frontend domain
   - Check API_BASE_URL in frontend

3. **Prediction errors:**
   - Verify all required fields are provided
   - Check field names match API schema

### Logs

- **Backend logs:** Check terminal output or Render logs
- **Frontend logs:** Open browser developer console

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original notebook analysis and insights
- FastAPI for the excellent web framework
- Scikit-learn for machine learning tools
- Vercel and Render for deployment platforms

---

**Built with ❤️ for machine learning and web development**