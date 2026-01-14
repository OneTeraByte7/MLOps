"""
src/api/app.py - FastAPI application for model serving with A/B testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import xgboost as xgb
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import yaml
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import random
import sys
sys.path.append('src')
from explainability.explainer import ChurnExplainer

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Production ML API for customer churn prediction with A/B testing",
    version="1.0.0"
)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['model_version', 'prediction'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_version_gauge = Gauge('model_version', 'Current model version', ['version'])
drift_score = Gauge('drift_score', 'Data drift score')

# Global model storage
models = {}
preprocessor = None
explainer = None

class CustomerFeatures(BaseModel):
    """Input features for prediction"""
    customer_id: str = Field(..., example="CUST_000123")
    account_age_days: int = Field(..., ge=0, example=365)
    subscription_tier: str = Field(..., example="Professional")
    monthly_revenue: float = Field(..., ge=0, example=199.99)
    logins_per_month: int = Field(..., ge=0, example=25)
    feature_usage_depth: float = Field(..., ge=0, le=1, example=0.65)
    support_tickets: int = Field(..., ge=0, example=2)
    avg_ticket_resolution_days: float = Field(..., ge=0, example=3.5)
    nps_score: int = Field(..., ge=0, le=10, example=8)
    payment_delays: int = Field(..., ge=0, example=0)
    contract_length_months: int = Field(..., example=12)
    team_size: int = Field(..., ge=1, example=10)
    api_calls_per_month: int = Field(..., ge=0, example=15000)
    days_since_last_login: int = Field(..., ge=0, example=2)

class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: str
    churn_probability: float
    churn_prediction: str
    risk_level: str
    model_version: str
    timestamp: str
    confidence_score: float
    explanation: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[CustomerFeatures]

class ModelInfo(BaseModel):
    """Model information"""
    model_a_version: str
    model_b_version: Optional[str]
    ab_testing_enabled: bool
    traffic_split: dict

def load_models():
    """Load models from disk"""
    global models, preprocessor, explainer
    
    model_dir = 'models'
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("✓ Loaded preprocessor")
    
    # Load model A (latest)
    model_a_path = os.path.join(model_dir, 'model.json')
    if os.path.exists(model_a_path):
        models['model_a'] = xgb.Booster()
        models['model_a'].load_model(model_a_path)
        models['model_a_version'] = 'v1.0.0'
        model_version_gauge.labels(version='model_a').set(1.0)
        print("✓ Loaded model A (v1.0.0)")
    
    # Load model B (if exists for A/B testing)
    model_b_path = os.path.join(model_dir, 'model_b.json')
    if os.path.exists(model_b_path):
        models['model_b'] = xgb.Booster()
        models['model_b'].load_model(model_b_path)
        models['model_b_version'] = 'v1.1.0'
        model_version_gauge.labels(version='model_b').set(1.1)
        print("✓ Loaded model B (v1.1.0)")
    
    # Load explainer
    try:
        explainer = ChurnExplainer(model_a_path)
        print("✓ Loaded explainer")
    except Exception as e:
        print(f"⚠ Could not load explainer: {e}")
        explainer = None
    
    if not models:
        raise RuntimeError("No models found! Train a model first.")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    load_models()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model_info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_info", response_model=ModelInfo)
async def get_model_info():
    """Get current model information"""
    return {
        "model_a_version": models.get('model_a_version', 'unknown'),
        "model_b_version": models.get('model_b_version'),
        "ab_testing_enabled": config['ab_testing']['enabled'] and 'model_b' in models,
        "traffic_split": config['ab_testing']['traffic_split']
    }

def select_model():
    """Select model for prediction (A/B testing logic)"""
    if not config['ab_testing']['enabled'] or 'model_b' not in models:
        return models['model_a'], models['model_a_version']
    
    # Traffic split
    split = config['ab_testing']['traffic_split']
    
    if random.random() < split['model_a']:
        return models['model_a'], models['model_a_version']
    else:
        return models['model_b'], models['model_b_version']

def preprocess_input(customer: CustomerFeatures):
    """Preprocess input features"""
    # Convert to DataFrame
    df = pd.DataFrame([customer.dict()])
    
    # Feature engineering (same as training)
    df['revenue_per_user'] = df['monthly_revenue'] / df['team_size']
    df['engagement_score'] = (
        df['logins_per_month'] * 0.3 +
        df['feature_usage_depth'] * 100 * 0.3 +
        (df['api_calls_per_month'] / 1000) * 0.4
    )
    df['support_burden'] = df['support_tickets'] * df['avg_ticket_resolution_days']
    df['days_inactive_ratio'] = df['days_since_last_login'] / np.maximum(df['account_age_days'], 1)
    df['total_contract_value'] = df['monthly_revenue'] * df['contract_length_months']
    df['payment_reliability'] = 1 / (1 + df['payment_delays'])
    df['nps_category'] = pd.cut(df['nps_score'], 
                                 bins=[-1, 6, 8, 10], 
                                 labels=['Detractor', 'Passive', 'Promoter'])
    
    # Get feature columns
    numeric_features = config['data']['numeric_features'] + [
        'revenue_per_user', 'engagement_score', 'support_burden',
        'days_inactive_ratio', 'total_contract_value', 'payment_reliability'
    ]
    categorical_features = config['data']['categorical_features'] + ['nps_category']
    
    X = df[numeric_features + categorical_features]
    
    # Transform
    X_transformed = preprocessor.transform(X)
    
    return X_transformed

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures, explain: bool = False):
    """Make churn prediction for a single customer"""
    
    try:
        # Start timing
        start_time = datetime.now()
        # Select model (A/B testing)
        model, model_version = select_model()
        
        # Preprocess
        X = preprocess_input(customer)
        
        # Create DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # Predict
        churn_prob = float(model.predict(dmatrix)[0])
        churn_pred = "Yes" if churn_prob > 0.5 else "No"
        
        # Risk level
        if churn_prob < 0.3:
            risk_level = "Low"
        elif churn_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Confidence (distance from decision boundary)
        confidence = abs(churn_prob - 0.5) * 2
        
        # Generate explanation if requested
        explanation_text = None
        if explain and explainer is not None:
            explanation = explainer.explain_predictions(X, customer.customer_id)
            explanation_text = explanation['explanation_text']
        
        # Log metrics
        prediction_counter.labels(
            model_version=model_version, 
            prediction=churn_pred
        ).inc()
        
        # Log latency
        latency = (datetime.now() - start_time).total_seconds()
        prediction_latency.observe(latency)
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(churn_prob, 4),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            confidence_score=round(confidence, 4),
            explanation=explanation_text
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch predictions"""
    
    try:
        results = []
        
        for customer in request.customers:
            # Make prediction
            pred = await predict(customer)
            results.append(pred.dict())
        
        return {
            "predictions": results,
            "total_customers": len(results),
            "high_risk_count": sum(1 for r in results if r['risk_level'] == 'High'),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/reload_models")
async def reload_models():
    """Reload models from disk (for A/B testing or model updates)"""
    try:
        load_models()
        return {"status": "success", "message": "Models reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload error: {str(e)}")

@app.post("/explain")
async def explain_prediction(customer: CustomerFeatures):
    """Get detailed explanation for a prediction"""
    
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not available")
    
    try:
        # Preprocess
        X = preprocess_input(customer)
        
        # Generate explanation
        explanation = explainer.explain_predictions(X, customer.customer_id)
        
        return {
            "customer_id": explanation['customer_id'],
            "prediction": explanation['prediction'],
            "explanation_text": explanation['explanation_text'],
            "top_drivers": explanation['top_drivers'],
            "feature_contributions": explanation['feature_contributions'][:10]  # Top 10
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config['serving']['host'], 
        port=config['serving']['port']
    )