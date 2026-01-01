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
from fastapi.responses import Responses
import random


with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
app = FastAPI(
    title = "Churn Prediction API",
    description = "Production ML API fro customer churn prediction with A/B testing",
    version = "1.0.0"
)

prediction_counter = Counter('prediction_total, "Total predictions', ['model_version', 'prediction'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
model_version_gauge = Gauge('model_version', 'Current model version', ['version'])
drift_score = Gauge('drift_score', 'Data drift score')

model = {}
preprocessor = None
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example = "CUST_000123")
    account_age_days: int  = Field(..., ge = 0, example=365)
    subscription_tier: str = Field(..., example = "Professional")
    monthly_revenue: float = Field(...,ge = 0, example = 199.99)
    logins_per_month: int = Field(..., ge=0,example=25)
    feature_usage_depth: float = Field(...,ge=0, le=1, example=0.65)
    support_tickets: int  = Field(..., ge=0, example=2)
    avg_ticket_resolution_days: int = Field(..., ge=0, example=3.5)
    nps_score: int = Field(..., ge=0, le=10, example=8)
    payment_delays: int = Field(..., ge=0, example=0)
    contract_length_months: int = Field(..., example=12)
    team_size: int = Field(..., ge=1, example=10)
    api_calls_per_month: int = Field(..., ge=0, example=15000)
    days_since_last_login: int = Field(..., ge=0, example=2)
    
class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: str
    risk_level: str
    model_version: str
    timestamp: str
    confidance_score: float
    
class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]
    
class ModelInfo(BaseModel):
    model_a_version: str
    model_b_version: Optional[str]
    ab_testing_enables: bool
    traffic_split: dict
    
def load_models():
    global models, preprocessor
    model_dir = 'models'
    
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("Loaded Preprocessor")
        
    model_a_path = os.path.join(model_dir, 'model.json')
    if os.path.exists(model_a_path):
        models['model_a'] = xgb.Booster()
        models['model_a'].load_model(model_a_path)
        models['model_a_version'] = 'v1.0.0'
        model_version_gauge.labels(version='model_a').set(1.0)
        print("Loaded model A (v1.0.0)")
        
    model_b_path = os.path.join(model_dir, 'model_b.json')
    if os.path.exists(model_b_path):
        models['model_b'] = xbg.Booster()
        models['model_b'].load_model(model_b_path)
        models['model_b_version'] = 'v1.1.0'
        model_version_gauge.labels(version = 'model_b').set(1.1)
        print("Loaded model B (v1.1.0)")
        
    if not models:
        raise RuntimeError("No models found! Train a model first")
    
@app.lifespan("startup")
async def startup_event():
    load_models()
    
async def root():
    return{
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
    return{
        "status": "healthy",
        "models_loaded": len(models),
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }        
    
@app.get("/model_info", response_model = ModelInfo)
async def get_model_info():
    return{
        "model_a_version": models.get('model_a_version', 'unknown'),
        "model_b_version": models.get('model_b_version'),
        "ab_testing_enabled": config['ab_testing']['enabled'] and 'model_b' is models,
        "traffic_split": config['ab_testing']['traffic_split']
    }
    
def select_model():
    if not config['ab_testing']['enabled'] or 'model_b' not in models:
        return models['model_a'], models['model_a_version']
    
    
    split = config['ab_testing']['traffic_split']
    
    if random.random() < split['model_a']:
        return models['model_a'], models['model_a_version']
    else:
        return models['model_b'], models['model_b_version']
    
    def preprocess_input(cutomer: CustomerFeatures):
        
        df = pd.DataFrame([customer.dict()])
        
        df['revenue_per_user'] = df['monthly_revenue'] / df['team_size']
        df['engagement_score'] = (
            df['logins_per_month'] * 0.3 +
            df['features_usage_depth'] * 100 * 0.3 + 
            (df['api_calls_per_month'] / 1000) * 0.4
        )
        
        df['support_burden'] = df['support_tickets'] * df['avg_ticket_resolution_days']
        df['days_inactive_ratio'] = df['days_since_last_login'] / np.maximum(df['account_age_days'], 1)
        df['total_contract_value'] = df['monthly_revenue'] * df['contract_length_months']
        df['payment_reliabillty'] = df1 / (1 + df['payment_delays'])
        df['nps_category'] = pd.cut(df['nps_score'],
                                    bins = [-1, 6, 8, 10],
                                    labels = ['Detractor', 'Passive', 'Promoter'])
        
        numeric_features = config['data']['numeric_features'] + ['revernue_per_user', 'engagement_score', 'support_burden', 'days_inactive_ration', 'total_contract value', 'payment_reliability']
        
        categorical+features = config['data']['categorical_features'] + ['nps_category']
        
        X = df[numeric_features + categorical_features]
        
        X_transformed = preprocessor.transform(X)
        
        return X_transfomed
        