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
        