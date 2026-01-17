"""
src/api/app.py - FastAPI application for model serving with A/B testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
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
import os

# Get the project root directory (2 levels up from this file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(project_root)
sys.path.append('src')
sys.path.append(project_root)

# Import explainer with error handling
try:
    from explainability.explainer import ChurnExplainer
except ImportError:
    try:
        sys.path.append('.')
        from src.explainability.explainer import ChurnExplainer
    except ImportError:
        print("⚠️  Warning: Could not import ChurnExplainer. Explainability features disabled.")
        ChurnExplainer = None

# Import prediction logger
from api.prediction_logger import prediction_logger

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Production ML API for customer churn prediction with A/B testing",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        if ChurnExplainer is not None:
            explainer = ChurnExplainer(model_a_path)
            print("✓ Loaded explainer")
        else:
            explainer = None
            print("⚠ Explainer not available")
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
        
        # Create response
        response = PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=round(churn_prob, 4),
            churn_prediction=churn_pred,
            risk_level=risk_level,
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            confidence_score=round(confidence, 4),
            explanation=explanation_text
        )
        
        # Log prediction for dashboard
        prediction_logger.log_prediction({
            'timestamp': response.timestamp,
            'customer_id': response.customer_id,
            'churn_probability': response.churn_probability,
            'prediction': response.churn_prediction,
            'risk_level': response.risk_level,
            'model_version': response.model_version
        })
        
        return response
    
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

@app.get("/api/dashboard/overview")
async def get_dashboard_overview():
    """Get dashboard overview metrics from real prediction data"""
    try:
        # Get predictions from logger
        predictions = prediction_logger.get_predictions(limit=10000)
        
        if not predictions:
            # Return empty state
            return {
                "total_predictions": 0,
                "churn_rate": 0,
                "high_risk_customers": 0,
                "avg_confidence": 0,
                "predictions": [],
                "timestamp": datetime.now().isoformat(),
                "message": "No predictions available yet. Make predictions using /predict endpoint."
            }
        
        # Calculate metrics
        total = len(predictions)
        churn_count = sum(1 for p in predictions if p.get('prediction') == 'Yes')
        high_risk = sum(1 for p in predictions if p.get('risk_level') == 'High')
        avg_confidence = sum(p.get('churn_probability', 0) for p in predictions) / total
        
        # Recent 24h predictions
        recent_24h = [p for p in predictions if 
                     (datetime.now() - datetime.fromisoformat(p['timestamp'])).total_seconds() < 86400]
        
        return {
            "total_predictions": len(recent_24h),
            "churn_rate": churn_count / total if total > 0 else 0,
            "high_risk_customers": high_risk,
            "avg_confidence": avg_confidence,
            "predictions": predictions[-100:] if predictions else [],  # Latest 100
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error in dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/mlflow/runs")
async def get_mlflow_runs():
    """Get MLflow experiment runs directly from SQLite database"""
    try:
        import sqlite3
        
        # Connect to MLflow database
        conn = sqlite3.connect('mlflow.db')
        
        # Query runs with metrics and params
        query = """
        SELECT 
            r.run_uuid,
            r.experiment_id,
            r.status,
            datetime(r.start_time/1000, 'unixepoch') as start_time,
            r.lifecycle_stage
        FROM runs r
        WHERE r.lifecycle_stage = 'active'
        ORDER BY r.start_time DESC
        LIMIT 20
        """
        
        runs_df = pd.read_sql(query, conn)
        
        if len(runs_df) == 0:
            conn.close()
            return {"runs": [], "count": 0, "message": "No training runs found. Run: python train_model.py"}
        
        # Get metrics for each run
        runs_data = []
        for _, run in runs_df.iterrows():
            run_uuid = run['run_uuid']
            
            # Get metrics for this run
            metrics_query = f"""
            SELECT key, value 
            FROM metrics 
            WHERE run_uuid = '{run_uuid}'
            """
            metrics_df = pd.read_sql(metrics_query, conn)
            metrics_dict = dict(zip(metrics_df['key'], metrics_df['value']))
            
            # Get params for this run
            params_query = f"""
            SELECT key, value 
            FROM params 
            WHERE run_uuid = '{run_uuid}'
            """
            params_df = pd.read_sql(params_query, conn)
            params_dict = dict(zip(params_df['key'], params_df['value']))
            
            runs_data.append({
                'run_id': run_uuid,
                'start_time': run['start_time'],
                'status': run['status'],
                'metrics': {
                    'test_auc': float(metrics_dict.get('test_auc', 0)),
                    'test_f1': float(metrics_dict.get('test_f1', 0)),
                    'test_precision': float(metrics_dict.get('test_precision', 0)),
                    'test_recall': float(metrics_dict.get('test_recall', 0)),
                },
                'params': params_dict
            })
        
        conn.close()
        return {"runs": runs_data, "count": len(runs_data)}
            
    except Exception as e:
        print(f"MLflow database error: {e}")
        return {"runs": [], "count": 0, "error": str(e)}


@app.get("/api/drift/latest")
async def get_drift_report():
    """Get latest drift report"""
    try:
        report_path = 'monitoring/reports/latest_report.json'
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            return report
        else:
            # No drift report available
            return {
                'drift_status': {
                    'overall_status': 'healthy',
                    'alerts': [],
                    'recommendations': []
                },
                'drifted_features_count': 0,
                'feature_drift': {},
                'label_drift': {'churn_rate_change': 0},
                'model_performance': {'auc': 0},
                'timestamp': datetime.now().isoformat(),
                'message': 'No drift report available. Run drift detection first.'
            }
            
    except Exception as e:
        print(f"Drift report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/stats")
async def get_training_stats():
    """Get training statistics from MLflow database"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('mlflow.db')
        
        # Get total training runs
        total_runs = pd.read_sql("SELECT COUNT(*) as count FROM runs", conn)['count'][0]
        
        # Get recent runs count
        recent_query = """
        SELECT COUNT(*) as count 
        FROM runs 
        WHERE start_time > (strftime('%s', 'now', '-7 days') * 1000)
        """
        recent_runs = pd.read_sql(recent_query, conn)['count'][0]
        
        # Get best model metrics
        best_model_query = """
        SELECT m.key, m.value
        FROM metrics m
        JOIN runs r ON m.run_uuid = r.run_uuid
        WHERE m.key IN ('test_auc', 'test_f1', 'test_precision', 'test_recall')
        ORDER BY r.start_time DESC
        LIMIT 4
        """
        best_metrics = pd.read_sql(best_model_query, conn)
        best_metrics_dict = dict(zip(best_metrics['key'], best_metrics['value']))
        
        # Get training history
        history_query = """
        SELECT 
            datetime(r.start_time/1000, 'unixepoch') as date,
            COUNT(*) as runs_count
        FROM runs r
        GROUP BY date(r.start_time/1000, 'unixepoch')
        ORDER BY r.start_time DESC
        LIMIT 30
        """
        history = pd.read_sql(history_query, conn)
        
        conn.close()
        
        return {
            "total_runs": int(total_runs),
            "recent_runs_7d": int(recent_runs),
            "best_model": {
                "auc": float(best_metrics_dict.get('test_auc', 0)),
                "f1": float(best_metrics_dict.get('test_f1', 0)),
                "precision": float(best_metrics_dict.get('test_precision', 0)),
                "recall": float(best_metrics_dict.get('test_recall', 0))
            },
            "training_history": history.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Training stats error: {e}")
        return {
            "total_runs": 0,
            "recent_runs_7d": 0,
            "best_model": {"auc": 0, "f1": 0, "precision": 0, "recall": 0},
            "training_history": [],
            "error": str(e)
        }


@app.get("/api/explainability/importance")
async def get_feature_importance():
    """Get global feature importance"""
    try:
        # Try to load from saved file or calculate from model
        importance_file = 'monitoring/reports/explainability/feature_importance.csv'
        
        if os.path.exists(importance_file):
            df = pd.read_csv(importance_file)
            return {
                "features": df.to_dict('records'),
                "count": len(df)
            }
        else:
            # Generate from model
            if 'model_a' in models:
                importance_dict = models['model_a'].get_score(importance_type='gain')
                
                features = []
                for fname, score in importance_dict.items():
                    features.append({
                        'feature': fname,
                        'importance': float(score)
                    })
                
                # Normalize
                total = sum(f['importance'] for f in features)
                for f in features:
                    f['importance'] = f['importance'] / total if total > 0 else 0
                
                features.sort(key=lambda x: x['importance'], reverse=True)
                return {"features": features[:15], "count": len(features)}
            else:
                return {"features": [], "count": 0, "message": "No model available"}
                
    except Exception as e:
        print(f"Feature importance error: {e}")
        return {"features": [], "count": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config['serving']['host'], 
        port=config['serving']['port']
    )