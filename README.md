# MLOps Churn Prediction Pipeline
<img width="1024" height="338" alt="Gemini_Generated_Image_hs3nijhs3nijhs3n" src="https://github.com/user-attachments/assets/36164c6e-b91b-4066-baf6-971e71f4fd79" />

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.8.0-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/Status-completed-green)

**A production-ready, end-to-end MLOps pipeline for customer churn prediction in SaaS businesses.**

This project demonstrates best practices for deploying machine learning models to production, including automated training, monitoring, drift detection, A/B testing, and CI/CD.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Automated Retraining](#automated-retraining)
- [API Documentation](#api-documentation)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment](#deployment)

---

## ✨ Features

### Core MLOps Capabilities

✅ **Data Versioning (DVC)**
- Track data changes over time
- Reproducible experiments
- Easy rollback to previous versions

✅ **Experiment Tracking (MLflow)**
- Log parameters, metrics, and artifacts
- Compare model performance across runs
- Model registry for version control

✅ **Automated Training Pipeline**
- End-to-end feature engineering
- XGBoost model training with early stopping
- Comprehensive evaluation metrics

✅ **Model Serving API (FastAPI)**
- REST API for real-time predictions
- Batch prediction support
- Health checks and monitoring endpoints

✅ **A/B Testing Infrastructure**
- Compare multiple model versions in production
- Configurable traffic splitting
- Performance tracking per model

✅ **Drift Detection & Monitoring**
- Feature drift detection (PSI, KS test)
- Label drift monitoring
- Performance degradation alerts
- Prometheus metrics integration

✅ **Automated Retraining**
- Trigger-based retraining (drift, performance, schedule)
- Model validation before deployment
- Safe rollback mechanisms

✅ **CI/CD Pipeline (GitHub Actions)**
- Automated testing and validation
- Model training on schedule
- Automated deployment to production

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION & VERSIONING                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Raw Data  │───▶│     DVC     │───▶│  Versioned  │        │
│  │   Sources   │    │  Tracking   │    │    Data     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│  ┌──────────────────────────────────────────────────┐          │
│  │  • Derived Features  • Encoding  • Scaling       │          │
│  └──────────────────────────────────────────────────┘          │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   XGBoost   │───▶│   MLflow    │───▶│    Model    │        │
│  │  Training   │    │  Tracking   │    │  Registry   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL VALIDATION                             │
│  ┌──────────────────────────────────────────────────┐          │
│  │  • Performance Tests  • Drift Checks             │          │
│  └──────────────────────────────────────────────────┘          │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DEPLOYMENT & SERVING                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   FastAPI   │───▶│  A/B Test   │───▶│   Docker    │        │
│  │     API     │    │  Controller │    │  Container  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              MONITORING & FEEDBACK LOOP                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Prometheus  │───▶│    Drift    │───▶│  Automated  │        │
│  │  Metrics    │    │  Detection  │    │ Retraining  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# Git
git --version

# Docker (optional)
docker --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mlops-churn-prediction.git
cd mlops-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize tools**
```bash
# Initialize DVC
dvc init

# Start MLflow server (in separate terminal)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

### Generate Data & Train First Model

```bash
# Generate synthetic SaaS customer data
python src/data/data_generator.py

# Train initial model
python src/models/train.py

# View results in MLflow UI: http://localhost:5000
```

### Start API Server

```bash
# Start the API
python src/api/app.py

# API available at: http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Test Prediction

```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_TEST_001",
    "account_age_days": 365,
    "subscription_tier": "Professional",
    "monthly_revenue": 199.99,
    "logins_per_month": 25,
    "feature_usage_depth": 0.65,
    "support_tickets": 2,
    "avg_ticket_resolution_days": 3.5,
    "nps_score": 8,
    "payment_delays": 0,
    "contract_length_months": 12,
    "team_size": 10,
    "api_calls_per_month": 15000,
    "days_since_last_login": 2
  }'
```

---

## 📁 Project Structure

```
mlops-churn-prediction/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml      # CI/CD pipeline
├── config/
│   └── config.yaml              # Configuration
├── data/
│   ├── raw/                     # Raw data (DVC tracked)
│   ├── processed/               # Processed features
│   └── versions/                # Data versions
├── models/                      # Saved models
│   ├── model.json              # Latest model
│   ├── preprocessor.pkl        # Feature preprocessor
│   └── feature_names.json      # Feature metadata
├── src/
│   ├── data/
│   │   └── data_generator.py   # Synthetic data generation
│   ├── features/
│   │   └── feature_engineering.py  # Feature pipeline
│   ├── models/
│   │   └── train.py            # Training script
│   ├── monitoring/
│   │   └── drift_detector.py   # Drift detection
│   └── api/
│       └── app.py              # FastAPI application
├── pipelines/
│   ├── training_pipeline.py    # Complete training pipeline
│   └── retraining_pipeline.py  # Automated retraining
├── tests/                      # Unit tests
├── monitoring/
│   ├── reports/               # Drift reports
│   └── deployments/           # Deployment logs
├── Dockerfile                 # Container definition
├── docker-compose.yml        # Multi-service setup
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

---

## 📖 Usage Guide

### 1. Training Pipeline

```bash
# Full training pipeline with MLflow tracking
python src/models/train.py

# View experiment results
# Open http://localhost:5000 in browser
```

### 2. Making Predictions

**Single Prediction:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "customer_id": "CUST_123",
        "account_age_days": 450,
        # ... other features
    }
)

print(response.json())
# {
#   "customer_id": "CUST_123",
#   "churn_probability": 0.234,
#   "churn_prediction": "No",
#   "risk_level": "Low",
#   "model_version": "v1.0.0",
#   "confidence_score": 0.532
# }
```

**Batch Predictions:**
```python
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={
        "customers": [
            {customer_1_features},
            {customer_2_features},
            # ...
        ]
    }
)
```

### 3. Drift Detection

```bash
# Run drift detection manually
python src/monitoring/drift_detector.py

# View report
cat monitoring/reports/latest_report.json
```

### 4. Automated Retraining

```bash
# Check retraining triggers (won't retrain unless needed)
python pipelines/retraining_pipeline.py

# Force retraining
python pipelines/retraining_pipeline.py --force
```

---

## 📊 Monitoring & Drift Detection

### Drift Detection Metrics

**Population Stability Index (PSI)**
- PSI < 0.1: No significant change
- 0.1 ≤ PSI < 0.2: Moderate change
- PSI ≥ 0.2: Significant change (trigger retraining)

**Kolmogorov-Smirnov Test**
- Statistical test for distribution differences
- p-value < 0.05 indicates significant drift

### Prometheus Metrics

The API exposes metrics at `/metrics`:

```
# Model predictions
predictions_total{model_version="v1.0.0",prediction="Yes"} 152
predictions_total{model_version="v1.0.0",prediction="No"} 848

# Prediction latency
prediction_latency_seconds_sum 12.34
prediction_latency_seconds_count 1000

# Model version
model_version{version="model_a"} 1.0

# Drift score
drift_score 0.08
```

---

## 🔄 Automated Retraining

### Retraining Triggers

1. **Drift Trigger**: Activated when PSI > threshold for multiple features
2. **Performance Trigger**: Activated when AUC drops below threshold
3. **Schedule Trigger**: Weekly retraining (configurable via cron)
4. **Manual Trigger**: Force retraining via CLI or API

### Retraining Process

```
1. Check Triggers → 2. Collect Data → 3. Train Model
                                           ↓
6. Deploy ← 5. Validate ← 4. Evaluate Performance
```

### Safety Mechanisms

- Model validation before deployment
- Performance comparison with production model
- Automated rollback if new model underperforms
- Blue-green deployment strategy

---

## 🔌 API Documentation

### Endpoints

**Health Check**
```
GET /health
```

**Model Information**
```
GET /model_info
```

**Single Prediction**
```
POST /predict
Content-Type: application/json

{
  "customer_id": "string",
  "account_age_days": int,
  "subscription_tier": "string",
  ...
}
```

**Batch Prediction**
```
POST /batch_predict
Content-Type: application/json

{
  "customers": [...]
}
```

**Metrics**
```
GET /metrics
```

**Interactive Documentation**: http://localhost:8000/docs

---

## ⚙️ CI/CD Pipeline

### GitHub Actions Workflow

**Triggers:**
- Push to main/develop
- Pull requests
- Weekly schedule (Sunday 2 AM UTC)
- Manual dispatch

**Jobs:**
1. **Quality Check**: Linting, formatting, unit tests
2. **Data Validation**: Schema checks, data quality
3. **Model Training**: Train new model with MLflow
4. **Model Validation**: Performance tests, benchmarks
5. **Drift Detection**: Check for data/model drift
6. **Automated Retraining**: Retrain if needed
7. **Deploy API**: Build Docker, deploy to production
8. **Smoke Tests**: Production health checks

---

## 🐳 Deployment

### Docker Deployment

**Single Container:**
```bash
docker build -t churn-api .
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name churn-api \
  churn-api
```

**Full Stack (with monitoring):**
```bash
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

### Cloud Deployment

**AWS (ECS/Fargate)**
- Use provided Dockerfile
- Deploy via AWS CDK or CloudFormation
- Set up ALB for load balancing

**GCP (Cloud Run)**
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/churn-api
gcloud run deploy --image gcr.io/PROJECT-ID/churn-api
```

**Azure (Container Instances)**
```bash
az container create \
  --resource-group myResourceGroup \
  --name churn-api \
  --image myregistry.azurecr.io/churn-api:latest
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model.py -v
```

---

## 📈 Performance

- **Prediction Latency**: ~50ms per request
- **Throughput**: ~200 predictions/second (single container)
- **Model Size**: ~2MB
- **Memory Usage**: ~500MB

---

## 🛠️ Configuration

Edit `config/config.yaml` to customize:

- Model hyperparameters
- Feature engineering
- Drift thresholds
- Retraining triggers
- A/B testing splits
- Deployment strategy

---

## 📝 Best Practices Implemented

✅ **Separation of Concerns**: Clear boundaries between data, models, API, monitoring
✅ **Configuration Management**: Centralized YAML configuration
✅ **Version Control**: DVC for data, Git for code, MLflow for models
✅ **Automated Testing**: Unit tests, integration tests, smoke tests
✅ **Monitoring**: Prometheus metrics, drift detection, alerting
✅ **Documentation**: Inline comments, API docs, comprehensive README
✅ **Reproducibility**: Fixed seeds, versioned dependencies
✅ **Scalability**: Containerized, stateless API, horizontal scaling ready

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙋 FAQ

**Q: How often should I retrain the model?**
A: It depends on drift detection results. Weekly is a good starting point for SaaS churn.

**Q: Can I use my own data?**
A: Yes! Replace the data generator with your data loading logic, ensuring the same schema.

**Q: How do I add new features?**
A: Update `feature_engineering.py` and `config.yaml`, then retrain.

**Q: What if retraining degrades performance?**
A: The pipeline automatically validates new models and rejects worse performers.

---

## 📬 Contact

- **Email**: ml-team@example.com
- **Issues**: GitHub Issues
- **Slack**: #mlops-churn

---

**Built with ❤️ for production ML**
