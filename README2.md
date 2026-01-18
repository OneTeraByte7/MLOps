# MLOps Churn Prediction — Quick Reference

A compact, practical reference for running, populating, and deploying the MLOps Churn Prediction project.

## Overview
- Production-oriented churn prediction pipeline: training, serving, drift monitoring, A/B testing, and a React dashboard.

## Quick Local Run

Prerequisites: Python 3.9+, Node.js/npm, Docker (optional).

1. Create and activate a Python virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

2. Install frontend dependencies

```bash
cd dashboard/react-frontend
npm install
cd -
```

3. Start all services (recommended)

```bash
python start_services.py
```

This launches MLflow (http://localhost:5000), the FastAPI backend (http://localhost:8000) and the React dashboard (http://localhost:5173).

4. Populate the dashboard with real or training data

Place your CSV under `data/test_data.csv` or `data/raw/churn_data.csv` (columns should match features used by `src/api/app.py`), then run:

```bash
python populate_dashboard.py
```

Or make ad-hoc predictions:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample_customer.json
```

## Where to look
- React Dashboard: http://localhost:5173
- API docs (OpenAPI): http://localhost:8000/docs
- MLflow UI: http://localhost:5000
- Dashboard overview endpoint: `GET /api/dashboard/overview`

## Docker (local / CI-friendly)

The repo does not include Dockerfiles by default. Example minimal Dockerfiles below to build images for the API and frontend.

Example `deploy/Dockerfile.api`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Example `deploy/Dockerfile.frontend` (build static with Vite):

```dockerfile
FROM node:18 as builder
WORKDIR /app
COPY dashboard/react-frontend/package*.json ./dashboard/react-frontend/
COPY dashboard/react-frontend ./dashboard/react-frontend
WORKDIR /app/dashboard/react-frontend
RUN npm install
RUN npm run build

FROM nginx:stable-alpine
COPY --from=builder /app/dashboard/react-frontend/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Simple `docker-compose.yml` (local testing):

```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: deploy/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./mlruns:/app/mlruns
  frontend:
    build:
      context: .
      dockerfile: deploy/Dockerfile.frontend
    ports:
      - "5173:80"
  mlflow:
    image: python:3.9-slim
    command: sh -c "pip install mlflow && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
```

Build and run:

```bash
docker-compose build
docker-compose up -d
```

## CI/CD (GitHub Actions) — minimal plan

- Run: lint, pytest, build images, push images, and deploy.
- Use `docker/login-action` and `docker/build-push-action` to push to GHCR/ECR/GCR.
- Optional `deploy` job: run `kubectl apply` or call your cloud provider deploy steps.

High-level steps in workflow:
1. Checkout code
2. Set up Python and Node
3. Install deps, run tests
4. Build & push Docker images
5. Deploy (optional)

## Kubernetes (production notes)

- Use a production MLflow backend (RDS/CloudSQL) and artifact store (S3/GCS) instead of local `sqlite` and `mlruns`.
- Create k8s `Deployment` + `Service` for `churn-api` and `churn-frontend` (or serve frontend via CDN). Add `Ingress`/`TLS`.
- Use `ConfigMap`/`Secret` for configuration and credentials.
- Use `HorizontalPodAutoscaler` for API autoscaling. Use liveness/readiness probes.

Quick deploy commands (once manifests exist):

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/churn-api-deployment.yaml
kubectl apply -f k8s/churn-frontend-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

## Key repo files
- `start_services.py` — starts MLflow, API, and React dashboard locally
- `populate_dashboard.py` — populates dashboard from `data/test_data.csv` or generated samples
- `src/api/app.py` — FastAPI backend and dashboard endpoints
- `dashboard/react-frontend` — React dashboard (Vite)

## Tips
- For production MLflow, use a managed DB + object store; `sqlite` is only for local testing.
- Keep `config/config.yaml` updated for serving ports, A/B testing, and feature lists.
- To view real dashboard predictions, ensure you run `populate_dashboard.py` after the API is running and your `data/test_data.csv` is in place.

---
Built by Soham — quick reference for running and deploying the MLOps churn stack.

## Supabase Integration (recommended)

1. Create a Supabase project and a Storage bucket named `mlflow-artifacts`.
2. Run the SQL migration in `deploy/supabase_migrations.sql` from the Supabase SQL editor to create `predictions` and `drift_reports` tables.
3. Add these vars to your `.env` (example):

```env
SUPABASE_PROJECT_ID=pbrwtwlmjbngcizakfav
SUPABASE_URL=postgresql://postgres:YOUR_PASSWORD@db.pbrwtwlmjbngcizakfav.supabase.co:5432/postgres
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
# Optional: also set MLFLOW_TRACKING_URI if MLflow runs on another host
MLFLOW_TRACKING_URI=http://mlflow-host:5000
```

4. Start the stack. `start_services.py` will detect `SUPABASE_URL` and start MLflow with the Postgres backend and configure the S3 endpoint for Supabase Storage.

5. Confirm predictions are stored in Supabase by running the SQL queries in the Supabase SQL editor (see `deploy/supabase_migrations.sql` and sample queries in the main README).

Notes:
- The app uses the MLflow Tracking API (`MlflowClient`) and the Supabase REST/API for prediction logging.
- For production, keep the service role key secret (use GitHub Secrets or Kubernetes Secrets) and rotate regularly.
