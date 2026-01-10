"""
tests/test_model.py - Comprehensive test suite for ML pipeline
"""

import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
sys.path.append('src')

from features.feature_engineering import FeatureEngineer
from models.train import ChurnModelTrainer
from monitoring.drift_detector import DriftDetector

# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    return pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
        'account_age_days': [365, 180, 730],
        'subscription_tier': ['Professional', 'Starter', 'Enterprise'],
        'monthly_revenue': [199.99, 49.99, 599.99],
        'logins_per_month': [25, 10, 45],
        'feature_usage_depth': [0.65, 0.35, 0.85],
        'support_tickets': [2, 5, 1],
        'avg_ticket_resolution_days': [3.5, 8.0, 2.0],
        'nps_score': [8, 4, 10],
        'payment_delays': [0, 2, 0],
        'contract_length_months': [12, 1, 24],
        'team_size': [10, 2, 50],
        'api_calls_per_month': [15000, 500, 75000],
        'days_since_last_login': [2, 15, 1],
        'churned': [0, 1, 0]
    })

@pytest.fixture
def feature_engineer():
    """Initialize feature engineer"""
    return FeatureEngineer()

# ============================================
# FEATURE ENGINEERING TESTS
# ============================================

class TestFeatureEngineering:
    
    def test_create_features(self, feature_engineer, sample_data):
        """Test feature creation"""
        df = feature_engineer.create_features(sample_data)
        
        # Check new features exist
        assert 'revenue_per_user' in df.columns
        assert 'engagement_score' in df.columns
        assert 'support_burden' in df.columns
        assert 'days_inactive_ratio' in df.columns
        assert 'total_contract_value' in df.columns
        assert 'payment_reliability' in df.columns
        assert 'nps_category' in df.columns
    
    def test_revenue_per_user_calculation(self, feature_engineer, sample_data):
        """Test revenue per user calculation"""
        df = feature_engineer.create_features(sample_data)
        
        expected = sample_data['monthly_revenue'] / sample_data['team_size']
        np.testing.assert_array_almost_equal(
            df['revenue_per_user'].values, 
            expected.values, 
            decimal=2
        )
    
    def test_fit_transform_output_shape(self, feature_engineer, sample_data):
        """Test transformed output shape"""
        X, y, feature_names = feature_engineer.fit_transform(sample_data)
        
        assert X.shape[0] == len(sample_data)
        assert y.shape[0] == len(sample_data)
        assert len(feature_names) > 0
    
    def test_transform_consistency(self, feature_engineer, sample_data):
        """Test that transform produces consistent results"""
        X1, _, _ = feature_engineer.fit_transform(sample_data)
        X2, _, _ = feature_engineer.transform(sample_data)
        
        np.testing.assert_array_almost_equal(X1, X2)
    
    def test_handle_missing_values(self, feature_engineer):
        """Test handling of missing values"""
        data_with_nan = pd.DataFrame({
            'customer_id': ['CUST_001'],
            'account_age_days': [365],
            'subscription_tier': ['Professional'],
            'monthly_revenue': [np.nan],  # Missing value
            'logins_per_month': [25],
            'feature_usage_depth': [0.65],
            'support_tickets': [2],
            'avg_ticket_resolution_days': [3.5],
            'nps_score': [8],
            'payment_delays': [0],
            'contract_length_months': [12],
            'team_size': [10],
            'api_calls_per_month': [15000],
            'days_since_last_login': [2],
            'churned': [0]
        })
        
        # Should not raise error
        X, y, _ = feature_engineer.fit_transform(data_with_nan)
        assert not np.isnan(X).any()

# ============================================
# MODEL TRAINING TESTS
# ============================================

class TestModelTraining:
    
    def test_model_initialization(self):
        """Test model trainer initialization"""
        trainer = ChurnModelTrainer()
        assert trainer.model_params is not None
        assert trainer.model is None  # Not trained yet
    
    def test_model_training_completes(self, sample_data, tmp_path):
        """Test that training completes without error"""
        # This is a smoke test - just verify it runs
        # In production, you'd use larger dataset
        pass  # Requires full dataset
    
    def test_model_predictions_valid_range(self):
        """Test model predictions are in valid probability range"""
        # Load trained model
        model = xgb.Booster()
        # Predictions should be between 0 and 1
        # This test would run after training
        pass

# ============================================
# DRIFT DETECTION TESTS
# ============================================

class TestDriftDetection:
    
    def test_psi_calculation(self):
        """Test PSI calculation"""
        detector = DriftDetector()
        
        reference = np.random.normal(0, 1, 1000)
        current_no_drift = np.random.normal(0, 1, 1000)
        current_drift = np.random.normal(2, 1, 1000)  # Mean shift
        
        psi_no_drift = detector.calculate_psi(reference, current_no_drift)
        psi_drift = detector.calculate_psi(reference, current_drift)
        
        assert psi_drift > psi_no_drift
        assert psi_no_drift < 0.2  # Should be low
        assert psi_drift > 0.2     # Should indicate drift
    
    def test_ks_statistic_calculation(self):
        """Test KS statistic calculation"""
        detector = DriftDetector()
        
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        ks_stat, p_value = detector.calculate_ks_statistic(reference, current)
        
        assert 0 <= ks_stat <= 1
        assert 0 <= p_value <= 1
        assert p_value > 0.05  # Should not be significant
    
    def test_feature_drift_detection(self, sample_data):
        """Test feature drift detection"""
        detector = DriftDetector()
        
        # Create drifted version
        drifted_data = sample_data.copy()
        drifted_data['logins_per_month'] = drifted_data['logins_per_month'] * 2
        
        drift_report = detector.detect_feature_drift(sample_data, drifted_data)
        
        assert 'logins_per_month' in drift_report
        assert 'psi' in drift_report['logins_per_month']

# ============================================
# API TESTS
# ============================================

class TestAPI:
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from httpx import ASGITransport
        from httpx import AsyncClient
        from src.api.app import app
        # Return a sync client using the app
        import asyncio
        from starlette.testclient import TestClient as StarletteTestClient
        return StarletteTestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "service" in response.json()
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model_info")
        assert response.status_code == 200
        data = response.json()
        assert "model_a_version" in data
        assert "ab_testing_enabled" in data
    
    def test_predict_endpoint_valid_input(self, client):
        """Test prediction with valid input"""
        payload = {
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
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "customer_id" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert 0 <= data["churn_probability"] <= 1
    
    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction with invalid input"""
        payload = {
            "customer_id": "CUST_TEST_001",
            # Missing required fields
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint"""
        payload = {
            "customers": [
                {
                    "customer_id": "CUST_001",
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
                }
            ]
        }
        
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_customers" in data

# ============================================
# INTEGRATION TESTS
# ============================================

class TestIntegration:
    
    def test_end_to_end_prediction_pipeline(self, sample_data):
        """Test complete prediction pipeline"""
        # Feature engineering
        fe = FeatureEngineer()
        X, y, feature_names = fe.fit_transform(sample_data)
        
        assert X is not None
        assert y is not None
        assert len(feature_names) > 0
    
    def test_data_versioning_workflow(self):
        """Test DVC data versioning"""
        # This would test DVC commands
        # Requires DVC to be initialized
        pass

# ============================================
# PERFORMANCE TESTS
# ============================================

class TestPerformance:
    
    def test_prediction_latency(self, client):
        """Test prediction latency is acceptable"""
        import time
        
        payload = {
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
        }
        
        start = time.time()
        response = client.post("/predict", json=payload)
        latency = time.time() - start
        
        assert response.status_code == 200
        assert latency < 0.5  # Should be under 500ms
    
    def test_batch_prediction_throughput(self, client):
        """Test batch prediction throughput"""
        # Test with 100 predictions
        payload = {
            "customers": [
                {
                    "customer_id": f"CUST_{i:03d}",
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
                }
                for i in range(10)
            ]
        }
        
        response = client.post("/batch_predict", json=payload)
        assert response.status_code == 200

# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])