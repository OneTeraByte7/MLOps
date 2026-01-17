"""
Test API and populate dashboard with real customer data
"""
import requests
import random
import time
import pandas as pd
import os
from datetime import datetime

API_URL = "http://localhost:8000"

def load_training_data():
    """Load real customer data from training set"""
    try:
        # Try to load test data
        if os.path.exists('data/test_data.csv'):
            df = pd.read_csv('data/test_data.csv')
            print(f"✓ Loaded {len(df)} customers from test data")
            return df
        
        # Try to load raw data
        if os.path.exists('data/raw/churn_data.csv'):
            df = pd.read_csv('data/raw/churn_data.csv')
            print(f"✓ Loaded {len(df)} customers from raw data")
            return df.sample(n=min(100, len(df)))
        
        return None
    except Exception as e:
        print(f"⚠ Could not load training data: {e}")
        return None

def generate_sample_customers(num_customers=100):
    """Generate sample customer data"""
    subscription_tiers = ["Starter", "Professional", "Enterprise"]
    customers = []
    
    for i in range(num_customers):
        customers.append({
            "account_age_days": random.randint(30, 1000),
            "subscription_tier": random.choice(subscription_tiers),
            "monthly_revenue": round(random.uniform(50, 500), 2),
            "logins_per_month": random.randint(1, 60),
            "feature_usage_depth": round(random.uniform(0.1, 1.0), 2),
            "support_tickets": random.randint(0, 10),
            "avg_ticket_resolution_days": round(random.uniform(0.5, 7), 1),
            "nps_score": random.randint(0, 10),
            "payment_delays": random.randint(0, 3),
            "contract_length_months": random.choice([1, 6, 12, 24]),
            "team_size": random.randint(1, 50),
            "api_calls_per_month": random.randint(100, 50000),
            "days_since_last_login": random.randint(0, 30)
        })
    
    return customers

def test_health():
    """Test API health"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ API is healthy")
            print(f"  {response.json()}")
            return True
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print(f"\nMake sure the API is running:")
        print(f"  python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000")
        return False

def make_predictions(num_predictions=100):
    """Make predictions using real or generated data"""
    print(f"\n📊 Making {num_predictions} predictions...")
    
    # Try to load real training data first
    training_data = load_training_data()
    
    if training_data is not None:
        print("  Using REAL customer data from training set")
        customer_templates = []
        
        # Convert DataFrame rows to customer dicts
        for idx, row in training_data.head(num_predictions).iterrows():
            customer_templates.append({
                "account_age_days": int(row.get('account_age_days', random.randint(30, 1000))),
                "subscription_tier": str(row.get('subscription_tier', 'Professional')),
                "monthly_revenue": float(row.get('monthly_revenue', random.uniform(50, 500))),
                "logins_per_month": int(row.get('logins_per_month', random.randint(1, 60))),
                "feature_usage_depth": float(row.get('feature_usage_depth', random.uniform(0.1, 1.0))),
                "support_tickets": int(row.get('support_tickets', random.randint(0, 10))),
                "avg_ticket_resolution_days": float(row.get('avg_ticket_resolution_days', random.uniform(0.5, 7))),
                "nps_score": int(row.get('nps_score', random.randint(0, 10))),
                "payment_delays": int(row.get('payment_delays', random.randint(0, 3))),
                "contract_length_months": int(row.get('contract_length_months', 12)),
                "team_size": int(row.get('team_size', random.randint(1, 50))),
                "api_calls_per_month": int(row.get('api_calls_per_month', random.randint(100, 50000))),
                "days_since_last_login": int(row.get('days_since_last_login', random.randint(0, 30)))
            })
    else:
        print("  Using generated sample data")
        customer_templates = generate_sample_customers(num_predictions)
    
    successful = 0
    failed = 0
    
    for i in range(num_predictions):
        try:
            customer = customer_templates[i % len(customer_templates)].copy()
            customer["customer_id"] = f"CUST_{i:06d}"
            
            response = requests.post(
                f"{API_URL}/predict",
                json=customer,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                successful += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  ✓ {i + 1}/{num_predictions} predictions completed")
                    print(f"    Last: {result['customer_id']} - Risk: {result['risk_level']}, Prob: {result['churn_probability']:.2%}")
            else:
                failed += 1
                if failed <= 3:
                    print(f"  ✗ Failed prediction {i + 1}: {response.status_code}")
                
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ✗ Error on prediction {i + 1}: {e}")
        
        time.sleep(0.05)
    
    print(f"\n✓ Completed: {successful} successful, {failed} failed")
    return successful > 0

def check_dashboard_data():
    """Check if dashboard has data"""
    print("\n📈 Checking dashboard data...")
    
    try:
        response = requests.get(f"{API_URL}/api/dashboard/overview", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"  Total predictions: {data['total_predictions']}")
            print(f"  Churn rate: {data['churn_rate']:.2%}")
            print(f"  High risk customers: {data['high_risk_customers']}")
            print(f"  Avg confidence: {data['avg_confidence']:.2%}")
            
            if data['total_predictions'] > 0:
                print("\n✓ Dashboard has data! Open http://localhost:5173 to view")
                return True
            else:
                print("\n⚠ Dashboard has no data yet")
                return False
        else:
            print(f"  ✗ Dashboard API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Cannot check dashboard: {e}")
        return False

def main():
    print("=" * 70)
    print("MLOps Dashboard - Real Data Population Script")
    print("=" * 70)
    
    if not test_health():
        return
    
    print("\nThis will use REAL customer data from your training set")
    print("to make predictions and populate the dashboard.")
    
    if make_predictions(100):
        time.sleep(1)
        check_dashboard_data()
    else:
        print("\n✗ Failed to create predictions")
    
    print("\n" + "=" * 70)
    print("Dashboard URLs:")
    print("  React Frontend: http://localhost:5173")
    print("  API Docs: http://localhost:8000/docs")
    print("  MLflow UI: http://localhost:5000")
    print("=" * 70)

if __name__ == "__main__":
    main()
