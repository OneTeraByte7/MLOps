"""
Populate dashboard with real customer data from Supabase.
This script queries the `customers` table in Supabase and sends them to the API's /predict endpoint.
It no longer generates synthetic/random data.
"""
import os
import time
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

API_URL = os.environ.get('API_URL', 'http://localhost:8000')
SUPABASE_PROJECT_ID = os.environ.get('SUPABASE_PROJECT_ID')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY') or os.environ.get('SUPABASE_ANON_KEY')

def fetch_customers_from_supabase(limit=100):
    if not (SUPABASE_PROJECT_ID and SUPABASE_KEY):
        print('✗ Supabase credentials not found in .env')
        return []

    sb_url = f"https://{SUPABASE_PROJECT_ID}.supabase.co"
    sb = create_client(sb_url, SUPABASE_KEY)

    try:
        res = sb.table('customers').select('*').limit(limit).execute()
        data = res.data if hasattr(res, 'data') else res
        if not data:
            print('⚠ No customers found in Supabase `customers` table')
            return []
        print(f"✓ Fetched {len(data)} customers from Supabase")
        return data
    except Exception as e:
        print(f'✗ Supabase query failed: {e}')
        return []

def test_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print('✓ API is healthy')
            return True
        print(f'✗ API health check failed: {response.status_code}')
        return False
    except Exception as e:
        print(f'✗ Cannot connect to API: {e}')
        return False

def make_predictions_from_customers(customers):
    if not customers:
        print('✗ No customers to predict on')
        return False

    successful = 0
    failed = 0

    for i, cust in enumerate(customers):
        payload = {
            'customer_id': cust.get('customer_id') or f"CUST_{i:06d}",
            'account_age_days': int(cust.get('account_age_days', 0)),
            'subscription_tier': cust.get('subscription_tier'),
            'monthly_revenue': float(cust.get('monthly_revenue', 0)),
            'logins_per_month': int(cust.get('logins_per_month', 0)),
            'feature_usage_depth': float(cust.get('feature_usage_depth', 0)),
            'support_tickets': int(cust.get('support_tickets', 0)),
            'avg_ticket_resolution_days': float(cust.get('avg_ticket_resolution_days', 0)),
            'nps_score': int(cust.get('nps_score', 0)),
            'payment_delays': int(cust.get('payment_delays', 0)),
            'contract_length_months': int(cust.get('contract_length_months', 12)),
            'team_size': int(cust.get('team_size', 1)),
            'api_calls_per_month': int(cust.get('api_calls_per_month', 0)),
            'days_since_last_login': int(cust.get('days_since_last_login', 0)),
        }

        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

        time.sleep(0.02)

    print(f"✓ Completed: {successful} successful, {failed} failed")
    return successful > 0

def main():
    print('=' * 70)
    print('MLOps Dashboard - Populate from Supabase')
    print('=' * 70)

    if not test_health():
        return

    customers = fetch_customers_from_supabase(limit=200)
    if not customers:
        print('No customers found. Populate Supabase `customers` table first.')
        return

    make_predictions_from_customers(customers)

    print('\nService URLs:')
    print('  React Frontend: http://localhost:5173')
    print('  API Docs: http://localhost:8000/docs')
    print('  MLflow UI: http://localhost:5000')

if __name__ == '__main__':
    main()
