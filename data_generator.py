import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

#Generates real SaaS data
def generator_churn_data(n_customers=10000, output_dir='data/raw'):
    os.makedirs(output_dir, exist_ok=True)
    
    customer_ids = [f"CUST_{i:06d}" for i in range(n_customers)]
    
    account_age = np.random.exponential(scale=365, size = n_customers).astype(int)
    account_age = np.cleint(account_age, 30, 2000)
    
    tier_probs = [0.15, 0.35, 0.50]
    subscription_tier = np.random.choice(['Enterprise','Professional', "Starter"], size = n_customers, p = tier_probs)
    
    revenue_map = {'Enterprise':(500, 150), 'Professional':(200, 50), 'Starter':(50,20)}
    monthly_revenue = np.array([np.random.normal(revenue_map[tier][0], revenue_map[tier][1]) for tier in subscription_tier])
    
    monthly_revenue = np.clip(monthly_revenue, 10, 2000)
    
    logins_per_month = np.random.poission(lam=20, size=n_customers)
    logins_per_month = np.clip(logins_per_month, 0, 100)
    
    feature_usage_depth = np.random.beta(2, 5, size=n_customers)
    
    support_tickets = np.random.poisson(lam=2, size=n_customers)
    avg_tickets_resolution_days = np.random.gamma(shape=2, scale=2, size=n_customers)
    
    
    nps_score = np.random.choice(range(11), size = n_customers, p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.15, 0.15, 0.15, 0.15, 0.10])
    
    payment_delays = np.random.poissons(lam=0.5, size = n_customers)
    
    contract_length = np.random.choice([1, 6, 12, 24], size = n_customers, p = [0.40, 0.25, 0.25, 0.10])
    
    team_size = np.random.lognormal(mean=1.5, sigma=1, size = n_customers).astype(int)
    team_size = np.clip(team_size, 0, 100000)
    
    api_calls = np.random.lognormal(mean = 6, sigma=2, size = n_customers).astype(int)
    api_calls = np.clip(api_calls, 0, 100000)
    
    days_since_last_login = np.random.exponential(scale=10, size=n_customers).astype(int)
    days_since_alst_login = np.clip(days_since_last_login, 0, 90)
    
    churn_score = (
        -0.3 * (logins_per_month/100) +
        -0.2 * feature_usage_depth + 
        0.15 * (support_tickets/10)+
        -0.15 * (nps_score/10)+
        0.2 * (payment_delays/5)+
        0.25 * (days_since_last_login/90)+
        -0.1*(np.log1p(api_calls)/np.log1p(100000)) +
        0.1*(subscription_tier == 'Starter').astype(int)+\
            np.random.normal(0, 0.3, size = n_customers)
    )
    
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churned =  (churn_score > np.random.random(n_customers)).astype(int)
    
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'account_age_days': account_age,
        'subscription_tier': subscription_tier,
        'monthly_revenue': monthly_revenue,
        'logins_per_month': logins_per_month,
        'feature_usage_depth': feature_usage_depth,
        'support_tickets': support_tickets,
        'avg_ticket_resolution_days': avg_tickets_resolution_days,
        'nps_score': nps_score,
        'payment_delays': payment_delays,
        'contract_length_months': contract_length,
        'team_size': team_size,
        'api_calls_per_month': api_calls,
        'days_since_last_login': days_since_last_login,
        'churned': churned
    })