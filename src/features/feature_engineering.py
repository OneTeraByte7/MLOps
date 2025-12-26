import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import yaml

class FeatureEngineer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.numeric_features = self.config['data']['numeric_features']
        self.categorical_features = self.config['data']['categorical_features']
        self.target = self.config['data']['target']
        
        
        self.preprocessor = None
     #derived features   
    def create_features(self, df):
        df = df.copy()
        
        df['revenue_per_user'] = df['monthly_revenue'] / df['team_size'] #revenue per team memebr
        
        df['engagement_score'] = (  #engagement scores
            df['logins_per_month'] * 0.3 +
            df['features_usage_depth'] * 100 * 0.3 +
            (df['api_calls_per_month'] / 1000) * 0.4
        )
        
        df['support_burden'] = df['support_tickets'] * df['avg_ticket_resolution_days']
        
        df['days_inactive_ratio'] = df['days_since_last_login'] / np.maximum(df['account_age_days'], 1)
        
        df['total_contract_value'] = df['monthly_revenue'] * df['contract_length_months']
        
        df['payment_reliability'] = 1 / (1 + df['payment_delays'])
        
        df['nps_category'] = pd.cut(df['nps_score'], bins = [-1, 6, 8, 10],
                                    labels = ['Detector', 'Passive', 'Promoter'])
        
        self.numeric_features_extend([
            'revenue_per_user', 'engagement_score', 'support_burden', 'days_inactive_ratio', ' total_contract_value',' payment_reliability'
        ])
        
        self.categorical_features_append('nps_category')
        
        return df
    
def build_preprocessor(self):
    numeric_transformer = StandardScaler()
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    
        
        