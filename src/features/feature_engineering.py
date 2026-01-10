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
            
        self.base_numeric_features = self.config['data']['numeric_features'].copy()
        self.base_categorical_features = self.config['data']['categorical_features'].copy()
        self.target = self.config['data']['target']
        
        # Derived features
        self.derived_numeric = [
            'revenue_per_user', 'engagement_score', 'support_burden', 
            'days_inactive_ratio', 'total_contract_value', 'payment_reliability'
        ]
        self.derived_categorical = ['nps_category']
        
        self.numeric_features = self.base_numeric_features + self.derived_numeric
        self.categorical_features = self.base_categorical_features + self.derived_categorical
        
        self.preprocessor = None
     #derived features   
    def create_features(self, df):
        df = df.copy()
        
        df['revenue_per_user'] = df['monthly_revenue'] / np.maximum(df['team_size'], 1) #revenue per team member
        
        df['engagement_score'] = (  #engagement scores
            df['logins_per_month'] * 0.3 +
            df['feature_usage_depth'] * 100 * 0.3 +
            (df['api_calls_per_month'] / 1000) * 0.4
        )
        
        df['support_burden'] = df['support_tickets'] * df['avg_ticket_resolution_days']
        
        df['days_inactive_ratio'] = df['days_since_last_login'] / np.maximum(df['account_age_days'], 1)
        
        df['total_contract_value'] = df['monthly_revenue'] * df['contract_length_months']
        
        df['payment_reliability'] = 1 / (1 + df['payment_delays'])
        
        df['nps_category'] = pd.cut(df['nps_score'], bins = [-1, 6, 8, 10],
                                    labels = ['Detractor', 'Passive', 'Promoter'])
        
        return df
    
    
    def build_preprocessor(self):
        numeric_transformer = StandardScaler()
        
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            
            remainder='drop'
        )
        
        return self.preprocessor

    def fit_transform(self, df):
        df = self.create_features(df)
        
        X = df[self.numeric_features + self.categorical_features]
        Y = df[self.target] if self.target in df.columns else None
        
        if self.preprocessor is None:
            self.build_preprocessor()
            
        X_transformed = self.preprocessor.fit_transform(X)
        
        feature_names = self._get_feature_names()
        
        return X_transformed, Y, feature_names

    def transform(self, df):
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first")
        
        df = self.create_features(df)
        
        X = df[self.numeric_features + self.categorical_features]
        Y = df[self.target] if self.target in df.columns else None
        
        X_transformed = self.preprocessor.transform(X)
        
        feature_names = self._get_feature_names()
        
        return X_transformed, Y, feature_names

    def _get_feature_names(self):
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_features = transformer.get_feature_names_out(features)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend(features)
                    
        return feature_names

    def save_preprocessor(self, path='models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
        print(f"#Saved preprocessor to {path}")
        
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load fitted preprocessor"""
        self.preprocessor = joblib.load(path)
        print(f"✓ Loaded preprocessor from {path}")
        return self.preprocessor

# Example usage
if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('data/raw/train_data.csv')
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Fit and transform
    X_train, y_train, feature_names = fe.fit_transform(train_df)
    
    print(f"✓ Transformed shape: {X_train.shape}")
    print(f"✓ Number of features: {len(feature_names)}")
    print(f"✓ Target distribution: {np.bincount(y_train)}")
    
    # Save
    fe.save_preprocessor()