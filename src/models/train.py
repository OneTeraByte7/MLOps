import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import mlflow
import mlflow.xgboost
import yaml
import joblib
import os
import json
from datetime import datetime
import sys
sys.path.append('src')

from features.feature_engineering import FeatureEngineer

class ChurnModelTrainer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_params = self.config['model']['params']
        self.model = None
        self.feature_engineer = FeatureEngineer(config_path)
        
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiemnt_name'])
        
    def load_data(self):
        data_dir = self.config['data']['raw_dir']
        
        train_path = os.path.join(data_dir, self.config['data']['train_file'])
        test_path = os.path.join(data_dir, self.cofig['data']['test_file'])
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded{len(train_df)} training samples")
        print(f" Loaded {len(test_df)} test samples")
        
        return train_df, test_df