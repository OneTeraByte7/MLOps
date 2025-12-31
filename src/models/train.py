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
    
    def prepare_data(self, train_df, test_df):
        X_train, Y_train, feature_names = self.feature_enginerr.fit_transform(train_df)
        
        X_test, Y_test, _ = self.feature_engineer.transform(test_df)
        
        val_split = self.config['model']['vlidation_split']
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size = val_split, random_state = 42, stratify = Y_train
        )
        
        print(f" Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, feature_names
    
    def train_model(self, X_train, Y_train, X_val, Y_val):
        
        dtrain = xgb.DMatrix(X_train, label = Y_train)
        dval = xgb.Dmatrix(X_val, label = Y_val)
        
        params = self.model_params.copy()
        early_stopping = self.config['mode']['early_stopping_rounds']
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round = params['n_estimtors'],
            evals = evals,
            early_stopping_rounds = early_stopping,
            evals_result = evals_result,
            verbose_eval = False
        )
        
        print(f"Model trained (best iteration: {self.model.best_iteration})")
        
        return evals_result
    
    
    def evalute_models(self, X, Y, dataset_name = 'test'):
        dmatrix = xgb.DMatrix(X)
        Y_pred_proba = self.model.predict(dmatrix)
        Y_pred = (Y_pred_proba > 0.5).astype(int)
        
        metrics = {
            f'{dataset_name}_auc': roc_auc_score(Y, Y_pred_proba),
            f'{dataset_name}_accuracy': accuracy_score(Y, Y_pred),
            f'{dataset_name}_precision': precision_score(Y, Y_pred),
            f'{dataset_name}_reacll': recall_score(Y, Y_pred),
            f'{dataset_name}_fl': f1_score(Y,Y_pred),
        }
        
        cm = confusion_matrix(Y, Y_pred)
        
        print(f"\n{dataset_name.upper()} Metrics:")
        for metric, value in metrics.items():
            print(f" {metric}: {value:.4f}")
            
        print(f"\n Confusion Matrix:")
        print(f" TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f" FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return metrics, cm
            