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
    
    def run_training_pipeline(self):
        with mlflow.star_run(run_name = f"training_{datetime.now().strtime('%Y%m%d_%H%M%S')}"):
            
            mlflow.lof_params(self.model_params)
            mlflow.log_params("validation_split", self.config['model']['validation_split'], self.config['model']['validation_split'])
                              
            train_df, test_df = self.load_data()
            
            mlflow.log_metric('train_samples', len(train_df))
            mlflow.log_metric('test_samples', len(test_df))
            mlflow.log_metric("train_churn_rate", train_df['churned'].mean())
            
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
                self.prepare_data(train_df, test_df)
            
            # Train model
            print("\n" + "="*50)
            print("Training Model...")
            print("="*50)
            
            evals_result = self.train_model(X_train, y_train, X_val, y_val)
            
            # Log training curves
            for dataset in ['train', 'val']:
                for metric, values in evals_result[dataset].items():
                    for i, value in enumerate(values):
                        mlflow.log_metric(f"{dataset}_{metric}", value, step=i)
            
            # Evaluate on validation set
            val_metrics, val_cm = self.evaluate_model(X_val, y_val, 'validation')
            for metric, value in val_metrics.items():
                mlflow.log_metric(metric, value)
            
            # Evaluate on test set
            test_metrics, test_cm = self.evaluate_model(X_test, y_test, 'test')
            for metric, value in test_metrics.items():
                mlflow.log_metric(metric, value)
            
            # Feature importance
            importance = self.model.get_score(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False).head(20)
            
            print("\nTop 20 Important Features:")
            print(importance_df.to_string(index=False))
            
            # Save artifacts
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, 'model.json')
            self.model.save_model(model_path)
            mlflow.log_artifact(model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
            self.feature_engineer.save_preprocessor(preprocessor_path)
            mlflow.log_artifact(preprocessor_path)
            
            # Save feature names
            feature_names_path = os.path.join(model_dir, 'feature_names.json')
            with open(feature_names_path, 'w') as f:
                json.dump(feature_names, f)
            mlflow.log_artifact(feature_names_path)
            
            # Save feature importance
            importance_path = os.path.join(model_dir, 'feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            
            # Log model to MLflow Model Registry
            mlflow.xgboost.log_model(
                self.model,
                "model",
                registered_model_name="churn_predictor"
            )
            
            
            
            