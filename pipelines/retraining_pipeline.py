import sys
sys.path.append('src')

import pandas as pd
import json 
import yaml
from datetime import datetime, timedelta
import os
from matplotlib import path

from models.train import ChurnModelTrainer
from monitoring.drift_detector import DriftDetector
import mlflow

class RetrainingPipeline:
    def __init__(self, config_path = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load()
            
        self.trainer = ChurnModelTrainer(config_path)
        self.detector = DriftDetector(config_path)
        
        self.retraining_config = self.config['retraining']
        
    def check_retraining_triggers(self):
        #checks if retraining needs to be triggered
        triggers = {
            'drift_detetcted': False,
            'performace_degraded': False,
            'scheduled': False,
            'manual': False 
        }
        
        reasons = []
        
        report_path = 'monitoring/reports/latest_report.json'
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                drift_report = json.load(f)
                
            if self.retraining_config['drift_trigger']:
                if drift_report['drift_features_count'] > 0:
                    triggers['drift_detected'] = True 
                    reasons.append(f"Data drift detected in {drift_report['drift_features_count']} features")
                    
                    if self.retraining_config['performance_trigger']:
                        if 'model_performance' in drift_report:
                            perf =drift_report['model_performance']
                            if perf['performance_degraded']:
                                triggers['performace_degraded'] = True
                                reasons.append(f"Perfromace degraded: AUC = {perf['auc']:.3f}")
                                
                    if self.retraining_config['scheduled_trigger']:
                        last_training = self._get_last_training_time()
                        if last_training:
                            days_since = (datetime.now() - last_training).days
                            
                            if days_since >= 7:
                                triggers['scheduled'] = True
                                reasons.append(f"Scheduled retraining (last trained {days_since} days ago)")
                                
                    should_retrain = any(triggers.values())
                    
                    return should_retrain, triggers, reasons
                
    def _get_last_training_time(self):
        
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        runs = mlflow.search_runs(order_by = ["start_time DESC"], max_results = 1)
        
        if not runs.empty:
            return pd.to_datetime(runs.iloc[0]['start_time'])
        
        return None
    
    def collect_training_data(self):
        
        print("\n" + "=" * 60)
        print("COLLECTING TRAINING DATA")
        print("=" * 60)
        
        train_df = pd.read_csv('data/raw/train_data.csv')
        test_df = pd.read_csv('data/raw/test_data.csv')
        
        combined_df = pd.concat([train_df, test_df], ignore_index = True)
        
        min_samples = self.retraining_config['min_training_samples']
        if len(combined_df) < min_samples:
            raise ValueError(f"Insuffiecient training samples: {len(combined_df)} < {min_samples}")
        
        train_size = int(0.8 * len(combined_df))
        new_train_df = combined_df.iloc[:train_size]
        new_test_df = combined_df.iloc[train_size:]
        
        print(f"Collected {len(new_train_df)} training samples")
        print(f"Collected {len(new_test_df)} test samples")
        
        return new_train_df, new_test_df

    def save_training_data(self, train_df, test_df):
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_dir = f"data/version/v_{timestamp}"
        os.makedirs(version_dir, exist_ok = True)
        
        train_path = os.path.join(version_dir, 'train_data.csv')
        test_path = os.path.join(version_dir, 'test_data.csv')
        
        train_df.to_csv(train_path, index = False)
        test_df.to_csv(test_path, index = False)
        
        train_df.to_csv('data/raw/train_data.csv', index = False)
        test_df.to_csv('data/raw/test_data.csv', index = False)
        
        print(f"Saved training data to {version_dir}")
        
        return version_dir
