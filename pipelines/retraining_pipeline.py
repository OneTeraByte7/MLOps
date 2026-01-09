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