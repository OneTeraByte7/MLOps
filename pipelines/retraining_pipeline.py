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
    
    def train_new_model(self):
        print("\n" + "=" * 60)
        print("TRAINING NEW MODEL")
        print("=" * 60)
        
        model, metrics = self.trainer.run_training_pipeline()
        
        return model, metrics
    def validate_new_model(self, new_metrics):
        print("\n" + "=" * 60)
        print("VALIDATING NEW MODEL")
        print("=" * 60)
        
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        runs = mlflow.search_runs(
            order_by = ["metrics.test_auc DESC"],
            max_results = 2
        )
        
        if len(runs) < 2:
            print("First model - automatically approved")
            return True, "First model trained"
        
        previous_auc = runs.iloc[1]['metrics.test_auc']
        new_auc = new_metrics['test_auc']
        
        min_drop = self.retraining_config['min_perfromnace_drop']
        
        improvement = new_auc = previous_auc
        
        print(f"Previous best AUC: {previous_auc:.4f}")
        print(f"new Mpdel AUC:     {new_auc:.4f}")
        print(f"Change:            {improvement:+.4f}")
        
        if improvement >= 0:
            print("New model is better = APPROVED")
            return True, f"Improved by ({improvement:.4f}"
        
        elif abs(improvement) < min_drop:
            print("Performance drop within tolerance - APPROVED")
            return True, f"Within tolerance ({improvement:.4f})"
        
        else:
            print("New model significantly worse - REJECTED")
            return False, f"Performance dropped by {abs(improvement):.4f}"
        
    
    def deploy_new_model(self, approved, reason):
        if not approved:
            print("\n Model deployment REJECTED")
            print(f"Reason: {reason}")
            return False

        print("\n" + "=" * 60)
        print("DEPLOYING NEW MODEL")
        print("=" * 60)
        
        
        print("Model promoted to production")
        print(f"Reason: {reason}")
        
        deployment_log = {
            'timestamp': datetime.now().isoformat(),
            'approved': approved,
            'reason': reason,
            'deployment_strategy': self.config['deployment']['strategy']
        }
        
        log_dir = 'monotoring/deployemnt_logs'
        os.makedirs(log_dir, exist_ok = True)
        
        log_path = os.path.join(log_dir, f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(log_path, 'w') as f:
            json.dump(deployment_log, f, indent = 2)
            
        return True
    
    def run_retraining_pipeline(self, force = False):
        print("\n" + "=" * 60)
        print("AUTOMATED RETRANING PIPELINE")
        print("=" * 70)
        print(f"started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not force:
            should_retrain, triggers, reasons = self.check_retraining_triggers()
            
            print("\n Retraining Triggers:")
            for trigger, status in triggers.items():
                print(f" {trigger:<25}{'YES' if status else 'NO'}")
                
                
            if reasons:
                print("\n Reasons:")
                for reaosns in reasons:
                    print(f" - {reasons}")
                    
            if not should_retrain:
                print("\n No retraining needed at this time")
                return False
            
            else:
                print("\n FORCED retraining (manual trigger)")
                
            try:
                train_df, test_df = self.collect_training_data()
                
                version_dir = self.save_training_data(train_df, test_df)
                
                model, metrics = self.train_new_model()
                
                approved, reason = self.validate_new_model(metrics)
                
                deployed = self.deploy_model(approved, reason)
                
                print("\n" + "=" * 70)
                if deployed:
                    print("RETRAINING PIPELINE COMPLECTED SUCCESSFULLY")
                    
                else:
                    print("RETRAINING PIPELINE COMPLETED (Model not deployed)")
                
                print("=" * 70)
                
                return deployed
            except Exception as e:
                print(f"RETRAINING PIPELINE FAILED")
                print(f" Eroor: {str(e)}")
                
                error_log = {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'status': 'failed'
                }
                
                log_dir = 'monitoring/errors'
                os.make_dirs(log_dir, exist_ok = True)
                
                log_path = os.path.join(log_dir, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json')")
                
                with open(log_path, 'w') as f:
                    json.dump(error_log, f, indent = 2)
                    
                raise
            
            
            
        if __name__ == '__main__':
            import argparse
                
            parser = argparse.ArgumentParser(description = 'Automated Retraining Pipeline')
            parser.add_argumnet('--force', action = 'store_true', help = 'Force retraining regardless of triggers')
            args = parser.parse_args()
            
            pipeline = RetrainingPipeline()
            pipeline.run_retraining_pipeline(force=args.force) 