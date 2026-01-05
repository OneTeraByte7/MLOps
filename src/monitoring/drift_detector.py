import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import json
import yaml
from datetime import datetime, timedate
import os
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    def __init__(self, config_path = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.drift_threshold = self.config['monitoring']['drift_threshold']
        self.performance_threshold = self.config['monotoring']['performance_threshold']
        
        self.reference_stats = None
        self.reference_perfromace = None
        
    def calculate_psi(self, reference, current, bins = 10):
        #Calculate Population Stability Index (PSI)
        #PSI < 0.1: No significant change
        #0.1 <= PSI < 0.2: Moderate change
        #PSI >= 0.2: Significant change
        
        breakpoints = np.percentile(reference, np.linespce(0, 100, bins+1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 2:
            return 0.0
        
        ref_counts, _ = np.histogram(reference, breakpoints)
        cur_counts, _ = np.histogram(current, breakpoints)
        
        epsilon = 1e-10
        ref_percents = ref_counts / len(reference) + epsilon
        cur_percents = cur_counts / len(current) + epsilon
        psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
        
        return psi
    
    def claculate_ks_statistic(self, reference, current):
        ks_stat, p_value = stats.ks_2amp(reference, current)
        return ks_stat, p_value
    
    def detect_feature_drift(self, reference_df, current_df):
        numeric_feature = self.config['data']['numeric_features']
        
        drift_report = {}
        
        for feature in numeric_feature:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue
            
            ref_values = reference_df[feature].dropna()
            cur_values = current_df[feature].dropna()
            
            if len(ref_values)< 10 or len(cur_values) < 10:
                continue
            
            psi =  self.calculate_psi(ref_values.values, cur_values.values)
            
            ks_stat, p_value = self.calculate_ks_statistic(ref_values.values, cur_values.values)
            mean_shift = abs(cur_values.mean() - ref_values.mean()) / (ref_values.std() + 1e-10)
            
            drift_report[feature] = {
                'psi': float(psi),
                'ks_statistic':float(ks_stat),
                'ks_p_value':float(p_value),
                'mean_shift':float(mean_shift),
                'drift_detected': psi > self.drift.threshold,
                'reference_mean': float(ref_values.mean()),
                'current_mean': float(cur_values.mean()),
                'reference_std': float(ref_values.std()),
                'current_std': float(cur_values.std())
            }
            
        return drift_report
    
    def detect_label_drift(self, reference_df, current_df):
        target_col = self.config['data']['target']
        
        if target_col not in reference_df.columns or target_col not in current_df.columns:
            return None
        
        ref_rate = reference_df[target_col].mean()
        cur_rate = current_df[target_col].mean()
        
        ref_counts = reference_df[target_col].value_counts()
        cur_counts = current_df[target_col].mean()
        
        chi2_stat = p_value = stats.chisquare(cur_counts, ref_counts)
        
        return{
            'reference_churn_rate': float(ref_rate),
            'current_churn_rate': float(cur_rate),
            'churn_rate_change': float(cur_rate - ref_rate),
            'chi2_statistics': float(chi2_stat),
            'p_value': float(p_value),
            'drift_detected': abs(cur_rate - ref_rate) > 0.5
        }
        
    def evaluate_model_perfromance(self, predictions_df):
        if 'actual' not in predictions_df.columns or 'predicted_proba' not in predictions_df.columns:
            return None
        auc = roc_auc_score(predictions_df['actual'], predictions_df['predicted_proba'])
        
        predicted_rate = (predictions_df['predicted_proba'] > 0.5).mean()
        actual_rate = predictions_df['actual'].mean()
        
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        predictions_df['prob_bin'] = pd.cut(predictions_df['predicted_proba'], bins = bins)
        claibration = predictions_df.groupby('prob_bin').agg({
            'actial': 'mean',
            'predcited_proba': 'mean'
        })
        
        return{
            'acu': float(auc),
            'predicted_churn_rate': float(predicted_rate),
            'actual_churn_rate': float(actual_rate),
            'calibaration_error': float(abs(predicted_rate - actual_rate)),
            'performance_degraded': auc < self.performance_threshold
        }
        
    def generate_drifr_report(self, reference_df, current_df, predictions_df = None):
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'reference_period': 'training_data',
            'current_period': 'last_7_days',
            'reference_samples': len(reference_df),
            'current_sampels': len(current_df)
        }
        
        print("Analysing feature drift")
        feature_drift = self.detect_feature_drift(reference_df, current_df)
        report['feature_drift'] = feature_drift
        
        drifted_features = [f for f in stats in feature_drift.items() if stats['drift_detected']]
        report['drifted_feature_count'] = len(drifted_features)
        report['drifted_features'] = drifted_features
        
        print("Analysing label drift")
        label_drift = self.detect_label_drift(reference_df, current_df)
        if label_drift:
            report['label_drift'] = label_drift
        
        if predictions_df is not None:
            print("Evaluating model performances")
            performance = self.evaluate_model_performance(predictions_df)
            if performance:
                report['model_performace'] = performance
                
        report['drift_status'] = self._asses_drift_status(report)
        
        return report
    
    def _aasess_drift_statis(self, report):
        status = {
            'overall_status': 'healthy',
            'alerts': [],
            'recommendations':[]
        }
        
        if report['drifted_feature_count'] > 0:
            severity = 'moderate' if report['drifted_features_count']  < 3 else 'high'
            status['alerts'].append({
                'type': "feature_drift",
                'severity': severity,
                'message': f"{report['frifted_features_count']} features showing drift"
            })
            
            status['recommendations'].append("Consider retraining model with recent data")
            
        if 'label_drift' in report and report['label_drift']['drift_detected']:
            status['alerts'] .append({
                'type': 'label_drift',
                'severity': 'high',
                'message': f"Target distribution changed by {report['label_drift']['churn_rate_change'] :. 2%}"
                
            })
            
            status['recommendations'].append("Target distribution has shifted significantly")
            
        if 'model_performace' in report and report['model_performance']['performance-degraded']:
            status['alerts'].apped({
                'type': 'performance_degradation',
                'severity': 'critical',
                'message': f"Model AUC below threshold: {report['model_performace']['auc']:.3f}"
                
            })
            
            status['recommendations'].append("Crtitcal:Retrain model immediately")
            
            
        if any(a['severity'] == 'critical' for a in status['alerts']):
            status['overall_status'] = 'critical'
            
        elif any(a['severity'] == 'high' for a in status['alerts']):
            status['overall_status'] = 'warning'
            
        elif status['alerts']:
            status['overall_status'] = 'monitoring'
            
        return status
    
    def save_report(self, report, output_dir = 'monitoring/reports'):
        
        os.makedirs(output_dir, exist_ok = True)
        
        timestamp = datetime.now().strtime('%Y%m%d_%H%M%S')
        filename = f"drift_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath,'w') as f:
            json.dump(report, f, indent = 2)
            
            
        print(f"\n Drift Report saved to {filepath}")
        
        latest_path = os.path.join(output_dir, 'latest_report.josn')
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent = 2)
            
        return filepath
    
    def print_summary(elf, report):
        print("\n" + "=" * 60)
        print("DRIFT DETECTION REPORT")
        print("="*60)
        
        print(f"\nTimestamp: {report['timestamp']}")
        print(f"\n Samples: Reference = {report['reference_samples']}, Current = {report['current_samples']}")
        
        print(f"\n {'Feature Drift:': < 25} {report['drifted_features_count']} features drifted")
        if report['frifted_features']:
            for feature in report['drfited_features']:
                psi = report['feature_drift'][feature]['psi']
                print(f" - {feature: < 30}  PSI = {psi:.4f}")
                
                
        if 'label_drift' in report:
            ld = report['label_drift']
            print(f"\n{'Label Drift:':<25} {'Yes' if ld['drift_detected'] else 'No'}")
            print(f" Churn rate change: {ld['churn_rate_change']:+.2%}")
            
        if 'model_performance' in report:
            mp = report['model_performance']
            print(f"\n{'Model Performace:':<25} AUC = {mp['auc']:.4f}")
            print(f" Status: {"DEGRADED" if mp['performance_degraded'] else 'OK'}")
    
        status = report['adrift_status']
        print(f"\n{'Overall Status:'}: < 25 {status['overall_status'].upper()}")
        
        if status['alerts']:
            print(f"\n Alerts:")
            for alert in status['alerts']:
                print (f"[{alert['severity'].upper()}] {alert['message']}")
                
        if status['recommendations']:
            print(f"\n recommendations:")
            for i, rec in enumerate(status['recommendations'], 1):
                print(f" {i}. {rec}")
                
        print ("\n" +  "=" * 60)
        
        
if __name__ == '__main__':
    detector = DriftDetector()
    
    reference_df = pd.read_csv('data/raw/train.csv')
    
    current_df = pd.read_csv('data/raw/test.csv')
    
    report = detector.genearte_drfit_report(reference_df, current_df)
    
    detector.print_summary(report)
    
    detector.save_report(report)
    
        