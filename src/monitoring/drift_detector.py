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
    
    