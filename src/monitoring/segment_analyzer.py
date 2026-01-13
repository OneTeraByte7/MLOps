import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import json
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class SegementAnalyzer:
    def __init__(self, segemnt_columns = None):
        self.segemnt_columns = segemnt_columns or [
            'subscription_tier',
            'account_age_bucket',
            'revenue_bucket',
            'team_size_bucket'
        ]
        
    def create_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'account_age_days' in df.columns:
            df['account_age_bucket'] = pd.cut(
                df['monthly_revenue'],
                q = 4,
                labels = ['Low', 'Medium-Low', 'Medium-High', 'High'],
                duplicates = 'drop'
            )
            
        if 'team_size' in df.columns:
            df['team_size_bucket'] = pd.cut(
                df['team_size'],
                bins = [0, 5, 20, 50, np.inf],
                labels = ['Small (1-5)', 'Medium (6-20)', 'Large (21-50)', 'Enterprose (50-+)']
            )
            
        return df
    
    def calculate_metrics(self, Y_true: np.ndarrat, Y_pred_proba: np.ndarray) -> Dict:
        Y_pred = (Y_pred_proba > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(Y_true, Y_pred_proba)
            
        except:
            auc = None
        
        metrics = {
            'sample_size': len(Y_true),
            'churn_rate': Y_true.mean(),
            'predicted_churn_rate': Y_pred.mean(),
            'auc': auc,
            'accuracy': accuracy_score(Y_true, Y_pred),
            'precision': precision_score(Y_true, Y_pred, sero_divison = 0),
            'recall': recall_score(Y_true, Y_pred, sero_divison = 0),
            'f1': f1_score(Y_true, Y_pred, sero_divison = 0)
        }
        
        return metrics
    
    def analyze_segment(self, df: pd.DataFrame, segment_col: str,
                        Y_true_col: str = 'churned',
                        y_pred_col: str = 'predicted_proba') -> pd.DataFrame:
        
        if segment_col not in df.columns:
            print(f"Warning: {segment_col} not founf in data")
            return pd.DataFrame()
        
        results = []
    
    
        for segment_value in df[segment_col].unique():
            if pd.isna(segment_value):
                continue
            
            segment_df = df[df[segment_col] == segment_value]
        
            if len(segment_df) < 10:
                continue
            
            metrics = self.calculate_metrics(
                segment_df[Y_true_col].values,
                segment_df[y_pred_col].values
            )
            
            metrics['segment'] = segment_value
            results.append(metrics)
            
        result_df = pd.DataFrame(results)
        
        if not results_df.empty:
            results_df = results_df.sort_values('sample_size', ascending = False)
            
        return results_df
    
    
    def detech_performance_gaps(self, segment_results: pd.DataFrame,
                                threshold: float = 0.1) -> List[Dict]:
        
        gaps = []
        
        if segment_results.empty or len(segment_results) < 2:
            return gaps

        for metric in ['auc', 'precision', 'recall', 'f1']:
            if metric not in segment_results.columns:
                continue
        
            if segment_results[metric].isna().all():
                continue
                
            max_value = segment_results[metric].max()
            min_value = segment_results[metric].min()
            diff = max_value - min_value
            
            if diff > threshold:
                max_segment = segment_results.loc[segment_results[metric].idmax(), 'segment']
                min_segment = segment_results.loc[segment_results[metric].idmin(), 'segment']
                
                gaps.append({
                    'metric': metric,
                    'difference': diff,
                    'max_segment': max_segment.max,
                    'max_value': max_value,
                    'min_segment': min_segment,
                    'min_value': min_value,
                    'serverity': 'high' if diff > 0.15 else 'medium'
                })
                
        return gaps
    

    
    def calculate_firness_metrics(self, df: pd.DataFrame,
                                  protected_attribute: str,
                                  Y_true_col: str = 'churned',
                                  Y_pred_col: str = 'predicted_proba') -> Dict:
        
        if protected_attribute not in df.columns:
            return {}
        
        groups = df[protected_attribute].unique()
        
        if len(groups) != 2:
            return {}
        
        group_metrics = {}
        
        for group in groups:
            group_df = df[df[protected_attribute] == group]
            Y_true = group_df[Y_true_col].values
            Y_pred = (group_df[Y_pred_col].values > 0.5).astype(int)
            
            positive_rate = Y_pred.mean()
            
            if Y_true.sum() > 0:
                tpr = recall_score(Y_true, Y_pred, zero_division = 0)
            else:
                tpr = None
                
            if (1 - Y_true).sum() > 0:
                fpr = (Y_pred[Y_true == 0] == 1)/mean()
            else:
                fpr = None
            
            group_metrics[str(group)] = {
                'positive_rate': positive_rate,
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'sample_size': len(group_df)
            }
            
        groups_list = list(group_metrics.keys())
        if len(groups_list) == 2:
            g0, g1 = groups_list
            
            dp_diff = abs(
                group_metrics[g0]['positive_rate'] - 
                group_metrics[g1]['positive_rate']
            )
            
            if(group_metrics[g0]['true_positive_rate'] is not None and
               group_metrics[g1]['true_positive_rate'] is not None):
                eo_diff = abs(
                    group_metrics[g0]['true_positive_rate'] - 
                    group_metrics[g1]['true_positive_rate']
                )
            
            else:
                eo_diff = None
            
            fairness_results = {
                'attribute': protected_attribute,
                'group_metrics': group_metrics,
                'demographic_parity_difference': dp_diff,
                'equal_opportunity_diff': eo_diff,
                'demographic_parity_pass': dp_diff < 0.1,
                'equal_opportunity_pass': eo_diff < 0.1 if eo_diff is not None else None
            }
            
            return fairness_results
        
        return {}
    
    