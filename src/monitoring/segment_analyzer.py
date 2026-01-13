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
    
    