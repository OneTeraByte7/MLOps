import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import json
from datetime import datetime
import os

class ABTestAnalyzer:
    def __init__(self, alpha = 0.05, power = 0.8, min_sample_size = 1000):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        
    
    def calculate_sample_size(self, baseline_rate: float, mde: float = 0.05) -> int:
        
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        
        h = 2 * (np.arcsin(np.sqrt(p1))) - np.arcsin(np.sqrt(p2))
        
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_beta = norm.ppf(self.power)
        
        n = ((z_alpha + z_beta) / h) ** 2
        
        return int(np.ceil(n))
    
    def proportions_ztest(self, successes_a: int, n_a: int, successes_b: int, n_b: int) -> Dict:
        
        p_a = successes_a / n_a
        p_b = successes_b / n_b
        
        p_pool = (successes_a + successes_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z_score = (p_b - p_a) / se
        p_value =  2 * (1 - stats.norm.cdf(abs(z_score)))
        
        ci_se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        ci = stats.norm.ppf(1 - self.alpha / 2) * ci_se
        diff = p_b - p_a
        ci_lower = diff - ci
        ci_upper = diff + ci
        
        return{
            'variant_a_rate': p_a,
            'variant_b_rate': p_b,
            'difference': diff,
            'relative_difference': diff / p_a if p_a > 0 else 0,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': 1 - self.alpha
        }
        
    def sequential_test(self, results_a: list, results_b: list) -> Dict:
        
        n_a = len(results_a)
        n_b = len(results_b)
        
        if n_a < self.min_sample_size or n_b < self.min_sample_size:
            return {
                'decision': 'continue',
                'message': f'Need {self.min_sample_size} samples per variant (have {n_a}/{n_b})'
            }
            
        p_a = np.mean(results_a)
        p_b = np.mean(results_b)
        
        successes_a = sum(results_a)
        successes_b = sum(results_b)
        
        test_result = self.proportions_ztest(successes_a, n_a, successes_b, n_b)
        
        alpha_spending = self.alpha / 5
        
        if test_result['p_value'] < alpha_spending:
            if p_b > p_a:
                decision = 'B wins'
            else:
                decision = 'A wins'
                
        elif test_result['p_value'] > 1 - alpha_spending:
            decision = 'no_difference'
        else:
            decision = 'continue'
            
        return{
            'decision': decision,
            'samples_a': n_a,
            'samples_b': n_b,
            'rate_a': p_a,
            'rate_b': p_b,
            'p_value': test_result['p_value'],
            'confidence_interval': (test_result['ci_lower'], test_result['ci_upper']),
            'message': self._get_decision_message(decision, test_result)
        }