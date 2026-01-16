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
        
    def _get_decision_message(self, decision: str, test_result: Dict) -> str:
        
        if decision == 'continue':
            return "Continue collecting data. No conclusive result yet"
        
        elif decision == 'no difference':
            return "No significant difference detected. Variants perform similarity"
        
        elif decision == 'B wins':
            rel_improvement = test_result['relative_difference'] * 100
            return f"Variant B wins with {rel_improvement:+.1f}% improvement (p = {test_result['p_value']:.4f})"
        
        else:
            rel_improvement = -test_result['relative_difference'] * 100
            return f"Variant A wins with {rel_improvement:+.1f}% better performance (p = {test_result['p_value']:.4f})"
        
        
    def bayesian_test(self, successes_a: int, n_a: int, successes_b: int, n_b: int, prior_alpha: float = 1, prior_beta:float = 1) -> Dict:
        
        alpha_a = prior_alpha + successes_a
        beta_a = prior_beta + (n_a - successes_a)
        
        alpha_b = prior_alpha + successes_b
        beta_b = prior_beta + (n_b  -successes_b)
        
        samples = 100000
        posterior_a = np.random.beta(alpha_a, beta_a, samples)
        posterior_b = np.random.beta(alpha_b, beta_b, samples)
        
        prob_b_better = np.random(posterior_b > posterior_a)
        
        loss_if_choose_a = np.mean(np.maximum(posterior_b - posterior_a, 0))
        loss_if_choose_b = np.mean(np.maximum(posterior_a - posterior_b, 0))
        
        ci_a = np.percentile(posterior_a, [2.5, 97.5])
        ci_b = np.percentile(posterior_b, [2.5,97.5])
        
        expected_a = alpha_a / (alpha_a + beta_a)
        expected_b = alpha_b / (alpha_b + beta_b)
        
        return {
            'prob_b_better_than': prob_b_better,
            'expected_rate_a': expected_a,
            'expected_rate_b': expected_b,
            'credible_interval_a': ci_a.tolist(),
            'credible_interval_b': ci_b.tolist(),
            'expected_loss_if_choose_a':loss_if_choose_a,
            'expected_loss_if_choose_b': loss_if_choose_b,
            'recommendation': 'B' if prob_b_better > 0.95 else 'A' if prob_b_better < 0.05 else 'Continue'
        }
        
    def multi_metric_test(self, metrics_a: Dict[str, list],
                          metrics_b: Dict[str, list]) -> Dict:
        
        n_metrics = len(metrics_a)
        adjusted_alpha = self.alpha / n_metrics
        results = {}
        
        for metric_name in metrics_a.keys():
            data_a = metrics_a[metric_name]
            data_b = metrics_b[metric_name]
            
            t_stat, p_value = stats.ttest_ind(data_a, data_b)
            
            mean_a = np.mean(data_a)
            mean_b = np.mean(data_b)
            
            results[metric_name] = {
                'mean_a': mean_a,
                'mean_b': mean_b,
                'difference': mean_b - mean_a,
                'relative_difference': (mean_b - mean_a) / mean_a if mean_a != 0 else 0,
                't_stattistics': t_stat,
                'p_value': p_value,
                'significant': p_value < adjusted_alpha,
                'adjusted_alpha': adjusted_alpha
            }
            
        return results
    def generate_report(self, test_results: Dict,
                        output_path: str = 'monitoring_reports/ab_test_results.json'):
        
        report = {
            'timestamp': datetime.now().iosformat(),
            'test_parameter': {
                'alpha': self.alpha,
                'power': self.power,
                'min_sample_size': self.min_sample_size
            },
            'results': test_results
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent = 2)
            
        print(f"A/B test report saved to {output_path}")
        
        return report
    
    def print_results(self, results: Dict):
        print("\n " + "=" * 60)
        print("AB Test results")
        print("=" * 60)
        
        if 'variant_a_rate' in results:
            print(f"\n Variant A; {results['variant_a_rate']:.4f}")
            print(f"\n Variant B: {results['variant_b_rate']:.4f}")
            print(f"\n Difference: {results['difference']:.4f} ({results['relative_difference']:+1%})")
            print(f"P-value: {results['p_value']:.4f}")
            print(f"95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
            
            if results['significant']:
                winner = 'B' if results['difference'] > 0 else 'A'
                print(f"Significant: Variant {winner} wins !")
                
            else:
                print(f"Not Significant: No clear winner yet")
                
        elif 'pron_b_better_than_a' in results:
            print(f"\n Expected Rate A: {results['expected_tae_a']:.4f}")
            print(f"Expected Rate B: {results['expected_rate_b']:.4f}")
            print(f"P(B > A): {results['prob_b_better_than_a']:.1%}")
            print(f"Recommendation: {results['recommendation']}")
            
        print("=" * 60)
        
    
if __name__ == '__main__':
    ab_test = ABTestAnalyzer(alpha = 0.05, power = 0.8)
    
    print("\n Example 1:Required Sample Size")
    print("-" * 40)
    
    baseline_accuracy = 0.85
    min_detectable_effect = 0.05
    
    required_n = ab_test.calculate_sample_size(baseline_accuracy, min_detectable_effect)
    print(f"Baseline accuracy: {baseline_accuracy:.1%}")
    print(f"Want to detect: {min_detectable_effect:+.1%} change")
    print(f"Required sample size: {required_n}")
    
    print("\n\n Example 2: Frequentist Z-Test")
    print("-" * 40)
    
    
    results_freq = ab_test.proportions_ztest(
        successes_a = 850, n_a = 1000,
        success_b = 870, n_b = 1000
    )
    
    ab_test.print_results(results_freq)
    
    print("\n\n Example 3: Sequential Test (Peek at Results)")
    print("-" * 40)
    
    np.random.seed(42)
    results_a = np.random.binomial(1, 0.85, 500).tolist()
    results_b = np.random.binomial(1, 0.87, 500).tolist()
    
    seq_result = ab_test.sequential_test(results_a, results_b)
    
    print(f"Decision: {seq_result['decision']}")
    print(f"Message: {seq_result['message']}")
    print(f"Sample collected: A = {seq_result['sample_a']}, B = {seq_result['sample_b']}")
    
    print("\n\n Example 4: Bayesian Test")
    print("-" * 40)
    
    results_bayes = ab_test.bayesian_test(
        successes_a = 850, n_a = 1000,
        successes_b = 870, n_b = 1000
    )
    
    ab_test.print_results(results_bayes)
    
    print("\n\n Example 5: Multi-Metric Testing")
    print("-" * 40)
    
    metrics_a = {
        'accuracy': np.random.normal(0.85, 0.02, 1000).tolist(),
        'latency_ms': np.random.normal(50, 10, 1000).tolist(),
        'precision': np.random.normal(0.82, 0.03, 1000).tolist()
    }
    
    metrics_b = {
        'accuracy': np.random.normal(0.87, 0.02, 1000).tolist(),
        'latency_ms': np.random.normal(48, 10, 1000).tolist(),
        'precision': np.random.normal(0.84, 0.03, 1000).tolist()
    }
    
    multi_results = ab_test.multi_metric_test(metrics_a, metrics_b)
    
    print("\n Multi-Metric Results:")
    for metric, result in multi_results.items():
        print(f"\n {metric}:")
        print(f" A: {result['mean_a']:.4f}")
        print(f" B: {result['mean_b']:.4f}")
        print(f" Change: {result['relative_difference']:+.1%}")
        print(f" Significant: {'Yes' if result['significant'] else 'No'}")
        
        
    print("\n" + "=" * 60)
    print("Key Benefits of Statistical A/B Testing")
    print("=" * 60)
    print("Prevents premature conclusion")
    print("Controls false positive rate")
    print("Provides confidence intervals")
    print("Supports early stopping (SPRT)")
    print("Handles multiple metrics correctly")
    print("=" * 60)