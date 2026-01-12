import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ChurnExplainer:
    #Gives explanation for churn predictions
    
    def __init__(self, model_path = 'models/model.json', feature_names_path = 'models/feature_names.json'):
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
            
        self.explainer = shap.TreeExplainer(self.model)
        
        print(f"Explainer initialized with {len(self.feature_names)} features")
        
    def explain_predictions(self,  X: np.ndarray, customer_id: str = None) -> Dict:
        
        shap_values = self.explainer.shap_values(X)
        base_value  = self.explainer.expected_value
        dmatrix = xgb.DMatrix(X)
        prediction = self.model.predict(dmatrix)[0]
        
        feature_contributions = self._get_feature_contributions(X[0], shap_values[0])
        
        top_drivers = self._get_top_drivers(feature_contributions, top_k = 5)
        
        explanation_text = self._generate_explanation_text(
            prediction,
            top_drivers,
            customer_id
        )
        
        return{
            'customer_id': customer_id,
            'prediction': float(prediction),
            'base_value': float(base_value),
            'feature_contribution': feature_contributions,
            'top_drivers': top_drivers,
            'explanation_text': explanation_text,
            'shap_values': shap_values[0].tolist(),
            'feature_values': X[0].tolist()
        }
        
    def explain_batch(self, X: np.ndarray, customer_ids: List[str] = None) -> List[Dict]:
        
        if customer_ids is None:
            customer_ids = [f"customer_{i}" for i in range(len(X))]
            
        explanations = []
        for i, cust_id in enumerate(customer_ids):
            exp = self.explain_predictions(X[i:i+1], cust_id)
            explanations.append(exp)
            
        return explanations

    def _get_feature_contribution(self, feature_values: np.ndarray,
                                  shap_values: np.ndarray) -> List[Dict]:
        
        contributions = []
        for i, (feat_name, feat_val, shap_val) in enumerate(
            zip(self.feature_names, feature_values, shap_values)
        ):
            contributions.append({
                'feature': feat_name,
                'value': float(feat_val),
                'impact': float(shap_val),
                'abs_impact': abs(float(shap_val))
            })
            
        contributions.sort(key=lambda x: x['abs_impact'], reverse = True)
        
        return contributions
    
    def _get_top_drivers(self, contributions: List[Dict], top_k: int = 5) -> List[Dict]:
        
        top_drivers = []
        for contrib in contributions[:top_k]:
            direction = "increase" if contrib['impact'] > 0 else "decreases"
            
            top_drivers.append({
                'feature': contrib['feature'],
                'value': contrib['value'],
                'impact': contrib['impact'],
                'direction': direction,
                'importance_rank': len(top_drivers) + 1
            })
            
        return top_drivers
    
    def _generate_explanation_text(self, prediction: float,
                                top_drivers: List[Dict],
                                customer_id: str = None) -> str:
        
        risk_level = "HIGH" if prediction > 0.7 else "MEDIUM" if prediction > 0.3 else "LOW"
        explanation = f"Churn Risk: {risk_level} ({prediction:.1%})\n\n"
        
        if customer_id:
            explanation += f"Customer {customer_id} analysis: \n\n"
            
        explanation += "Top factors influencing this prediction: \n\n"
        
        for i, driver in enumerate(top_drivers, 1):
            feature = driver['feature'].replace('_', ' ').title()
            impact_pct = abs(driver['impact']) * 100
            direction = driver['direction']
            
            explanation += f"{i}. {feature} (value: {driver['value']:.2f})\n"
            explanation += f"{direction.capitalize()} churn risk by {impact_pct:.1f}%\n\n"
        
        explantion += self._generate_recommendations(prediction, top_drivers)
        
        return explanation
    
    def _generate_recommendations(self, prediction: float, top_drivers: List[Dict]) -> str:
        
        recommendations = "\n Recommended Actions: \n"
        
        if prediction > 0.7:
            recommendations += " URGENT: High churn risk detetcted \n"
            
            for driver in top_drivers[:3]:
                feat = driver['feature']
                
                if 'days_since_last_login' in feat and driver['impact'] > 0:
                    recommendations += " # Customer is inactive - send re-engagement email \n"
                    
                if 'support_tickets' in feat and driver['impact'] > 0:
                    recommendations += " # High support burden - schedule customer success call \n"
                    
                if 'nps_score' in feat and driver['impact'] > 0:
                    recommendations += "# Low satisfaction - offer product training session \n"
                    
                if 'payment_delays' in feat and driver['impact'] > 0:
                    recommendations += " # Payment issues - contact billing team\n"
                    
                if 'feature_usuage' in feat and driver['impact'] > 0:
                    recommendations += " # Low engagement - provide feature onboarding \n"
                    
        elif prediction > 0.3:
            recommendations += " # Monitor closely ad consider preventive outreach \n"
            
        else:
            recommendations += " # Customer appears healthy - maintain regular touchpoints \n"
            
        return recommendations
    
    def generate_global_importance(self, X_sample: np.ndarray,
                                   output_path: str = 'monitoring/reports/feature_importance.png'):
    
        shap_values = self.explainer.shap_values(X_sample)
        
        plt.figure(figsize = (10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names = self.feature_names, show = False, max_display = 20)
        
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        plt.tight_layout()
        plt.savefig(output_path, dpi = 150, bbox_inches = 'tight')
        plt.close()
        
        print(f" GLobal importance plot saved to {output_path}")
        
    def generate_force_plot(self, X: np.ndarray, customer_id: str = None,
                           output_path: str = 'monitoring/reports/force_plot.html'):
        
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value
        
        force_plot = shap.force_plot(
            base_value,
            shap_values[0],
            X[0],
            feature_names = self.feature_names,
            matplotlib = False,
        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        shap.save_html(output_path, force_plot)
        
        print(f"Force plot saved to {output_path}")
        
    
    def generate_waterfall_plot(self, X:np.ndarray, customer_id: str = None,
                                output_path: str = 'monitoring/reports/waterfall_plot.png'):
        
        shap_values = self.explainer.shap_values(X)
        base_value = self.explainer.expected_value
        explanation = shap.Explanation(
            values = shap_values[0],
            base_values = base_value,
            data = X[0],
            feature_names = self.feature_names
        )
        
        plt.figure(figsize = (10, 8))
        shap.waterfall_plot(explanation, max_display = 15, show=False)
        
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        plt.tight_layout()
        plt.savefig(output_path, dpi = 150, bbox_inches = 'tight')
        plt.close()
        
        print(f"Waterfall plot saved to {output_path}")
        
    def generate_dependance_plot(self, X_sample: np.ndarray, feature_name: str,
                                 output_path: str = None):
        
        if feature_name not in self.feature_names:
            print(f"feature '{feature_name}' not found")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        shap_values = self.explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 6))
        shap.dependance_plot(
            feature_idx,
            shap_values,
            X_sample,
            feature_names = self.feature_names,
            show=False
        )
        
        if output_path is None:
            output_path = f'monitoring/reports/dependance_{feature_name}.png'
            
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        plt.tight_layout()
        plt.savefig(output_path, dpi = 150, bbox_inches = 'tight')
        plt.close()
        
        print(f"dependance plot saved to {output_path}")
        
    def explain_model_globally(self, X_sample: np.ndarray,
                               output_dir: str = 'monitorinf.reports/explanability'):
        
        print("\n" + "=" * 60)
        print("GENERATING GLOBAL MODEL EXPLANATION")
        print("=" * 60)
        
        os.makedirs(output_dir, exist_ok = True)
        
        print("\n 1. Creating Summary Plot")
        self.generate_global_importance(
            X_sample,
            f"{output_dir}/global_importance.png"
        )
        
        print("\n 2. Calculating feature importance values")
        shap_values = self.explainer.shap_values(X_sample)
        feature_importance = np.abs(shap_values).mean(axis = 0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending = False)
        
        importance_df.to_csv(f"{output_dir}/feature_importance.csv", idex = False)
        
        
        
        print("\n 3, Creating dependence plots for top 5 features")
        top_features = importance_df.head(5)['feature'].tolist()
        
        for feat in top_features:
            self.generate_dependance_plot(
                X_sample,
                feat,
                f"{output_dir}/dependence_{feat}.png"
            )
            
        print("\n" + "=" * 60)
        print("GLOBAL EXPLANATION COMPLETE")
        print(f"Reports saved to {output_dir}/")
        print("=" * 60)
        
        return importance_df
    

if __name__ == "__main__":
    import sys
    sys.append('src')
    from features.feature_engineering import FeatureEngineer
    
    test_df = pd.read_csv('data/raw/test_data.csv')
    
    fe = FeatureEngineer()
    fe.load_preprocessor('models/preprocessor.pkl')
    X_test, Y_test, _ = fe.transform(test_df)
    
    explainer = ChurnExplainer()
    
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Single Customer Explanation")
    print("=" * 60)
    
    customer_idx = 0
    explanation = explainer.explain_prediction(
        X_test[customer_idx:customer_idx+1],
        customer_id = test_df.iloc[customer_idx]['customer_id']
        )
    
    print(explanation['explanation_text'])
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Generating Visualization")
    print("="*60)
    
    high_risk_idx = np.where(explainer.model.predict(xgb.DMatrix(X_test)) > 0.7)[0]
    if len(high_risk_idx) > 0:
        idx = high_risk_idx[0]
        explainer.generate_force_plot(
            X_test[idx:idx+1],
            test_df.iloc[idx]['customer_id']
        )
        
        explainer.generate_waterfall_plot(
            X_test[idx:idx+1],
            test_df.iloc[idx]['customer_id']
        )
        
    print("\n" + "=" * 60)
    print("EXAMPLE 3: GLobal Model Analysis")
    print("="*60)
    
    sample_size = min(1000, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace = False)
    X_sample = X_test[sample_indices]
    
    importance_df = explainer.explain_model_globally(X_sample)
    
    print("\n Top 10 Important Features: ")
    print(importance_df.head(10).to_string(index = False))