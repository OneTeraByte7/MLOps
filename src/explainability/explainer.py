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
    
    def        