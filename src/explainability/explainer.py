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